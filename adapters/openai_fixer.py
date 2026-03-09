"""OpenAI 格式修复

修复 Cursor 发出的 OpenAI 格式请求和上游返回的响应中的各种兼容性问题：
  请求修复: Cursor 扁平格式 tools → 标准嵌套格式, tool_choice 规范化
  响应修复: reasoningContent → reasoning_content, <think> 标签提取,
           function_call → tool_calls, tool_calls 字段补全, 参数修复
"""

import json
import logging

from utils.http import gen_id
from utils.tool_fixer import normalize_args, repair_str_replace_args
from utils.think_tag import extract_from_text

logger = logging.getLogger(__name__)


# ─── 请求预处理 ───────────────────────────────────


def normalize_request(payload, upstream_model=None):
    """预处理 Cursor 发来的 OpenAI 格式请求"""
    if upstream_model:
        payload['model'] = upstream_model

    # Cursor 可能在 CC 端点发送 Anthropic 格式的 tool_use/tool_result 消息
    if 'messages' in payload:
        payload['messages'] = _convert_anthropic_messages(payload['messages'])

    if 'tools' not in payload:
        return payload

    # 修复 Cursor 可能发出的扁平格式 tools
    normalized = []
    for tool in payload['tools']:
        if tool.get('type') == 'function' and 'function' in tool:
            normalized.append(tool)
        elif 'name' in tool:
            normalized.append({
                'type': 'function',
                'function': {
                    'name': tool.get('name', ''),
                    'description': tool.get('description', ''),
                    'parameters': tool.get('input_schema')
                                  or tool.get('parameters')
                                  or {'type': 'object', 'properties': {}},
                },
            })
        else:
            normalized.append(tool)
    payload['tools'] = normalized

    # tool_choice 规范化
    tc = payload.get('tool_choice')
    if isinstance(tc, dict):
        if tc.get('type') == 'auto':
            payload['tool_choice'] = 'auto'
        elif tc.get('type') == 'any':
            payload['tool_choice'] = 'required'

    return payload


def _convert_anthropic_messages(messages):
    """将消息中的 Anthropic 格式 tool_use/tool_result 转为 OpenAI 格式

    Cursor 有时在 CC 端点中发送 Anthropic 风格的内容块：
      assistant: [{"type":"tool_use", "id":"...", "name":"Read", "input":{...}}]
      user:      [{"type":"tool_result", "tool_use_id":"...", "content":[...]}]
    OpenAI 格式应为：
      assistant: {"tool_calls":[{"id":"...", "function":{"name":"Read","arguments":"..."}}]}
      tool:      {"tool_call_id":"...", "content":"..."}
    """
    converted = []
    for msg in messages:
        content = msg.get('content')
        if not isinstance(content, list):
            converted.append(msg)
            continue

        has_tool_use = any(
            isinstance(b, dict) and b.get('type') == 'tool_use' for b in content
        )
        has_tool_result = any(
            isinstance(b, dict) and b.get('type') == 'tool_result' for b in content
        )

        if not has_tool_use and not has_tool_result:
            converted.append(msg)
            continue

        role = msg.get('role', '')

        if role == 'assistant' and has_tool_use:
            text_parts = []
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                elif block.get('type') == 'tool_use':
                    tool_calls.append({
                        'id': block.get('id', gen_id('call_')),
                        'type': 'function',
                        'function': {
                            'name': block.get('name', ''),
                            'arguments': json.dumps(
                                block.get('input', {}), ensure_ascii=False
                            ),
                        },
                    })
            new_msg = {'role': 'assistant'}
            new_msg['content'] = '\n'.join(text_parts) if text_parts else None
            if tool_calls:
                new_msg['tool_calls'] = tool_calls
            converted.append(new_msg)

        elif has_tool_result:
            other_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'tool_result':
                    rc = block.get('content', '')
                    if isinstance(rc, list):
                        rc = '\n'.join(
                            b.get('text', '') for b in rc
                            if isinstance(b, dict) and b.get('type') == 'text'
                        )
                    elif not isinstance(rc, str):
                        rc = str(rc)
                    converted.append({
                        'role': 'tool',
                        'tool_call_id': block.get('tool_use_id', ''),
                        'content': rc,
                    })
                else:
                    other_parts.append(block)
            if other_parts:
                converted.append({'role': role, 'content': other_parts})
        else:
            converted.append(msg)

    return converted


# ─── 非流式响应修复 ───────────────────────────────


def fix_response(data):
    """修复上游返回的非流式 OpenAI 响应"""
    if not isinstance(data, dict):
        return data

    for choice in (data.get('choices') or []):
        msg = choice.get('message') or {}

        # reasoningContent → reasoning_content
        if 'reasoningContent' in msg and 'reasoning_content' not in msg:
            msg['reasoning_content'] = msg.pop('reasoningContent')

        # <think> 标签 → reasoning_content
        content = msg.get('content') or ''
        if isinstance(content, str) and '<think>' in content and not msg.get('reasoning_content'):
            cleaned, reasoning = extract_from_text(content)
            if reasoning:
                msg['reasoning_content'] = reasoning
                msg['content'] = cleaned
                logger.info(f'提取 <think> 标签 → reasoning_content ({len(reasoning)} 字符)')

        # 旧版 function_call → 新版 tool_calls
        if 'function_call' in msg and 'tool_calls' not in msg:
            fc = msg.pop('function_call')
            msg['tool_calls'] = [{
                'id': gen_id('call_'),
                'type': 'function',
                'function': {
                    'name': fc.get('name', ''),
                    'arguments': fc.get('arguments', '{}'),
                },
            }]
            if choice.get('finish_reason') == 'function_call':
                choice['finish_reason'] = 'tool_calls'

        # 修复 tool_calls 字段
        _fix_tool_calls(msg, choice)

    return data


# ─── 流式 chunk 修复 ──────────────────────────────


def fix_stream_chunk(data):
    """修复上游返回的流式 OpenAI chunk"""
    if not isinstance(data, dict):
        return data

    for choice in (data.get('choices') or []):
        delta = choice.get('delta') or {}

        # reasoningContent → reasoning_content
        if 'reasoningContent' in delta and 'reasoning_content' not in delta:
            delta['reasoning_content'] = delta.pop('reasoningContent')

        # 旧版 function_call → tool_calls
        if 'function_call' in delta and 'tool_calls' not in delta:
            fc = delta.pop('function_call')
            tc = {'index': 0, 'type': 'function', 'function': {}}
            if 'name' in fc:
                tc['id'] = gen_id('call_')
                tc['function']['name'] = fc['name']
            if 'arguments' in fc:
                tc['function']['arguments'] = fc['arguments']
            delta['tool_calls'] = [tc]
            if choice.get('finish_reason') == 'function_call':
                choice['finish_reason'] = 'tool_calls'

        # 补全 tool_calls 字段
        for tc in (delta.get('tool_calls') or []):
            if 'index' not in tc:
                tc['index'] = 0
            func = tc.get('function') or {}
            if 'id' in tc or 'name' in func:
                if not tc.get('id'):
                    tc['id'] = gen_id('call_')
                if 'type' not in tc:
                    tc['type'] = 'function'

        if choice.get('finish_reason') == 'function_call':
            choice['finish_reason'] = 'tool_calls'

    return data


# ─── 内部辅助 ─────────────────────────────────────


def _fix_tool_calls(msg, choice):
    """修复消息中的 tool_calls 字段"""
    tool_calls = msg.get('tool_calls')
    if not tool_calls:
        return

    for i, tc in enumerate(tool_calls):
        if not tc.get('id'):
            tc['id'] = gen_id('call_')
        if 'index' not in tc:
            tc['index'] = i
        if tc.get('type') != 'function':
            tc['type'] = 'function'

        func = tc.get('function', {})
        args_raw = func.get('arguments', '{}')
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}

        args = normalize_args(args)
        args = repair_str_replace_args(func.get('name', ''), args)
        func['arguments'] = json.dumps(args, ensure_ascii=False)

    if choice.get('finish_reason') not in ('tool_calls', 'function_call'):
        choice['finish_reason'] = 'tool_calls'

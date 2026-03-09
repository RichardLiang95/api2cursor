"""OpenAI Chat Completions ↔ Anthropic Messages 格式转换

请求方向: CC → Messages（Cursor 的 CC 请求转为 Anthropic 格式发给上游）
响应方向: Messages → CC（上游 Anthropic 响应转为 CC 格式返回给 Cursor）
包含非流式和流式两种转换。
"""

import json
import uuid
import logging

from utils.tool_fixer import normalize_args, repair_str_replace_args, fix_anthropic_tool_use
from utils.http import gen_id

logger = logging.getLogger(__name__)

# Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'stop_sequence': 'stop',
}


# ═══════════════════════════════════════════════════════════
#  请求转换: CC → Messages
# ═══════════════════════════════════════════════════════════


def cc_to_messages_request(payload):
    """将 OpenAI CC 格式请求转换为 Anthropic Messages 格式"""
    messages = payload.get('messages', [])
    anthropic_msgs = []
    system_parts = []

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        # system 消息提取到顶层
        if role == 'system':
            system_parts.append(_flatten_text(content))
            continue

        anthropic_role = 'assistant' if role == 'assistant' else 'user'
        anthropic_content = _convert_content(msg)

        # assistant 的 tool_calls → tool_use content blocks
        if role == 'assistant' and 'tool_calls' in msg:
            blocks = _to_blocks(anthropic_content)
            for tc in msg['tool_calls']:
                func = tc.get('function', {})
                arguments = func.get('arguments', '{}')
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                blocks.append({
                    'type': 'tool_use',
                    'id': tc.get('id', f'toolu_{uuid.uuid4().hex[:24]}'),
                    'name': func.get('name', ''),
                    'input': arguments,
                })
            anthropic_content = blocks

        # tool 角色 → user + tool_result
        if role == 'tool':
            text = content if isinstance(content, str) else json.dumps(content)
            anthropic_content = [{
                'type': 'tool_result',
                'tool_use_id': msg.get('tool_call_id', ''),
                'content': text,
            }]
            anthropic_role = 'user'

        if not anthropic_content and anthropic_content != 0:
            continue

        anthropic_msgs.append({'role': anthropic_role, 'content': anthropic_content})

    # Anthropic 要求角色必须交替
    anthropic_msgs = _merge_same_role(anthropic_msgs)

    result = {
        'model': payload.get('model', 'claude-sonnet-4-20250514'),
        'messages': anthropic_msgs,
        'max_tokens': max(payload.get('max_tokens') or 8192, 8192),
    }

    if system_parts:
        result['system'] = '\n\n'.join(system_parts)
    if 'tools' in payload:
        result['tools'] = _convert_tools(payload['tools'])
    for key in ('temperature', 'top_p', 'stream'):
        if key in payload:
            result[key] = payload[key]

    return result


# ═══════════════════════════════════════════════════════════
#  非流式响应转换: Messages → CC
# ═══════════════════════════════════════════════════════════


def messages_to_cc_response(data, request_id=None):
    """将 Anthropic Messages 响应转换为 OpenAI CC 格式"""
    request_id = request_id or gen_id('chatcmpl-')
    data = fix_anthropic_tool_use(data)

    content_text = ''
    reasoning = ''
    tool_calls = []

    for block in data.get('content', []):
        if not isinstance(block, dict):
            continue
        btype = block.get('type', '')
        if btype == 'text':
            content_text += block.get('text', '')
        elif btype == 'thinking':
            reasoning += block.get('thinking', '')
        elif btype == 'tool_use':
            args = block.get('input', {})
            if isinstance(args, dict):
                args = normalize_args(args)
                args = repair_str_replace_args(block.get('name', ''), args)
            tool_calls.append({
                'index': len(tool_calls),
                'id': block.get('id', f'toolu_{uuid.uuid4().hex[:24]}'),
                'type': 'function',
                'function': {
                    'name': block.get('name', ''),
                    'arguments': json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                },
            })

    stop_reason = data.get('stop_reason', 'end_turn')
    message = {'role': 'assistant', 'content': content_text or None}
    if reasoning:
        message['reasoning_content'] = reasoning
    if tool_calls:
        message['tool_calls'] = tool_calls

    usage = data.get('usage', {})
    return {
        'id': request_id,
        'object': 'chat.completion',
        'model': data.get('model', 'claude'),
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': _STOP_REASON_MAP.get(stop_reason, 'stop'),
        }],
        'usage': {
            'prompt_tokens': usage.get('input_tokens', 0),
            'completion_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
        },
    }


# ═══════════════════════════════════════════════════════════
#  流式响应转换: Anthropic SSE → CC chunks
# ═══════════════════════════════════════════════════════════


class AnthropicStreamConverter:
    """将 Anthropic SSE 事件逐个转换为 OpenAI CC 流式 chunk"""

    def __init__(self, request_id=None):
        self._id = request_id or gen_id('chatcmpl-')
        self._tool_index = -1
        self._input_tokens = 0
        self._output_tokens = 0

    def process_event(self, event_type, event_data):
        """处理一个 Anthropic SSE 事件，返回 CC chunk JSON 字符串列表"""
        chunks = []

        if event_type == 'message_start':
            msg = event_data.get('message', {})
            self._input_tokens = msg.get('usage', {}).get('input_tokens', 0)
            chunk = self._make_chunk(delta={'role': 'assistant', 'content': ''})
            if msg.get('model'):
                chunk['model'] = msg['model']
            chunks.append(json.dumps(chunk))

        elif event_type == 'content_block_start':
            block = event_data.get('content_block', {})
            if block.get('type') == 'tool_use':
                self._tool_index += 1
                chunks.append(json.dumps(self._make_chunk(delta={
                    'tool_calls': [{
                        'index': self._tool_index,
                        'id': block.get('id', f'toolu_{uuid.uuid4().hex[:24]}'),
                        'type': 'function',
                        'function': {'name': block.get('name', ''), 'arguments': ''},
                    }]
                })))

        elif event_type == 'content_block_delta':
            delta = event_data.get('delta', {})
            dtype = delta.get('type', '')
            if dtype == 'text_delta' and delta.get('text'):
                chunks.append(json.dumps(self._make_chunk(
                    delta={'content': delta['text']})))
            elif dtype == 'thinking_delta' and delta.get('thinking'):
                chunks.append(json.dumps(self._make_chunk(
                    delta={'reasoning_content': delta['thinking']})))
            elif dtype == 'input_json_delta' and delta.get('partial_json'):
                chunks.append(json.dumps(self._make_chunk(delta={
                    'tool_calls': [{
                        'index': self._tool_index,
                        'function': {'arguments': delta['partial_json']},
                    }]
                })))

        elif event_type == 'message_delta':
            delta = event_data.get('delta', {})
            usage = event_data.get('usage', {})
            self._output_tokens = usage.get('output_tokens', 0)
            finish = _STOP_REASON_MAP.get(delta.get('stop_reason', ''), 'stop')
            chunk = self._make_chunk(delta={}, finish_reason=finish)
            chunk['usage'] = {
                'prompt_tokens': self._input_tokens,
                'completion_tokens': self._output_tokens,
                'total_tokens': self._input_tokens + self._output_tokens,
            }
            chunks.append(json.dumps(chunk))

        return chunks

    def _make_chunk(self, delta, finish_reason=None):
        choice = {'index': 0, 'delta': delta}
        if finish_reason:
            choice['finish_reason'] = finish_reason
        return {
            'id': self._id,
            'object': 'chat.completion.chunk',
            'model': 'claude',
            'choices': [choice],
        }


# ═══════════════════════════════════════════════════════════
#  内部辅助函数
# ═══════════════════════════════════════════════════════════


def _flatten_text(content):
    """将 content 扁平化为纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get('type') == 'text':
                parts.append(p.get('text', ''))
        return '\n'.join(parts)
    return str(content)


def _convert_content(msg):
    """将 OpenAI 消息的 content 字段转为 Anthropic 格式"""
    content = msg.get('content', '')
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = []
        for part in content:
            if isinstance(part, str):
                blocks.append({'type': 'text', 'text': part})
            elif isinstance(part, dict):
                ptype = part.get('type', '')
                if ptype == 'text':
                    blocks.append({'type': 'text', 'text': part.get('text', '')})
                elif ptype == 'image_url':
                    blocks.append(_convert_image(part))
                elif ptype in ('tool_use', 'tool_result'):
                    blocks.append(part)
        return blocks
    return str(content)


def _convert_image(part):
    """将 OpenAI image_url 格式转为 Anthropic image 格式"""
    url_data = part.get('image_url', {})
    url = url_data.get('url', '') if isinstance(url_data, dict) else str(url_data)
    if url.startswith('data:'):
        media_type, _, b64 = url.partition(';base64,')
        return {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': media_type.replace('data:', '') or 'image/png',
                'data': b64,
            },
        }
    return {'type': 'image', 'source': {'type': 'url', 'url': url}}


def _convert_tools(tools):
    """将 OpenAI tools 转为 Anthropic tools 格式（兼容 Cursor 扁平格式）"""
    result = []
    for tool in tools:
        if tool.get('type') == 'function' and 'function' in tool:
            func = tool['function']
            result.append({
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'input_schema': func.get('parameters', {'type': 'object', 'properties': {}}),
            })
        elif 'name' in tool and 'input_schema' in tool:
            result.append({
                'name': tool.get('name', ''),
                'description': tool.get('description', ''),
                'input_schema': tool.get('input_schema', {'type': 'object', 'properties': {}}),
            })
    return result


def _to_blocks(content):
    """将 content 统一转为 blocks 列表"""
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    if isinstance(content, list):
        return list(content)
    return [{'type': 'text', 'text': str(content)}] if content else []


def _merge_same_role(messages):
    """合并相邻同角色消息（Anthropic 要求角色必须交替）"""
    if not messages:
        return messages
    merged = [messages[0]]
    for msg in messages[1:]:
        if msg['role'] == merged[-1]['role']:
            prev = _to_blocks(merged[-1]['content'])
            curr = _to_blocks(msg['content'])
            merged[-1]['content'] = prev + curr
        else:
            merged.append(msg)
    return merged

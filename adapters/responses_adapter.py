"""Responses API 适配

Cursor 对 GPT/Claude-Opus 等模型使用 /v1/responses 格式。
本模块将 Responses 格式与 Chat Completions 格式互相转换：
  请求: Responses → CC
  响应: CC → Responses（非流式 + 流式）
  流式: 支持从 CC chunks 或 Anthropic SSE 事件直接转换
"""

import json
import logging

from utils.http import gen_id

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  请求转换: Responses → CC
# ═══════════════════════════════════════════════════════════


def responses_to_cc(payload):
    """将 /v1/responses 请求转换为 /v1/chat/completions 格式"""
    messages = []

    if payload.get('instructions'):
        messages.append({'role': 'system', 'content': payload['instructions']})

    input_data = payload.get('input', [])
    if isinstance(input_data, str):
        messages.append({'role': 'user', 'content': input_data})
    elif isinstance(input_data, list):
        _convert_input_items(input_data, messages)

    result = {
        'model': payload.get('model', ''),
        'messages': messages,
        'stream': payload.get('stream', False),
    }

    if 'tools' in payload:
        result['tools'] = _convert_tools(payload['tools'])
    for key in ('temperature', 'top_p'):
        if key in payload:
            result[key] = payload[key]
    if 'max_output_tokens' in payload:
        result['max_tokens'] = payload['max_output_tokens']
    if 'tool_choice' in payload:
        result['tool_choice'] = payload['tool_choice']

    return result


# ═══════════════════════════════════════════════════════════
#  非流式响应转换: CC → Responses
# ═══════════════════════════════════════════════════════════


def cc_to_responses(cc_resp, model=''):
    """将 CC 响应转换为 Responses 格式"""
    choice = (cc_resp.get('choices') or [{}])[0]
    msg = choice.get('message') or {}
    finish = choice.get('finish_reason', 'stop')

    output = []

    if msg.get('reasoning_content'):
        output.append({
            'type': 'reasoning',
            'id': gen_id('rs_'),
            'summary': [{'type': 'summary_text', 'text': msg['reasoning_content']}],
        })

    if msg.get('content'):
        output.append({
            'type': 'message',
            'id': gen_id('msg_'),
            'status': 'completed',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': msg['content']}],
        })

    for tc in (msg.get('tool_calls') or []):
        func = tc.get('function') or {}
        output.append({
            'type': 'function_call',
            'id': gen_id('fc_'),
            'status': 'completed',
            'call_id': tc.get('id', gen_id('call_')),
            'name': func.get('name', ''),
            'arguments': func.get('arguments', '{}'),
        })

    usage = cc_resp.get('usage', {})
    return {
        'id': cc_resp.get('id', gen_id('resp_')),
        'object': 'response',
        'status': 'incomplete' if finish == 'length' else 'completed',
        'model': model or cc_resp.get('model', ''),
        'output': output,
        'usage': {
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
        },
    }


# ═══════════════════════════════════════════════════════════
#  流式转换器: CC chunks / Anthropic SSE → Responses SSE
# ═══════════════════════════════════════════════════════════


class ResponsesStreamConverter:
    """有状态转换器：将 CC 流式 chunk 或 Anthropic SSE 事件转为 Responses SSE 事件"""

    def __init__(self, response_id=None, model=''):
        self.resp_id = response_id or gen_id('resp_')
        self.model = model

        # 思考内容缓冲
        self._rs_buf = ''
        self._rs_started = False
        self._rs_closed = False
        self._rs_id = gen_id('rs_')

        # 文本内容缓冲
        self._text_buf = ''
        self._text_started = False
        self._text_closed = False
        self._msg_id = gen_id('msg_')

        # 工具调用缓冲 {index: {name, args, call_id, fc_id}}
        self._tools = {}
        self._output_items = []
        self._finished = False
        self._input_tokens = 0

    # ─── 公开接口 ─────────────────────────────────

    def start_events(self):
        """生成流开始事件"""
        return [self._sse('response.created', {
            'id': self.resp_id, 'object': 'response',
            'status': 'in_progress', 'model': self.model, 'output': [],
        })]

    def process_cc_chunk(self, chunk):
        """处理 CC 格式的流式 chunk，返回 Responses SSE 事件列表"""
        events = []
        for choice in (chunk.get('choices') or []):
            delta = choice.get('delta') or {}
            finish = choice.get('finish_reason')

            if delta.get('reasoning_content'):
                events.extend(self._on_reasoning(delta['reasoning_content']))
            if delta.get('content') is not None and delta['content'] != '':
                events.extend(self._on_text(delta['content']))
            for tc in (delta.get('tool_calls') or []):
                events.extend(self._on_tool_call(tc))
            if finish and not self._finished:
                self._finished = True
                events.extend(self._do_finish(finish, chunk.get('usage')))

        return events

    def process_anthropic_event(self, event_type, event_data):
        """直接处理 Anthropic SSE 事件（跳过 CC 中间转换，更高效）"""
        events = []

        if event_type == 'message_start':
            usage = event_data.get('message', {}).get('usage', {})
            self._input_tokens = usage.get('input_tokens', 0)

        elif event_type == 'content_block_start':
            block = event_data.get('content_block', {})
            btype = block.get('type', '')
            if btype == 'thinking' and not self._rs_started:
                self._rs_started = True
                events.append(self._sse('response.output_item.added', {
                    'type': 'reasoning', 'id': self._rs_id, 'summary': [],
                }))
            elif btype == 'text':
                events.extend(self._ensure_text_started())
            elif btype == 'tool_use':
                events.extend(self._start_tool_from_block(block))

        elif event_type == 'content_block_delta':
            delta = event_data.get('delta', {})
            dtype = delta.get('type', '')
            if dtype == 'thinking_delta' and delta.get('thinking'):
                self._rs_buf += delta['thinking']
                events.append(self._sse('response.reasoning_summary_text.delta', {
                    'type': 'summary_text', 'delta': delta['thinking'],
                }))
            elif dtype == 'text_delta' and delta.get('text'):
                self._text_buf += delta['text']
                events.append(self._sse('response.output_text.delta', {
                    'type': 'output_text', 'delta': delta['text'],
                }))
            elif dtype == 'input_json_delta' and delta.get('partial_json') and self._tools:
                idx = max(self._tools.keys())
                self._tools[idx]['args'] += delta['partial_json']
                events.append(self._sse('response.function_call_arguments.delta', {
                    'type': 'function_call', 'delta': delta['partial_json'],
                }))

        elif event_type == 'message_delta':
            delta = event_data.get('delta', {})
            stop = delta.get('stop_reason', 'end_turn')
            usage = event_data.get('usage', {})
            finish = {'tool_use': 'tool_calls', 'max_tokens': 'length'}.get(stop, 'stop')
            if not self._finished:
                self._finished = True
                u = {
                    'input_tokens': self._input_tokens,
                    'output_tokens': usage.get('output_tokens', 0),
                    'total_tokens': self._input_tokens + usage.get('output_tokens', 0),
                }
                events.extend(self._do_finish(finish, u))

        return events

    def finalize(self):
        """流结束时补发未关闭的事件"""
        if self._finished:
            return []
        self._finished = True
        return self._do_finish('stop', None)

    # ─── 内部事件处理 ─────────────────────────────

    def _on_reasoning(self, text):
        """处理思考内容 delta"""
        events = []
        if not self._rs_started:
            self._rs_started = True
            events.append(self._sse('response.output_item.added', {
                'type': 'reasoning', 'id': self._rs_id, 'summary': [],
            }))
        self._rs_buf += text
        events.append(self._sse('response.reasoning_summary_text.delta', {
            'type': 'summary_text', 'delta': text,
        }))
        return events

    def _on_text(self, text):
        """处理文本内容 delta"""
        events = self._ensure_text_started()
        self._text_buf += text
        events.append(self._sse('response.output_text.delta', {
            'type': 'output_text', 'delta': text,
        }))
        return events

    def _on_tool_call(self, tc):
        """处理工具调用 delta"""
        events = []
        idx = tc.get('index', 0)
        func = tc.get('function') or {}

        if idx not in self._tools:
            if self._rs_started and not self._rs_closed:
                events.extend(self._close_reasoning())
            if self._text_started and not self._text_closed:
                events.extend(self._close_text())
            call_id = tc.get('id', gen_id('call_'))
            name = func.get('name', '')
            fc_id = gen_id('fc_')
            self._tools[idx] = {'name': name, 'args': '', 'call_id': call_id, 'fc_id': fc_id}
            events.append(self._sse('response.output_item.added', {
                'type': 'function_call', 'id': fc_id,
                'status': 'in_progress', 'call_id': call_id,
                'name': name, 'arguments': '',
            }))

        if func.get('name'):
            self._tools[idx]['name'] = func['name']
        if func.get('arguments', ''):
            self._tools[idx]['args'] += func['arguments']
            events.append(self._sse('response.function_call_arguments.delta', {
                'type': 'function_call', 'delta': func['arguments'],
            }))
        return events

    def _ensure_text_started(self):
        """确保文本输出项已开始"""
        events = []
        if self._rs_started and not self._rs_closed:
            events.extend(self._close_reasoning())
        if not self._text_started:
            self._text_started = True
            events.append(self._sse('response.output_item.added', {
                'type': 'message', 'id': self._msg_id,
                'status': 'in_progress', 'role': 'assistant', 'content': [],
            }))
            events.append(self._sse('response.content_part.added', {
                'type': 'output_text', 'text': '',
            }))
        return events

    def _start_tool_from_block(self, block):
        """从 Anthropic tool_use block 开始新的工具调用"""
        events = []
        if self._rs_started and not self._rs_closed:
            events.extend(self._close_reasoning())
        if self._text_started and not self._text_closed:
            events.extend(self._close_text())
        idx = len(self._tools)
        tool_id = block.get('id', gen_id('toolu_'))
        name = block.get('name', '')
        fc_id = gen_id('fc_')
        self._tools[idx] = {'name': name, 'args': '', 'call_id': tool_id, 'fc_id': fc_id}
        events.append(self._sse('response.output_item.added', {
            'type': 'function_call', 'id': fc_id,
            'status': 'in_progress', 'call_id': tool_id,
            'name': name, 'arguments': '',
        }))
        return events

    # ─── 关闭/结束事件 ────────────────────────────

    def _close_reasoning(self):
        if self._rs_closed:
            return []
        self._rs_closed = True
        rs = {
            'type': 'reasoning', 'id': self._rs_id,
            'summary': [{'type': 'summary_text', 'text': self._rs_buf}],
        }
        self._output_items.append(rs)
        return [
            self._sse('response.reasoning_summary_text.done', {
                'type': 'summary_text', 'text': self._rs_buf,
            }),
            self._sse('response.output_item.done', rs),
        ]

    def _close_text(self):
        if self._text_closed:
            return []
        self._text_closed = True
        msg = {
            'type': 'message', 'id': self._msg_id,
            'status': 'completed', 'role': 'assistant',
            'content': [{'type': 'output_text', 'text': self._text_buf}],
        }
        self._output_items.append(msg)
        return [
            self._sse('response.output_text.done', {'type': 'output_text', 'text': self._text_buf}),
            self._sse('response.output_item.done', msg),
        ]

    def _do_finish(self, finish_reason, usage):
        """生成流结束的所有关闭事件"""
        events = []
        if self._rs_started and not self._rs_closed:
            events.extend(self._close_reasoning())
        if self._text_started and not self._text_closed:
            events.extend(self._close_text())

        for idx in sorted(self._tools.keys()):
            buf = self._tools[idx]
            events.append(self._sse('response.function_call_arguments.done', {
                'type': 'function_call', 'arguments': buf['args'],
            }))
            fc = {
                'type': 'function_call', 'id': buf['fc_id'],
                'status': 'completed', 'call_id': buf['call_id'],
                'name': buf['name'], 'arguments': buf['args'],
            }
            events.append(self._sse('response.output_item.done', fc))
            self._output_items.append(fc)

        usage_data = usage if isinstance(usage, dict) else {}
        events.append(self._sse('response.completed', {
            'id': self.resp_id, 'object': 'response',
            'status': 'incomplete' if finish_reason == 'length' else 'completed',
            'model': self.model, 'output': self._output_items, 'usage': usage_data,
        }))
        return events

    def _sse(self, event_type, data):
        """构建 SSE 事件字符串"""
        return f'event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


# ═══════════════════════════════════════════════════════════
#  内部辅助函数
# ═══════════════════════════════════════════════════════════


def _convert_input_items(items, messages):
    """将 Responses input 数组转换为 CC messages"""
    i = 0
    while i < len(items):
        item = items[i]

        if isinstance(item, str):
            messages.append({'role': 'user', 'content': item})
            i += 1
            continue

        if not isinstance(item, dict):
            i += 1
            continue

        item_type = item.get('type', '')
        role = item.get('role', '')

        # 简单角色消息（无 type 字段）
        if role and not item_type:
            content = item.get('content', '')
            if isinstance(content, list):
                content = _extract_text(content)
            messages.append({'role': role, 'content': content or ''})
            i += 1
            continue

        # Responses message 对象
        if item_type == 'message' or (role and not item_type):
            role = item.get('role', 'assistant')
            content = _extract_text(item.get('content', []))
            msg = {'role': role, 'content': content or ''}
            if role == 'assistant':
                tool_calls, consumed = _collect_function_calls(items, i + 1)
                if tool_calls:
                    msg['tool_calls'] = tool_calls
                    if not msg['content']:
                        msg['content'] = None
                    messages.append(msg)
                    i += 1 + consumed
                    continue
            messages.append(msg)
            i += 1
            continue

        # function_call（工具调用）
        if item_type == 'function_call':
            tc = {
                'id': item.get('call_id') or gen_id('call_'),
                'type': 'function',
                'function': {
                    'name': item.get('name', ''),
                    'arguments': item.get('arguments', '{}'),
                },
            }
            if messages and messages[-1]['role'] == 'assistant':
                messages[-1].setdefault('tool_calls', []).append(tc)
                if not messages[-1].get('content'):
                    messages[-1]['content'] = None
            else:
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [tc]})
            i += 1
            continue

        # function_call_output（工具结果）
        if item_type == 'function_call_output':
            output = item.get('output', '')
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({
                'role': 'tool',
                'tool_call_id': item.get('call_id', ''),
                'content': output,
            })
            i += 1
            continue

        if role:
            messages.append({'role': role, 'content': str(item.get('content', ''))})
        i += 1


def _collect_function_calls(items, start):
    """收集紧随 assistant message 之后的连续 function_call 项"""
    tool_calls = []
    j = start
    while j < len(items):
        nxt = items[j]
        if isinstance(nxt, dict) and nxt.get('type') == 'function_call':
            tool_calls.append({
                'id': nxt.get('call_id') or gen_id('call_'),
                'type': 'function',
                'function': {
                    'name': nxt.get('name', ''),
                    'arguments': nxt.get('arguments', '{}'),
                },
            })
            j += 1
        else:
            break
    return tool_calls, j - start


def _extract_text(content):
    """从 content 中提取纯文本"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ''
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict):
            t = part.get('type', '')
            if t in ('output_text', 'input_text', 'text'):
                texts.append(part.get('text', ''))
            elif t == 'refusal':
                texts.append(part.get('refusal', ''))
    return '\n'.join(texts) if texts else ''


def _convert_tools(tools):
    """将 Responses tools 转为 CC tools 格式"""
    result = []
    for t in tools:
        if t.get('type') != 'function':
            continue
        if 'function' in t:
            result.append(t)
        else:
            result.append({
                'type': 'function',
                'function': {
                    'name': t.get('name', ''),
                    'description': t.get('description', ''),
                    'parameters': t.get('parameters', {'type': 'object', 'properties': {}}),
                },
            })
    return result

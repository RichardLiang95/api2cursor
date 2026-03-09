"""路由: /v1/chat/completions

处理 Cursor 发来的 OpenAI Chat Completions 格式请求。
根据模型映射的 backend 字段分发到 OpenAI 或 Anthropic 后端。
"""

import json
import logging

from flask import Blueprint, request, jsonify

import settings
from config import Config
from adapters.openai_fixer import normalize_request, fix_response, fix_stream_chunk
from adapters.openai_anthropic import (
    cc_to_messages_request, messages_to_cc_response, AnthropicStreamConverter,
)
from adapters.responses_adapter import responses_to_cc
from utils.http import (
    build_openai_headers, build_anthropic_headers,
    forward_request, sse_response,
    iter_openai_sse, iter_anthropic_sse,
)
from utils.think_tag import ThinkTagExtractor
logger = logging.getLogger(__name__)


def _dbg(msg):
    """DEBUG 模式下输出详细日志"""
    if Config.DEBUG:
        logger.info(f'[调试] {msg}')

bp = Blueprint('chat', __name__)


@bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    payload = request.get_json(force=True)
    is_stream = payload.get('stream', False)
    # 保留 Cursor 发送的原始模型名，响应时需要回填
    cursor_model = payload.get('model', 'unknown')
    msg_count = len(payload.get('messages', []))

    # 容错：Responses 格式误入 CC 端点
    if msg_count == 0 and 'input' in payload:
        logger.info('检测到 Responses 格式（有 input 无 messages），自动转换')
        payload = responses_to_cc(payload)
        msg_count = len(payload.get('messages', []))
    elif msg_count == 0:
        logger.warning(f'messages 为空, payload keys: {list(payload.keys())}')

    mapping = settings.resolve_model(cursor_model)
    backend = mapping['backend']
    upstream = mapping['upstream_model']
    url_base = mapping['target_url']
    api_key = mapping['api_key']

    logger.info(
        f'[CC] {cursor_model} → {upstream} '
        f'后端={backend} 流式={is_stream} 消息数={msg_count}'
    )
    _log_messages(payload)

    if backend == 'openai':
        return _via_openai(payload, upstream, url_base, api_key, is_stream, cursor_model)
    else:
        return _via_anthropic(payload, upstream, url_base, api_key, is_stream, cursor_model)


# ─── OpenAI 后端 ──────────────────────────────────


def _via_openai(payload, upstream, url_base, api_key, is_stream, cursor_model):
    """通过 OpenAI 兼容后端转发"""
    _dbg(f'Cursor 原始请求 keys={list(payload.keys())} '
         f'其他字段={json.dumps({k: v for k, v in payload.items() if k != "messages"}, ensure_ascii=False, default=str)[:500]}')

    payload = normalize_request(payload, upstream)
    _dbg(f'normalize 后 model={payload.get("model")} tools数={len(payload.get("tools", []))}')

    headers = build_openai_headers(api_key)
    url = f'{url_base.rstrip("/")}/v1/chat/completions'

    if not is_stream:
        payload['stream'] = False
        resp, err = forward_request(url, headers, payload)
        if err:
            return err
        raw = resp.json()
        _dbg(f'上游原始响应={json.dumps(raw, ensure_ascii=False, default=str)[:1000]}')
        data = fix_response(raw)
        data['model'] = cursor_model
        _dbg(f'修复后响应={json.dumps(data, ensure_ascii=False, default=str)[:1000]}')
        usage = data.get('usage', {})
        logger.info(
            f'[CC] 完成 prompt={usage.get("prompt_tokens", 0)} '
            f'completion={usage.get("completion_tokens", 0)}'
        )
        return jsonify(data)

    # 流式处理
    payload['stream'] = True
    _n = [0]

    def generate():
        resp, err = forward_request(url, headers, payload, stream=True)
        if err:
            yield f'data: {json.dumps({"error": {"message": err, "type": "upstream_error"}})}\n\n'
            return

        think_ext = ThinkTagExtractor()

        for chunk in iter_openai_sse(resp):
            if chunk is None:  # [DONE]
                _dbg(f'流结束，共 {_n[0]} 个 chunk')
                yield 'data: [DONE]\n\n'
                return

            if _n[0] < 10:
                _dbg(f'上游原始 chunk#{_n[0]}={json.dumps(chunk, ensure_ascii=False, default=str)[:500]}')

            chunk = fix_stream_chunk(chunk)
            chunk['model'] = cursor_model

            for out in think_ext.process_chunk(chunk):
                if _n[0] < 10:
                    _dbg(f'发给Cursor chunk#{_n[0]}={json.dumps(out, ensure_ascii=False, default=str)[:500]}')
                yield f'data: {json.dumps(out)}\n\n'

            _n[0] += 1

    return sse_response(generate())


# ─── Anthropic 后端 ───────────────────────────────


def _via_anthropic(payload, upstream, url_base, api_key, is_stream, cursor_model):
    """通过 Anthropic 后端转发（CC → Messages → CC）"""
    payload['model'] = upstream
    anthropic_payload = cc_to_messages_request(payload)
    _dbg(f'CC→Messages 转换后 keys={list(anthropic_payload.keys())} '
         f'messages数={len(anthropic_payload.get("messages", []))}')

    headers = build_anthropic_headers(api_key)
    url = f'{url_base.rstrip("/")}/v1/messages'

    if not is_stream:
        anthropic_payload['stream'] = False
        resp, err = forward_request(url, headers, anthropic_payload)
        if err:
            return err
        raw = resp.json()
        _dbg(f'上游原始响应={json.dumps(raw, ensure_ascii=False, default=str)[:1000]}')
        data = messages_to_cc_response(raw)
        data['model'] = cursor_model
        _dbg(f'Messages→CC 转换后={json.dumps(data, ensure_ascii=False, default=str)[:1000]}')
        usage = data.get('usage', {})
        logger.info(
            f'[CC] 完成 prompt={usage.get("prompt_tokens", 0)} '
            f'completion={usage.get("completion_tokens", 0)}'
        )
        return jsonify(data)

    # 流式处理
    anthropic_payload['stream'] = True
    converter = AnthropicStreamConverter()
    _n = [0]

    def generate():
        resp, err = forward_request(url, headers, anthropic_payload, stream=True)
        if err:
            yield f'data: {json.dumps({"error": {"message": err, "type": "upstream_error"}})}\n\n'
            return

        for event_type, event_data in iter_anthropic_sse(resp):
            if _n[0] < 10:
                _dbg(f'上游事件#{_n[0]} {event_type}={json.dumps(event_data, ensure_ascii=False, default=str)[:500]}')

            for chunk_str in converter.process_event(event_type, event_data):
                try:
                    chunk_obj = json.loads(chunk_str)
                    chunk_obj['model'] = cursor_model
                    chunk_str = json.dumps(chunk_obj)
                except (json.JSONDecodeError, TypeError):
                    pass
                if _n[0] < 10:
                    _dbg(f'发给Cursor chunk#{_n[0]}={chunk_str[:500]}')
                yield f'data: {chunk_str}\n\n'

            _n[0] += 1

        _dbg(f'流结束，共 {_n[0]} 个事件')
        yield 'data: [DONE]\n\n'

    return sse_response(generate())


def _log_messages(payload):
    """记录请求中的消息摘要"""
    for i, msg in enumerate(payload.get('messages', [])):
        role = msg.get('role', '?')
        content = msg.get('content')
        extra = ''
        if 'tool_calls' in msg:
            extra += f' tool_calls={len(msg["tool_calls"])}'
        if msg.get('tool_call_id'):
            extra += f' tool_call_id={msg["tool_call_id"]}'

        if isinstance(content, list):
            info = f'list[{len(content)}]'
        elif isinstance(content, str):
            info = f'str[{len(content)}]'
        else:
            info = type(content).__name__
        logger.info(f'  消息[{i}] {role} {info}{extra}')

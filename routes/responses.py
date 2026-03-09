"""路由: /v1/responses

处理 Cursor 对 GPT/Claude-Opus 等模型发出的 Responses API 格式请求。
转换为 CC 格式后分发到对应后端，响应再转回 Responses 格式。
"""

import json
import logging

from flask import Blueprint, request, jsonify

import settings
from adapters.responses_adapter import responses_to_cc, cc_to_responses, ResponsesStreamConverter
from adapters.openai_fixer import normalize_request, fix_response, fix_stream_chunk
from adapters.openai_anthropic import cc_to_messages_request, messages_to_cc_response
from utils.http import (
    build_openai_headers, build_anthropic_headers,
    forward_request, sse_response,
    iter_openai_sse, iter_anthropic_sse,
)
from utils.think_tag import ThinkTagExtractor

logger = logging.getLogger(__name__)

bp = Blueprint('responses', __name__)


@bp.route('/v1/responses', methods=['POST'])
def responses_endpoint():
    payload = request.get_json(force=True)
    model = payload.get('model', 'unknown')
    is_stream = payload.get('stream', False)

    mapping = settings.resolve_model(model)
    backend = mapping['backend']
    upstream = mapping['upstream_model']
    url_base = mapping['target_url']
    api_key = mapping['api_key']

    logger.info(f'[Responses] {model} → {upstream} 后端={backend} 流式={is_stream}')

    # Responses → CC
    cc_payload = responses_to_cc(payload)
    cc_payload['model'] = upstream

    if backend == 'openai':
        return _via_openai(cc_payload, url_base, api_key, is_stream, model)
    else:
        return _via_anthropic(cc_payload, url_base, api_key, is_stream, model)


# ─── OpenAI 后端 ──────────────────────────────────


def _via_openai(cc_payload, url_base, api_key, is_stream, display_model):
    """通过 OpenAI 后端处理"""
    cc_payload = normalize_request(cc_payload)
    headers = build_openai_headers(api_key)
    url = f'{url_base.rstrip("/")}/v1/chat/completions'

    if not is_stream:
        cc_payload['stream'] = False
        resp, err = forward_request(url, headers, cc_payload)
        if err:
            return err
        return jsonify(cc_to_responses(fix_response(resp.json()), display_model))

    # 流式处理
    cc_payload['stream'] = True
    converter = ResponsesStreamConverter(model=display_model)

    def generate():
        yield from converter.start_events()

        resp, err = forward_request(url, headers, cc_payload, stream=True)
        if err:
            yield f'event: error\ndata: {json.dumps({"error": err})}\n\n'
            return

        think_ext = ThinkTagExtractor()
        for chunk in iter_openai_sse(resp):
            if chunk is None:
                yield from converter.finalize()
                return
            chunk = fix_stream_chunk(chunk)
            for out in think_ext.process_chunk(chunk):
                yield from converter.process_cc_chunk(out)

    return sse_response(generate())


# ─── Anthropic 后端 ───────────────────────────────


def _via_anthropic(cc_payload, url_base, api_key, is_stream, display_model):
    """通过 Anthropic 后端处理"""
    anthropic_payload = cc_to_messages_request(cc_payload)
    headers = build_anthropic_headers(api_key)
    url = f'{url_base.rstrip("/")}/v1/messages'

    if not is_stream:
        anthropic_payload['stream'] = False
        resp, err = forward_request(url, headers, anthropic_payload)
        if err:
            return err
        cc_data = messages_to_cc_response(resp.json())
        return jsonify(cc_to_responses(cc_data, display_model))

    # 流式处理：Anthropic SSE → Responses SSE（跳过 CC 中间态）
    anthropic_payload['stream'] = True
    converter = ResponsesStreamConverter(model=display_model)

    def generate():
        yield from converter.start_events()

        resp, err = forward_request(url, headers, anthropic_payload, stream=True)
        if err:
            yield f'event: error\ndata: {json.dumps({"error": err})}\n\n'
            return

        for event_type, event_data in iter_anthropic_sse(resp):
            yield from converter.process_anthropic_event(event_type, event_data)

        yield from converter.finalize()

    return sse_response(generate())

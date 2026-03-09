"""路由: 管理面板

提供 Web 管理界面和 API：
  - /admin         — 管理面板页面
  - /v1/models     — 模型列表（供 Cursor 查询）
  - /api/admin/*   — 登录验证、全局设置 CRUD、模型映射 CRUD
"""

import os
import logging

from flask import Blueprint, request, jsonify, send_from_directory

import settings
from config import Config

logger = logging.getLogger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')

bp = Blueprint('admin', __name__)


# ─── 静态页面 ─────────────────────────────────────


@bp.route('/admin')
@bp.route('/admin/')
def admin_page():
    return send_from_directory(_STATIC_DIR, 'admin.html')


@bp.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(_STATIC_DIR, filename)


# ─── 模型列表 ─────────────────────────────────────


@bp.route('/v1/models', methods=['GET'])
def list_models():
    mappings = settings.get().get('model_mappings', {})
    models = [{
        'id': name,
        'object': 'model',
        'owned_by': info.get('backend', 'custom'),
    } for name, info in mappings.items()]

    if not models:
        models.append({
            'id': 'claude-sonnet-4-5-20250929',
            'object': 'model',
            'owned_by': 'anthropic',
        })
    return jsonify({'object': 'list', 'data': models})


# ─── 登录验证 ─────────────────────────────────────


@bp.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json(force=True)
    if not Config.ACCESS_API_KEY:
        return jsonify({'ok': True, 'message': '未配置鉴权'})
    if data.get('key', '') == Config.ACCESS_API_KEY:
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'message': '密钥错误'}), 401


# ─── 全局设置 ─────────────────────────────────────


@bp.route('/api/admin/settings', methods=['GET'])
def get_settings():
    err = _check_auth()
    if err:
        return err
    s = settings.get()
    return jsonify({
        'proxy_target_url': s.get('proxy_target_url', ''),
        'proxy_api_key': s.get('proxy_api_key', ''),
        'env_target_url': Config.PROXY_TARGET_URL,
        'env_api_key': '***' if Config.PROXY_API_KEY else '',
    })


@bp.route('/api/admin/settings', methods=['PUT'])
def update_settings():
    err = _check_auth()
    if err:
        return err
    data = request.get_json(force=True)
    s = settings.get()
    for key in ('proxy_target_url', 'proxy_api_key'):
        if key in data:
            s[key] = data[key]
    return _save_and_respond(s, '全局设置已更新')


# ─── 模型映射 CRUD ────────────────────────────────


@bp.route('/api/admin/mappings', methods=['GET'])
def list_mappings():
    err = _check_auth()
    if err:
        return err
    return jsonify(settings.get().get('model_mappings', {}))


@bp.route('/api/admin/mappings', methods=['POST'])
def add_mapping():
    err = _check_auth()
    if err:
        return err
    data = request.get_json(force=True)
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': '名称不能为空'}), 400

    s = settings.get()
    mappings = s.setdefault('model_mappings', {})
    mappings[name] = {
        'upstream_model': data.get('upstream_model', name),
        'backend': data.get('backend', 'auto'),
        'target_url': data.get('target_url', ''),
        'api_key': data.get('api_key', ''),
    }
    return _save_and_respond(s, f'映射已添加: {name}')


@bp.route('/api/admin/mappings/<path:name>', methods=['PUT'])
def update_mapping(name):
    err = _check_auth()
    if err:
        return err
    data = request.get_json(force=True)
    s = settings.get()
    mappings = s.get('model_mappings', {})
    if name not in mappings:
        return jsonify({'error': '映射不存在'}), 404

    new_name = data.get('name', name).strip()
    entry = {
        'upstream_model': data.get('upstream_model', name),
        'backend': data.get('backend', 'auto'),
        'target_url': data.get('target_url', ''),
        'api_key': data.get('api_key', ''),
    }
    if new_name != name:
        del mappings[name]
    mappings[new_name] = entry
    s['model_mappings'] = mappings
    return _save_and_respond(s, f'映射已更新: {name} → {new_name}')


@bp.route('/api/admin/mappings/<path:name>', methods=['DELETE'])
def delete_mapping(name):
    err = _check_auth()
    if err:
        return err
    s = settings.get()
    mappings = s.get('model_mappings', {})
    if name in mappings:
        del mappings[name]
        s['model_mappings'] = mappings
        return _save_and_respond(s, f'映射已删除: {name}')
    return jsonify({'ok': True})


# ─── 内部辅助 ─────────────────────────────────────


def _check_auth():
    """Admin API 鉴权，返回 None 表示通过"""
    if not Config.ACCESS_API_KEY:
        return None
    auth = request.headers.get('Authorization', '')
    token = auth[7:] if auth.startswith('Bearer ') else request.headers.get('x-api-key', '')
    if token != Config.ACCESS_API_KEY:
        return jsonify({'error': '未授权'}), 401
    return None


def _save_and_respond(data, log_msg):
    """保存配置并返回响应"""
    try:
        settings.save(data)
    except OSError as e:
        logger.error(f'保存失败: {e}')
        return jsonify({'error': {'message': f'保存失败: {e}', 'type': 'save_error'}}), 500
    logger.info(log_msg)
    return jsonify({'ok': True})

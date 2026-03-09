"""工具参数修复

修复 LLM 生成的工具调用参数中的常见问题：
  - 智能引号 → 普通引号
  - file_path → path 字段映射
  - StrReplace 工具的 old_string 精确匹配修复
  - Anthropic tool_use 块的 ID 和 stop_reason 修复
"""

import os
import re
import uuid

# 智能引号字符集
_SMART_DOUBLE = frozenset('«»\u201c\u201d\u275e\u201f\u201e\u275d')
_SMART_SINGLE = frozenset('\u2018\u2019\u201a\u201b')


def normalize_args(args):
    """规范化工具参数：file_path → path"""
    if isinstance(args, dict) and 'file_path' in args and 'path' not in args:
        args['path'] = args.pop('file_path')
    return args


def repair_str_replace_args(tool_name, args):
    """修复 StrReplace/search_replace 工具的精确匹配问题

    当 old_string 包含智能引号导致无法精确匹配文件内容时，
    用容错正则在文件中查找唯一匹配并替换为实际内容。
    """
    if not isinstance(args, dict):
        return args

    name_lower = (tool_name or '').lower()
    if 'str_replace' not in name_lower and 'search_replace' not in name_lower:
        return args

    old_str = args.get('old_string') or args.get('old_str')
    if not old_str:
        return args

    file_path = args.get('path') or args.get('file_path')
    if not file_path or not os.path.isfile(file_path):
        return args

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return args

    # 已精确匹配，无需修复
    if old_str in content:
        return args

    # 构建容错正则尝试匹配
    pattern = _build_fuzzy_pattern(old_str)
    try:
        matches = list(re.finditer(pattern, content))
    except re.error:
        return args

    # 仅在唯一匹配时修复，避免歧义
    if len(matches) != 1:
        return args

    matched = matches[0].group()
    if 'old_string' in args:
        args['old_string'] = matched
    elif 'old_str' in args:
        args['old_str'] = matched

    # 同步修复 new_string 中的智能引号
    new_str = args.get('new_string') or args.get('new_str')
    if new_str:
        fixed = _replace_smart_quotes(new_str)
        if 'new_string' in args:
            args['new_string'] = fixed
        elif 'new_str' in args:
            args['new_str'] = fixed

    return args


def fix_anthropic_tool_use(response_data):
    """修复 Anthropic 响应中的 tool_use 块（补全 ID、修正 stop_reason）"""
    if not isinstance(response_data, dict):
        return response_data

    content = response_data.get('content', [])
    if not isinstance(content, list):
        return response_data

    has_tool_use = False
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_use':
            has_tool_use = True
            if not block.get('id'):
                block['id'] = f'toolu_{uuid.uuid4().hex[:24]}'

    if has_tool_use and response_data.get('stop_reason') != 'tool_use':
        response_data['stop_reason'] = 'tool_use'

    return response_data


# ─── 内部辅助 ──────────────────────────────────────


def _build_fuzzy_pattern(text):
    """构建容错正则：智能引号可互换、空白可伸缩、反斜杠可重复"""
    parts = []
    for ch in text:
        if ch in _SMART_DOUBLE or ch == '"':
            parts.append('["\u00ab\u201c\u201d\u275e\u201f\u201e\u275d\u00bb]')
        elif ch in _SMART_SINGLE or ch == "'":
            parts.append("['\u2018\u2019\u201a\u201b]")
        elif ch in (' ', '\t'):
            parts.append(r'\s+')
        elif ch == '\\':
            parts.append(r'\\{1,2}')
        else:
            parts.append(re.escape(ch))
    return ''.join(parts)


def _replace_smart_quotes(text):
    """将智能引号替换为普通 ASCII 引号"""
    return ''.join(
        '"' if ch in _SMART_DOUBLE else
        "'" if ch in _SMART_SINGLE else
        ch for ch in text
    )

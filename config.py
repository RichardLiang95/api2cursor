"""环境变量配置"""

import os


class Config:
    # 上游 API 地址
    PROXY_TARGET_URL = os.getenv('PROXY_TARGET_URL', 'https://api.anthropic.com')
    # 上游 API 密钥
    PROXY_API_KEY = os.getenv('PROXY_API_KEY', '')
    # 服务监听端口
    PROXY_PORT = int(os.getenv('PROXY_PORT', '3029'))
    # 请求超时时间（秒）
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', '300'))
    # 访问鉴权密钥，留空则不启用鉴权
    ACCESS_API_KEY = os.getenv('ACCESS_API_KEY', '')
    # 调试模式：开启后输出详细的请求/响应日志
    DEBUG = os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes', 'on')

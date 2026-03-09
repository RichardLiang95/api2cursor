# API 2 Cursor

让 Cursor 通过第三方中转站使用任意 LLM 模型的 API 代理服务。

## 它解决什么问题

Cursor 根据模型名发送不同格式的请求：

| Cursor 模型名风格 | 请求格式 |
|---|---|
| `claude-sonnet-*`、`glm-*` | `/v1/chat/completions` (OpenAI CC) |
| `gpt-*`、`claude-opus-*` | `/v1/responses` (OpenAI Responses) |

而中转站通常只支持 `/v1/chat/completions` 或 `/v1/messages`。

本项目在中间做协议转换，**不管 Cursor 发什么格式，都能正确转发到中转站；不管中转站返回什么格式，都让 Cursor 能正确接收**。

## 架构

```
Cursor                      API 2 Cursor                      中转站
  │                              │                              │
  ├─ /v1/chat/completions ──→ chat.py ─┬─ openai 后端 ────────→ /v1/chat/completions
  │                                     └─ anthropic 后端 ────→ /v1/messages
  │                              │
  ├─ /v1/responses ──────→ responses.py → 转为 CC → 同上 → 转回 Responses
  │                              │
  └─ /v1/messages ───────→ messages.py → 直接透传 ────────────→ /v1/messages
```

## 快速开始

### 直接运行

```bash
cd api2cursor
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入中转站地址和密钥
python start.py
```

### Docker 部署

```bash
cd api2cursor
cp .env.example .env
# 编辑 .env
docker compose up -d
```

服务启动后访问 `http://localhost:3029/admin` 进入管理面板。

## 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `PROXY_TARGET_URL` | 上游中转站地址 | `https://api.anthropic.com` |
| `PROXY_API_KEY` | 上游 API 密钥 | |
| `PROXY_PORT` | 服务监听端口 | `3029` |
| `API_TIMEOUT` | 请求超时（秒） | `300` |
| `ACCESS_API_KEY` | 访问鉴权密钥，留空不启用 | |
| `DEBUG` | 调试模式，输出详细请求/响应日志 | `false` |

### 模型映射

在管理面板 (`/admin`) 中配置模型映射：

- **Cursor 模型名** — 在 Cursor 自定义模型中填入的名称
- **上游模型名** — 发送到中转站的实际模型名
- **后端类型** — `openai` (CC 格式) / `anthropic` (Messages 格式) / `auto` (自动检测)
- **自定义地址/密钥** — 可选，覆盖全局设置，实现分流到不同中转站

**示例**：在 Cursor 中添加 `claude-sonnet-4-5-20250929`，映射到上游 `gpt-5.3-codex`，后端选 `openai`。Cursor 会用 CC 格式发送请求，代理直接转发到中转站的 `/v1/chat/completions`。

> **提示**：使用 Claude 风格的模型名（如 `claude-sonnet-4-5-20250929`）可以让 Cursor 显示思考过程（thinking）。

### 在 Cursor 中配置

1. 打开 Cursor 设置 → Models
2. 添加自定义模型，名称填映射中配置的 Cursor 模型名
3. Override OpenAI Base URL 填 `http://localhost:3029`
4. API Key 填 `ACCESS_API_KEY` 的值（未配置则随意填）

## 项目结构

```
api2cursor/
├── start.py               # 启动入口
├── app.py                 # Flask 应用工厂
├── config.py              # 环境变量配置
├── settings.py            # 持久化配置管理
├── routes/                # 路由层
│   ├── chat.py            #   /v1/chat/completions
│   ├── responses.py       #   /v1/responses
│   ├── messages.py        #   /v1/messages (透传)
│   └── admin.py           #   管理面板 + API
├── adapters/              # 适配层（格式转换）
│   ├── openai_anthropic.py#   CC ↔ Messages 双向转换
│   ├── openai_fixer.py    #   OpenAI 请求/响应修复
│   └── responses_adapter.py#  Responses ↔ CC 双向转换
├── utils/                 # 工具层
│   ├── http.py            #   请求转发、SSE 解析
│   ├── tool_fixer.py      #   工具参数修复
│   └── think_tag.py       #   <think> 标签提取
└── static/                # 管理面板前端
    ├── admin.html
    ├── admin.css
    └── admin.js
```

## 兼容性修复

代理自动处理以下兼容性问题：

- Cursor 扁平格式 tools → 标准 OpenAI 嵌套格式
- `reasoningContent` → `reasoning_content`
- `<think>` 标签 → `reasoning_content`
- 旧版 `function_call` → 新版 `tool_calls`
- `tool_calls` 缺失 `id` / `index` / `type` 字段补全
- 智能引号 → 普通引号（StrReplace 工具精确匹配修复）
- `file_path` → `path` 字段映射
- `finish_reason` 修正

## 许可证

[MIT](LICENSE)

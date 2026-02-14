# sparouter: LiteLLM Proxy + Intent Router（AIRouter 风格最小可运行版）

这个工程实现一个“只暴露 `/v1/chat/completions`”的路由层：

- **LiteLLM Proxy**：作为 OpenAI-compatible 网关，承载多 deployment 负载均衡 + 重试 + failover。
- **Intent Router（FastAPI）**：规则识别意图，覆盖虚拟模型组（`GENERAL/CODE/IMAGE/VIDEO`），并透传所有字段（包括 `thinking`、`video_config`、多模态 `messages.content`、`stream=true` SSE）。
- **动态策略增强**：
  - 动态健康感知（成功率/延迟窗口）
  - 熔断与冷却（连续失败触发 cooldown）
  - 权重/延迟感知调度（BYPASS 模式演示）
  - 成本/预算控制（日预算阈值触发 GENERAL 降级）
  - 基础限流（按 `X-Client-Id + Authorization`）

> 说明：默认模式优先通过 LiteLLM；动态“按渠道跳过/熔断”在 `BYPASS_LITELLM=true` 时可完整演示。

## 文件树

```text
.
├── .github/workflows/docker-build.yml
├── docker-compose.yml
├── litellm/config.yaml
├── README.md
├── router-fastapi
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
└── router-nextjs
    ├── Dockerfile
    ├── README.md
    ├── next.config.js
    ├── package.json
    ├── pages/api/v1/chat/completions.ts
    └── tsconfig.json
```

## 环境变量清单

### Router（FastAPI）

- `ROUTER_LISTEN_PORT=8080`
- `UPSTREAM_BASE_URL=http://litellm:4000`（默认走 LiteLLM）
- `BYPASS_LITELLM=false|true`（`true` 时路由器直接按渠道转发）
- `RAW_UPSTREAM_BASE_URL=https://your-proxy/v1`（BYPASS 模式兜底）
- `REDIS_URL=redis://redis:6379/0`（可选，不配则内存模式）
- `RPM_LIMIT=60`
- `CIRCUIT_FAIL_THRESHOLD=3`
- `CIRCUIT_COOLDOWN_SECONDS=30`
- `LATENCY_WINDOW=20`
- `BUDGET_DAILY_USD=5`
- `COST_TIER_GENERAL=low|mid|high`
- `LITELLM_MASTER_KEY=change-me`
- `GENERAL_MODEL_NAME=grok-4`
- `GENERAL_MODEL_NAME_CHEAP=grok-4-mini`
- `CODE_MODEL_NAME=codex-1`
- `IMAGE_MODEL_NAME=grok-imagine-image`
- `VIDEO_MODEL_NAME=grok-video-1`

### 上游渠道（LiteLLM + BYPASS 共用）

- `CHANNEL_A_BASE_URL`
- `CHANNEL_A_API_KEY`
- `CHANNEL_B_BASE_URL`
- `CHANNEL_B_API_KEY`

## 意图识别规则

- `VIDEO`
  - `video_config` 存在，或文本包含：`生成视频/视频超分/超清/分辨率/时长/宽高比/帧率`
- `IMAGE`
  - 文本包含：`生成图片/画一张/出图/图片编辑/换风格/抠图/扩图`
  - 或 `messages.content` 为数组且含 `image_url/file`（且无 `video_config`）
- `CODE`
  - 文本包含：`写代码/实现/bug/报错/TypeError/Exception/stack trace/SQL/正则/算法/单元测试/项目结构/README/部署`
  - 或内容含代码块 ```` ``` ````
- `GENERAL`
  - 其它情况

## 运行

### 1) 启动

```bash
docker-compose up -d
```

### 2) 调用验证

#### GENERAL

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Client-Id: demo-general' \
  -d '{
    "model":"anything",
    "messages":[{"role":"user","content":"你好，做个自我介绍"}]
  }'
```

#### CODE

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Client-Id: demo-code' \
  -d '{
    "model":"anything",
    "messages":[{"role":"user","content":"请写一段 Python 快排并附单元测试"}]
  }'
```

#### IMAGE（chat/completions，多模态）

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Client-Id: demo-image' \
  -d '{
    "model":"anything",
    "messages":[{
      "role":"user",
      "content":[
        {"type":"text","text":"把这张图改成赛博朋克风格"},
        {"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}}
      ]
    }]
  }'
```

#### VIDEO（chat/completions + video_config）

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Client-Id: demo-video' \
  -d '{
    "model":"anything",
    "video_config":{"duration":4,"fps":24,"resolution":"1280x720"},
    "messages":[{"role":"user","content":"生成一个海边日落短视频"}]
  }'
```

#### SSE stream=true

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Client-Id: demo-stream' \
  -d '{
    "model":"anything",
    "stream":true,
    "messages":[{"role":"user","content":"请流式回答：什么是服务熔断"}]
  }'
```

### 3) 故障切换演示（LiteLLM 层）

1. 把 `CHANNEL_A_BASE_URL` 设为不可达地址并重启 `litellm`。
2. 连续发起请求，观察 LiteLLM 自动重试/failover 到 `CHANNEL_B`。

### 4) 熔断演示（Router BYPASS 模式）

1. 设置 `BYPASS_LITELLM=true`。
2. 让某个渠道持续返回 `500` 或超时。
3. 连续失败达到 `CIRCUIT_FAIL_THRESHOLD` 后，路由器将其标记 cooldown，`CIRCUIT_COOLDOWN_SECONDS` 内跳过该渠道。

### 5) Redis 增强模式

```bash
docker-compose --profile redis up -d
export REDIS_URL=redis://redis:6379/0
```

不配置 Redis 时自动回退到内存模式。

## Vercel 调试（router-nextjs）

- 部署 `router-nextjs` 到 Vercel。
- 设置环境变量：
  - `UPSTREAM_BASE_URL`（公网 LiteLLM 或上游中转站）
  - `LITELLM_MASTER_KEY`（如果上游要求）
- 调试端点：
  - `POST /api/v1/chat/completions`（Next.js 原生 API 路径）
  - `POST /v1/chat/completions`（已通过 rewrite 映射，便于 OpenAI-compatible 客户端）
  - `GET /healthz`（rewrite 到 API health）
- 选用 **Node runtime**，避免 Edge 在 SSE 透传场景下出现兼容性波动。


### Vercel Monorepo 404 说明（关键）

如果日志里 `/`、`/favicon.ico`、`/favicon.png` 全部 404，优先怀疑 Vercel 部署根目录指错（没有构建到 `router-nextjs`）。
本仓库已新增根目录 `vercel.json`，显式要求以 `router-nextjs/package.json` 构建 Next.js，降低配置错误概率。

### Vercel 常见 307/401 说明

如果日志显示 `GET / -> 307` 随后 `401`，通常是 Vercel 的 Deployment Protection 在平台层拦截，不是 Next.js 路由代码错误。
请在项目设置中检查并按团队策略调整保护配置；本仓库已补充主页、`/healthz`、`/v1/chat/completions` rewrite 与 favicon rewrite，避免无关 404 噪声。

## GHCR 构建发布（GitHub Actions）

Workflow：`.github/workflows/docker-build.yml`

- 触发：`push main`、`push tag(v*)`、手动触发。
- 默认使用 `secrets.GITHUB_TOKEN` 登录 GHCR。
- 会构建并推送：
  - `ghcr.io/<owner>/<repo>/router-fastapi:latest`
  - `ghcr.io/<owner>/<repo>/router-fastapi:<sha7>`
  - （可选）`router-nextjs` 镜像

若组织策略限制 `GITHUB_TOKEN` 推包，请改为 PAT 并替换 login step 的 password secret。

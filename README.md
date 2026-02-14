# sparouter - AI Router 控制台（WebUI + Router API + LiteLLM）

这是一个“AI Router 平台”最小可运行雏形，而非聊天产品。

## 组件
- `router-nextjs`：Web 控制台（Providers / Routing / Classifier / Generate / Playground）
- `router-fastapi`：后端 API + Router Runtime（`/v1/chat/completions`）
- `litellm`：LiteLLM Proxy

## 功能
1. Provider 管理（新增/编辑/删除、健康检查、模型同步、手动模型导入）
2. Routing 管理（组、deployment、weight/priority/fallback）
3. Classifier 管理（rules_only/classifier_only/hybrid + 测试台）
4. 配置生成（`litellm/config.yaml`、`router/rules.json`、`router/settings.json`）+ 应用
5. Playground 调试路由结果

## 安全
- api_key 仅后端存储（使用 `APP_SECRET` 做对称混淆加密）
- 前端和列表 API 不返回明文 key（仅显示掩码）
- `/api/apply` 需 `X-Admin-Token`

## API 清单
- Providers
  - GET `/api/providers`
  - POST `/api/providers`
  - PUT `/api/providers/{id}`
  - DELETE `/api/providers/{id}`
  - POST `/api/providers/{id}/sync-models`
  - GET `/api/providers/{id}/models`
  - POST `/api/providers/{id}/health-check`
- Routing
  - GET `/api/routing/groups`
  - POST `/api/routing/groups`
  - PUT `/api/routing/groups/{group}`
  - DELETE `/api/routing/groups/{group}`
  - POST `/api/routing/groups/{group}/deployments`
  - PUT `/api/routing/deployments/{id}`
  - DELETE `/api/routing/deployments/{id}`
- Classifier
  - GET `/api/classifier/config`
  - PUT `/api/classifier/config`
  - POST `/api/classifier/test`
- Generate
  - POST `/api/generate`
  - POST `/api/apply`
- Runtime
  - POST `/v1/chat/completions`

## 存储模式
- `STORAGE_MODE=file`：本地文件 `STATE_FILE`
- `STORAGE_MODE=sqlite`：SQLite（`DATABASE_URL=sqlite:///data/router.db`）

## 快速启动
```bash
cp .env.example .env
docker compose up -d --build
```

- 控制台：`http://localhost:3000`
- API：`http://localhost:8080`
- LiteLLM：`http://localhost:4000`

## Vercel 调试 WebUI
仅部署 `router-nextjs`，设置：
- `NEXT_PUBLIC_API_BASE=https://your-public-api`

## 生产建议
- 生产推荐 Docker 部署 FastAPI + LiteLLM
- 可扩展将 SQLite 切到 Postgres（可在后续接入 SQLAlchemy）

## 单容器部署（保留分容器优势）
你可以继续使用分容器（推荐生产职责分离），也可以在云平台一键上线单容器：

- 单容器镜像：`router-monolith`（FastAPI + 打包后的 Next.js 静态控制台）
- 单容器对外只开一个端口（默认 8080）：
  - `/`、`/providers`、`/routing`、`/classifier`、`/generate`、`/playground` -> WebUI
  - `/api/*`、`/v1/*` -> Router API/Runtime

本地试单容器：
```bash
docker compose --profile single up -d --build allinone litellm
```

> 这样可以“更容易上线”；同时保留了分容器模式的优势（独立扩缩容、独立发布、资源隔离）。

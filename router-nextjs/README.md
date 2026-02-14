# router-nextjs（Vercel 调试）

这是用于 Vercel 实时调试的 Next.js 版本意图路由器，使用 **Node runtime API Route**（不是 Edge），因为 SSE 透传在 Node runtime 下更稳定。

## 为什么你会看到 404

如果只实现了 API 路由（`/api/...`）但没有首页，那么访问 `/` 会返回 404。
这不代表框架选错，通常说明“部署成功但没有页面路由”。

本目录已补齐：

- `pages/index.tsx`：根路径可访问的说明页
- `rewrites`：把 `/v1/chat/completions` 重写到 `/api/v1/chat/completions`
- `GET /healthz`：重写到 `/api/healthz`


## 为什么会出现 307 + 401（即使代码没问题）

如果 Vercel 项目开启了 **Deployment Protection / Vercel Authentication**，请求会先被平台层重定向（常见 307）再拦截（401）。
这发生在你的应用代码之前，所以即使 Next.js 路由正常，也会先看到 307/401。

可选处理：

- 在 Vercel 项目设置里关闭该项目（或 Preview 环境）的保护；或
- 使用带保护绕过参数/令牌的访问方式（团队策略允许时）；或
- 直接测试 API 路径并带授权（如果你们组织策略要求）。

新增兼容项：

- 已提供 `/favicon.svg`，并把 `/favicon.ico`、`/favicon.png` 重写到该文件，减少浏览器默认图标请求导致的 404 噪声。

## 本地运行

```bash
npm install
npm run dev
```

## 关键环境变量

- `UPSTREAM_BASE_URL`：上游地址（通常是公网 LiteLLM）
- `LITELLM_MASTER_KEY`：可选，LiteLLM 网关 key

## API

- `POST /api/v1/chat/completions`（原生 API 路径）
- `POST /v1/chat/completions`（推荐：OpenAI-compatible 风格）
- `GET /healthz`

## Vercel 框架选择建议

- Framework Preset: **Next.js**
- Root Directory: `router-nextjs`
- Build Command: `npm run build`（默认）
- Output: Next.js 默认输出（不要改成静态导出）

如果你看到 `/` 404 + `/favicon.ico` 404，但 `/api/...` 可以访问，那么主要是页面资源缺失，不是 Next.js 框架选错。

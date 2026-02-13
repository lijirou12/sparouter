# router-nextjs（Vercel 调试）

这是一个用于实时调试的 Next.js API Route 版本意图路由器。默认使用 **Node runtime**（不是 Edge），因为流式 SSE 透传在 Node runtime 下更稳定。

## 本地运行

```bash
npm install
npm run dev
```

## 关键环境变量

- `UPSTREAM_BASE_URL`：上游地址（通常是公网 LiteLLM）
- `LITELLM_MASTER_KEY`：可选，LiteLLM 网关 key

## API

- `POST /api/v1/chat/completions`

请求体直接兼容 OpenAI `chat/completions`，会自动将 model 覆盖为：`GENERAL/CODE/IMAGE/VIDEO`。

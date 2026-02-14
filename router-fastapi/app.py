import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

try:
    import redis.asyncio as redis
except Exception:  # Redis is optional
    redis = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("router")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).lower().strip()
    return raw in {"1", "true", "yes", "on"}


def utc_date() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


ROUTER_LISTEN_PORT = int(os.getenv("ROUTER_LISTEN_PORT", "8080"))
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "http://litellm:4000").rstrip("/")
BYPASS_LITELLM = env_bool("BYPASS_LITELLM", False)
RAW_UPSTREAM_BASE_URL = os.getenv("RAW_UPSTREAM_BASE_URL", "https://your-proxy/v1").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "").strip()
RPM_LIMIT = int(os.getenv("RPM_LIMIT", "60"))
CIRCUIT_FAIL_THRESHOLD = int(os.getenv("CIRCUIT_FAIL_THRESHOLD", "3"))
CIRCUIT_COOLDOWN_SECONDS = int(os.getenv("CIRCUIT_COOLDOWN_SECONDS", "30"))
LATENCY_WINDOW = int(os.getenv("LATENCY_WINDOW", "20"))
BUDGET_DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "5"))
COST_TIER_GENERAL = os.getenv("COST_TIER_GENERAL", "mid").lower()

LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "")

GENERAL_MODEL_NAME = os.getenv("GENERAL_MODEL_NAME", "grok-4")
GENERAL_MODEL_NAME_CHEAP = os.getenv("GENERAL_MODEL_NAME_CHEAP", "grok-4-mini")
CODE_MODEL_NAME = os.getenv("CODE_MODEL_NAME", "codex-1")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "grok-imagine-image")
VIDEO_MODEL_NAME = os.getenv("VIDEO_MODEL_NAME", "grok-video-1")

CODE_PATTERNS = [
    r"写代码", r"实现", r"bug", r"报错", r"typeerror", r"exception", r"stack\s*trace",
    r"sql", r"正则", r"算法", r"单元测试", r"项目结构", r"readme", r"部署",
]
IMAGE_PATTERNS = [
    r"生成图片", r"画一张", r"出图", r"图片编辑", r"换风格", r"抠图", r"扩图",
]
VIDEO_PATTERNS = [
    r"生成视频", r"视频超分", r"超清", r"分辨率", r"时长", r"宽高比", r"帧率",
]

COST_BY_GROUP = {
    "GENERAL": {"low": 0.002, "mid": 0.006, "high": 0.012},
    "CODE": {"low": 0.003, "mid": 0.008, "high": 0.014},
    "IMAGE": {"low": 0.010, "mid": 0.030, "high": 0.060},
    "VIDEO": {"low": 0.050, "mid": 0.180, "high": 0.400},
}


@dataclass
class ChannelStats:
    latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0

    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 1.0

    def avg_latency(self) -> float:
        if not self.latencies_ms:
            return 500.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def is_cooldown(self) -> bool:
        return time.time() < self.cooldown_until


class StateStore:
    def __init__(self):
        self._redis = None
        self._rpm: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._budget: Dict[str, float] = defaultdict(float)
        self._stats: Dict[str, ChannelStats] = defaultdict(ChannelStats)

    async def init(self):
        if REDIS_URL and redis is not None:
            try:
                self._redis = redis.from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
                logger.info("Redis mode enabled")
            except Exception:
                logger.warning("Redis unavailable, fallback to memory mode")
                self._redis = None
        else:
            logger.info("Memory mode enabled")

    async def close(self):
        if self._redis:
            await self._redis.close()

    async def rpm_allow(self, key: str, limit: int) -> bool:
        now = time.time()
        if self._redis:
            pipe = self._redis.pipeline()
            zkey = f"rpm:{key}"
            pipe.zremrangebyscore(zkey, 0, now - 60)
            pipe.zcard(zkey)
            pipe.zadd(zkey, {str(now): now})
            pipe.expire(zkey, 120)
            _, count, _, _ = await pipe.execute()
            return int(count) < limit

        q = self._rpm[key]
        while q and q[0] < now - 60:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    async def add_budget(self, day_key: str, amount: float) -> float:
        if self._redis:
            key = f"budget:{day_key}"
            new_val = await self._redis.incrbyfloat(key, amount)
            await self._redis.expire(key, 172800)
            return float(new_val)

        self._budget[day_key] += amount
        return self._budget[day_key]

    async def get_budget(self, day_key: str) -> float:
        if self._redis:
            key = f"budget:{day_key}"
            val = await self._redis.get(key)
            return float(val or 0.0)
        return self._budget.get(day_key, 0.0)

    async def get_stats(self, channel: str) -> ChannelStats:
        return self._stats[channel]


store = StateStore()


class AdminStore:
    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.router_config: Dict[str, Any] = {
            "intent_mode": "rule",  # rule | llm
            "selector_provider_id": "",
            "selector_model": "",
            "llm_timeout_seconds": 15,
        }

    def list_providers(self) -> List[Dict[str, Any]]:
        return list(self.providers.values())

    def upsert_provider(self, provider: Dict[str, Any]) -> Dict[str, Any]:
        pid = provider.get("id") or str(uuid.uuid4())[:8]
        provider["id"] = pid
        self.providers[pid] = {
            "id": pid,
            "name": provider.get("name") or pid,
            "base_url": str(provider.get("base_url", "")).rstrip("/"),
            "api_key": provider.get("api_key", ""),
            "enabled": bool(provider.get("enabled", True)),
            "tags": provider.get("tags", []),
            "default_model": provider.get("default_model", ""),
            "last_models": provider.get("last_models", []),
        }
        return self.providers[pid]

    def delete_provider(self, pid: str) -> bool:
        return self.providers.pop(pid, None) is not None


admin_store = AdminStore()


def _mask_provider(p: Dict[str, Any]) -> Dict[str, Any]:
    masked = dict(p)
    key = masked.get("api_key", "")
    masked["api_key"] = ("***" + key[-4:]) if key else ""
    return masked


async def detect_intent(payload: Dict[str, Any]) -> str:
    mode = admin_store.router_config.get("intent_mode", "rule")
    if mode != "llm":
        return detect_intent_rule(payload)

    provider_id = admin_store.router_config.get("selector_provider_id", "")
    selector_model = admin_store.router_config.get("selector_model", "")
    provider = admin_store.providers.get(provider_id)
    if not provider or not provider.get("enabled") or not selector_model:
        return detect_intent_rule(payload)

    text = flatten_text(payload.get("messages", []))
    prompt = (
        "你是意图分类器。只输出以下四个标签之一，不要输出其它字符："
        "GENERAL, CODE, IMAGE, VIDEO。\n"
        f"用户内容:\n{text}"
    )

    url = f"{provider['base_url']}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    body = {
        "model": selector_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "你只返回GENERAL/CODE/IMAGE/VIDEO之一。"},
            {"role": "user", "content": prompt},
        ],
    }

    timeout = httpx.Timeout(float(admin_store.router_config.get("llm_timeout_seconds", 15)), connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                return detect_intent_rule(payload)
            data = resp.json()
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").upper()
            for tag in ("GENERAL", "CODE", "IMAGE", "VIDEO"):
                if tag in content:
                    return tag
    except Exception:
        pass
    return detect_intent_rule(payload)



def flatten_text(messages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    return "\n".join(chunks).lower()


def has_multimodal_file_or_image(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image_url", "file"}:
                    return True
    return False


def detect_intent_rule(payload: Dict[str, Any]) -> str:
    if payload.get("video_config") is not None:
        return "VIDEO"

    messages = payload.get("messages", [])
    text = flatten_text(messages)

    if re.search("|".join(VIDEO_PATTERNS), text, re.IGNORECASE):
        return "VIDEO"

    if has_multimodal_file_or_image(messages):
        return "IMAGE"

    if re.search("|".join(IMAGE_PATTERNS), text, re.IGNORECASE):
        return "IMAGE"

    if "```" in text or re.search("|".join(CODE_PATTERNS), text, re.IGNORECASE):
        return "CODE"

    return "GENERAL"


def estimate_cost(group: str) -> float:
    tier = COST_TIER_GENERAL if group == "GENERAL" else "mid"
    return COST_BY_GROUP.get(group, COST_BY_GROUP["GENERAL"]).get(tier, 0.006)


def pick_bypass_channel(group: str) -> Optional[str]:
    candidates = ["CHANNEL_A", "CHANNEL_B"]
    weighted: List[Tuple[str, float]] = []

    for channel in candidates:
        stats = store._stats[channel]
        if stats.is_cooldown():
            continue
        base_weight = 1.0
        health_factor = max(0.05, stats.success_rate())
        latency_factor = min(2.0, 500.0 / max(50.0, stats.avg_latency()))
        score = base_weight * health_factor * latency_factor
        weighted.append((channel, score))

    if not weighted:
        return None

    total = sum(score for _, score in weighted)
    r = random.random() * total
    upto = 0.0
    for channel, score in weighted:
        upto += score
        if upto >= r:
            return channel
    return weighted[-1][0]


def build_target(payload: Dict[str, Any], group: str, chosen_channel: Optional[str]) -> Tuple[str, Dict[str, str]]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if LITELLM_MASTER_KEY and not BYPASS_LITELLM:
        headers["Authorization"] = f"Bearer {LITELLM_MASTER_KEY}"

    if BYPASS_LITELLM:
        channel = chosen_channel or "CHANNEL_A"
        ch_base = os.getenv(f"{channel}_BASE_URL", RAW_UPSTREAM_BASE_URL).rstrip("/")
        ch_key = os.getenv(f"{channel}_API_KEY", "")
        if ch_key:
            headers["Authorization"] = f"Bearer {ch_key}"
        model_by_group = {
            "GENERAL": GENERAL_MODEL_NAME,
            "CODE": CODE_MODEL_NAME,
            "IMAGE": IMAGE_MODEL_NAME,
            "VIDEO": VIDEO_MODEL_NAME,
        }
        payload["model"] = model_by_group.get(group, GENERAL_MODEL_NAME)
        return f"{ch_base}/v1/chat/completions", headers

    payload["model"] = group
    return f"{UPSTREAM_BASE_URL}/v1/chat/completions", headers


def response_headers(group: str, channel: str, budget: float) -> Dict[str, str]:
    return {
        "X-Selected-Group": group,
        "X-Selected-Deployment": channel,
        "X-Budget-Spent-USD": f"{budget:.4f}",
    }


app = FastAPI(title="sparouter-intent-router")


@app.on_event("startup")
async def startup_event():
    await store.init()


@app.on_event("shutdown")
async def shutdown_event():
    await store.close()




@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>sparouter 管理页</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; background:#0b1020; color:#e5e7eb; margin:0; }
      .wrap { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
      .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
      .card { background:#111827; border:1px solid #374151; border-radius:12px; padding:16px; margin-bottom:16px; }
      h1,h2 { margin:0 0 10px 0; }
      input,select,button,textarea { width:100%; margin:6px 0; padding:8px; border-radius:8px; border:1px solid #374151; background:#0f172a; color:#e5e7eb; }
      button { cursor:pointer; background:#1d4ed8; }
      pre { background:#0f172a; border-radius:8px; padding:10px; overflow:auto; }
      .row { display:flex; gap:8px; }
      .row > * { flex:1; }
      .muted { color:#9ca3af; }
      @media (max-width: 900px){ .grid{ grid-template-columns:1fr; } }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>sparouter 管理页</h1>
        <div class="muted">在线导入供应商、拉取模型、配置小模型智能路由（rule/llm）。</div>
      </div>
      <div class="grid">
        <div class="card">
          <h2>新增/更新供应商</h2>
          <input id="pid" placeholder="id(留空自动生成)" />
          <input id="pname" placeholder="name" />
          <input id="purl" placeholder="base_url, 如 https://api.xxx.com" />
          <input id="pkey" placeholder="api_key" />
          <input id="pmodel" placeholder="default_model(可选)" />
          <button onclick="saveProvider()">保存供应商</button>
          <pre id="providers">加载中...</pre>
        </div>
        <div class="card">
          <h2>智能意图小模型配置</h2>
          <select id="intent_mode"><option value="rule">rule(规则)</option><option value="llm">llm(小模型分类)</option></select>
          <input id="selector_provider_id" placeholder="selector_provider_id" />
          <input id="selector_model" placeholder="selector_model" />
          <input id="llm_timeout_seconds" type="number" value="15" />
          <button onclick="saveConfig()">保存路由配置</button>
          <pre id="cfg"></pre>
        </div>
      </div>
      <div class="card">
        <h2>工具</h2>
        <div class="row">
          <input id="discover_id" placeholder="provider_id" />
          <button onclick="discoverModels()">拉取模型列表</button>
        </div>
        <pre id="discover"></pre>
      </div>
    </div>
    <script>
      async function j(url, opt={}){ const r = await fetch(url, Object.assign({headers:{'Content-Type':'application/json'}}, opt)); return {ok:r.ok, status:r.status, data: await r.json().catch(()=>({}))}; }
      async function load(){
        const ps = await j('/admin/providers');
        document.getElementById('providers').textContent = JSON.stringify(ps.data, null, 2);
        const cfg = await j('/admin/router-config');
        document.getElementById('cfg').textContent = JSON.stringify(cfg.data, null, 2);
        if (cfg.data.intent_mode) document.getElementById('intent_mode').value = cfg.data.intent_mode;
        document.getElementById('selector_provider_id').value = cfg.data.selector_provider_id || '';
        document.getElementById('selector_model').value = cfg.data.selector_model || '';
        document.getElementById('llm_timeout_seconds').value = cfg.data.llm_timeout_seconds || 15;
      }
      async function saveProvider(){
        const body = {id:pid.value.trim(), name:pname.value.trim(), base_url:purl.value.trim(), api_key:pkey.value.trim(), default_model:pmodel.value.trim(), enabled:true};
        await j('/admin/providers', {method:'POST', body: JSON.stringify(body)}); load();
      }
      async function saveConfig(){
        const body = {intent_mode:intent_mode.value, selector_provider_id:selector_provider_id.value.trim(), selector_model:selector_model.value.trim(), llm_timeout_seconds:Number(llm_timeout_seconds.value||15)};
        await j('/admin/router-config', {method:'POST', body: JSON.stringify(body)}); load();
      }
      async function discoverModels(){
        const id = discover_id.value.trim();
        const out = await j(`/admin/providers/${id}/discover-models`, {method:'POST'});
        document.getElementById('discover').textContent = JSON.stringify(out, null, 2);
        load();
      }
      load();
    </script>
  </body>
</html>"""


@app.get("/status.json")
async def status_json():
    return {
        "service": "sparouter-intent-router",
        "status": "ok",
        "endpoints": ["/", "/healthz", "/v1/chat/completions", "/status.json", "/admin/providers", "/admin/router-config"],
    }


@app.get("/admin/providers")
async def admin_list_providers():
    return {"providers": [_mask_provider(p) for p in admin_store.list_providers()]}


@app.post("/admin/providers")
async def admin_save_provider(req: Request):
    body = await req.json()
    if not body.get("base_url"):
        raise HTTPException(status_code=400, detail="base_url is required")
    saved = admin_store.upsert_provider(body)
    return {"provider": _mask_provider(saved)}


@app.delete("/admin/providers/{provider_id}")
async def admin_delete_provider(provider_id: str):
    ok = admin_store.delete_provider(provider_id)
    if not ok:
        raise HTTPException(status_code=404, detail="provider not found")
    return {"deleted": provider_id}


@app.post("/admin/providers/{provider_id}/discover-models")
async def admin_discover_models(provider_id: str):
    p = admin_store.providers.get(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail="provider not found")

    headers = {"Content-Type": "application/json"}
    if p.get("api_key"):
        headers["Authorization"] = f"Bearer {p['api_key']}"

    urls = [f"{p['base_url']}/v1/models", f"{p['base_url']}/models"]
    models: List[str] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
        for u in urls:
            try:
                r = await client.get(u, headers=headers)
                if r.status_code < 400:
                    data = r.json()
                    for item in data.get("data", []):
                        mid = item.get("id")
                        if mid:
                            models.append(mid)
                    if models:
                        break
            except Exception:
                continue

    p["last_models"] = sorted(set(models))
    return {"provider_id": provider_id, "models": p["last_models"]}


@app.get("/admin/router-config")
async def admin_get_router_config():
    return admin_store.router_config


@app.post("/admin/router-config")
async def admin_set_router_config(req: Request):
    body = await req.json()
    mode = body.get("intent_mode", "rule")
    if mode not in {"rule", "llm"}:
        raise HTTPException(status_code=400, detail="intent_mode must be rule or llm")
    admin_store.router_config.update(
        {
            "intent_mode": mode,
            "selector_provider_id": body.get("selector_provider_id", ""),
            "selector_model": body.get("selector_model", ""),
            "llm_timeout_seconds": int(body.get("llm_timeout_seconds", 15)),
        }
    )
    return admin_store.router_config


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "redis": bool(store._redis)}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()

    client_id = req.headers.get("X-Client-Id") or req.client.host or "anonymous"
    api_key_header = req.headers.get("Authorization", "")
    api_key_hash = hashlib.sha256(api_key_header.encode()).hexdigest()[:12] if api_key_header else "nokey"
    rate_key = f"{client_id}:{api_key_hash}"

    allowed = await store.rpm_allow(rate_key, RPM_LIMIT)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"RPM limit exceeded ({RPM_LIMIT})")

    group = await detect_intent(payload)
    day_key = utc_date()
    estimated = estimate_cost(group)
    spent = await store.get_budget(day_key)

    if spent >= BUDGET_DAILY_USD and group == "GENERAL":
        payload["model"] = GENERAL_MODEL_NAME_CHEAP

    chosen_channel = "LITELLM_POOL"
    if BYPASS_LITELLM:
        bypass = pick_bypass_channel(group)
        if bypass is None:
            raise HTTPException(status_code=503, detail="All channels are in cooldown")
        chosen_channel = bypass

    target_url, out_headers = build_target(payload, group, chosen_channel if BYPASS_LITELLM else None)
    out_headers["X-Route-Group"] = group
    out_headers["X-Client-Id"] = client_id
    out_headers["X-Request-Mode"] = "bypass" if BYPASS_LITELLM else "litellm"

    is_stream = bool(payload.get("stream", False))
    timeout = httpx.Timeout(150.0, connect=10.0)
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if is_stream:
                upstream = await client.stream("POST", target_url, headers=out_headers, json=payload)

                async def stream_iter():
                    try:
                        async for chunk in upstream.aiter_bytes():
                            if chunk:
                                yield chunk
                    finally:
                        await upstream.aclose()

                elapsed_ms = (time.perf_counter() - start) * 1000
                if BYPASS_LITELLM:
                    stats = await store.get_stats(chosen_channel)
                    stats.latencies_ms.append(elapsed_ms)
                    stats.successes += 1
                    stats.consecutive_failures = 0

                spent = await store.add_budget(day_key, estimated)
                return StreamingResponse(
                    stream_iter(),
                    media_type="text/event-stream",
                    status_code=upstream.status_code,
                    headers=response_headers(group, chosen_channel, spent),
                )

            upstream = await client.post(target_url, headers=out_headers, json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if BYPASS_LITELLM:
                stats = await store.get_stats(chosen_channel)
                stats.latencies_ms.append(elapsed_ms)
                if upstream.status_code < 400:
                    stats.successes += 1
                    stats.consecutive_failures = 0
                else:
                    stats.failures += 1
                    stats.consecutive_failures += 1
                    if (
                        upstream.status_code in (429, 500, 502, 503, 504)
                        and stats.consecutive_failures >= CIRCUIT_FAIL_THRESHOLD
                    ):
                        stats.cooldown_until = time.time() + CIRCUIT_COOLDOWN_SECONDS

            spent = await store.add_budget(day_key, estimated)
            return JSONResponse(
                status_code=upstream.status_code,
                content=upstream.json() if upstream.content else {},
                headers=response_headers(group, chosen_channel, spent),
            )

        except httpx.RequestError as exc:
            if BYPASS_LITELLM:
                stats = await store.get_stats(chosen_channel)
                stats.failures += 1
                stats.consecutive_failures += 1
                if stats.consecutive_failures >= CIRCUIT_FAIL_THRESHOLD:
                    stats.cooldown_until = time.time() + CIRCUIT_COOLDOWN_SECONDS
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(exc)}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=ROUTER_LISTEN_PORT)

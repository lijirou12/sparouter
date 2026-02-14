import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


def detect_intent(payload: Dict[str, Any]) -> str:
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




@app.get("/")
async def root():
    return {
        "service": "sparouter-intent-router",
        "status": "ok",
        "endpoints": ["/healthz", "/v1/chat/completions"],
    }


@app.get("/favicon.ico")
async def favicon():
    # Cloud runtimes and browsers often probe this path; return 204 to avoid noisy 404 logs.
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

    group = detect_intent(payload)
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

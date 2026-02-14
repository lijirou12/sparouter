import base64
import datetime as dt
import hashlib
import json
import logging
import os
import random
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

logger = logging.getLogger("router")
logging.basicConfig(level=logging.INFO)

# ---------------- env ----------------
APP_SECRET = os.getenv("APP_SECRET", "change-me")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-admin-token")
ROUTER_LISTEN_PORT = int(os.getenv("ROUTER_LISTEN_PORT", "8080"))
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "http://litellm:4000").rstrip("/")
BYPASS_LITELLM = os.getenv("BYPASS_LITELLM", "false").lower() in {"1", "true", "yes"}
RAW_UPSTREAM_BASE_URL = os.getenv("RAW_UPSTREAM_BASE_URL", "https://your-proxy.example.com").rstrip("/")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "")

STORAGE_MODE = os.getenv("STORAGE_MODE", "sqlite")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/router.db")
STATE_FILE = Path(os.getenv("STATE_FILE", "./data/state.json"))
RPM_LIMIT = int(os.getenv("RPM_LIMIT", "60"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

# --------------- crypto --------------
def _kstream(secret: str, n: int) -> bytes:
    seed = hashlib.sha256(secret.encode()).digest()
    out = b""
    i = 0
    while len(out) < n:
        out += hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        i += 1
    return out[:n]


def encrypt_text(plain: str) -> str:
    if not plain:
        return ""
    data = plain.encode()
    ks = _kstream(APP_SECRET, len(data))
    cipher = bytes([a ^ b for a, b in zip(data, ks)])
    return base64.b64encode(cipher).decode()


def decrypt_text(cipher_b64: str) -> str:
    if not cipher_b64:
        return ""
    data = base64.b64decode(cipher_b64.encode())
    ks = _kstream(APP_SECRET, len(data))
    plain = bytes([a ^ b for a, b in zip(data, ks)])
    return plain.decode(errors="ignore")


# --------- intent rules (baseline) ---------
CODE_PATTERNS = [r"写代码", r"bug", r"报错", r"typeerror", r"exception", r"stack\s*trace", r"sql", r"算法", r"单元测试", r"部署"]
IMAGE_PATTERNS = [r"生成图片", r"画一张", r"出图", r"图片编辑", r"换风格", r"抠图", r"扩图"]
VIDEO_PATTERNS = [r"生成视频", r"视频超分", r"超清", r"分辨率", r"时长", r"宽高比", r"帧率"]


def flatten_text(messages: List[Dict[str, Any]]) -> str:
    chunks = []
    for msg in messages or []:
        c = msg.get("content")
        if isinstance(c, str):
            chunks.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    return "\n".join(chunks).lower()


def has_multimodal(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages or []:
        c = msg.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") in {"image_url", "file"}:
                    return True
    return False


def detect_intent_rule(payload: Dict[str, Any], groups: List[str]) -> Dict[str, Any]:
    if payload.get("video_config") is not None and "VIDEO" in groups:
        return {"group": "VIDEO", "source": "rule", "confidence": 1.0}

    text = flatten_text(payload.get("messages", []))
    if re.search("|".join(VIDEO_PATTERNS), text, re.IGNORECASE) and "VIDEO" in groups:
        return {"group": "VIDEO", "source": "rule", "confidence": 0.95}
    if has_multimodal(payload.get("messages", [])) and "IMAGE" in groups:
        return {"group": "IMAGE", "source": "rule", "confidence": 0.9}
    if re.search("|".join(IMAGE_PATTERNS), text, re.IGNORECASE) and "IMAGE" in groups:
        return {"group": "IMAGE", "source": "rule", "confidence": 0.85}
    if ("```" in text or re.search("|".join(CODE_PATTERNS), text, re.IGNORECASE)) and "CODE" in groups:
        return {"group": "CODE", "source": "rule", "confidence": 0.85}
    return {"group": "GENERAL" if "GENERAL" in groups else (groups[0] if groups else "GENERAL"), "source": "rule", "confidence": 0.5}


# --------------- storage ----------------
class Storage:
    def __init__(self):
        self.mode = STORAGE_MODE
        self._rpm: Dict[str, List[float]] = {}
        if self.mode == "sqlite":
            self.db_path = DATABASE_URL.replace("sqlite:///", "")
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.init_sqlite()
        else:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not STATE_FILE.exists():
                STATE_FILE.write_text(json.dumps(self.default_state(), ensure_ascii=False, indent=2))

    def default_state(self):
        return {
            "providers": [],
            "models": [],
            "routing_groups": [
                {"name": "GENERAL", "enabled": True, "tag_filter": ""},
                {"name": "CODE", "enabled": True, "tag_filter": ""},
                {"name": "IMAGE", "enabled": True, "tag_filter": ""},
                {"name": "VIDEO", "enabled": True, "tag_filter": ""},
            ],
            "deployments": [],
            "classifier": {
                "classifier_provider": "",
                "classifier_model": "",
                "classifier_mode": "rules_only",
                "low_confidence_threshold": 0.7,
            },
            "route_logs": [],
        }

    # ----- file mode helpers
    def read_state(self):
        return json.loads(STATE_FILE.read_text())

    def write_state(self, data):
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ----- sqlite init
    def init_sqlite(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS providers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                api_key_enc TEXT,
                headers_json TEXT,
                enabled INTEGER DEFAULT 1,
                tags_json TEXT,
                last_health_json TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                provider_id TEXT,
                model_name TEXT,
                model_type TEXT,
                context_length INTEGER,
                note TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS routing_groups (
                name TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 1,
                tag_filter TEXT
            );
            CREATE TABLE IF NOT EXISTS deployments (
                id TEXT PRIMARY KEY,
                group_name TEXT,
                provider_id TEXT,
                model_name TEXT,
                weight INTEGER,
                priority INTEGER,
                fallback_on TEXT,
                enabled INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS classifier_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                classifier_provider TEXT,
                classifier_model TEXT,
                classifier_mode TEXT,
                low_confidence_threshold REAL
            );
            CREATE TABLE IF NOT EXISTS route_logs (
                id TEXT PRIMARY KEY,
                request_id TEXT,
                group_name TEXT,
                provider_id TEXT,
                model_name TEXT,
                latency_ms REAL,
                status_code INTEGER,
                fallback_count INTEGER,
                created_at TEXT
            );
            """
        )
        self.conn.commit()
        # seed
        cur.execute("SELECT COUNT(*) c FROM routing_groups")
        if cur.fetchone()["c"] == 0:
            for g in ["GENERAL", "CODE", "IMAGE", "VIDEO"]:
                cur.execute("INSERT INTO routing_groups(name, enabled, tag_filter) VALUES(?,?,?)", (g, 1, ""))
            self.conn.commit()
        cur.execute("SELECT COUNT(*) c FROM classifier_config")
        if cur.fetchone()["c"] == 0:
            cur.execute(
                "INSERT INTO classifier_config(id, classifier_provider, classifier_model, classifier_mode, low_confidence_threshold) VALUES(1,'','','rules_only',0.7)"
            )
            self.conn.commit()

    # ----- generic access API
    def list_providers(self):
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            rows = cur.execute("SELECT * FROM providers ORDER BY created_at DESC").fetchall()
            out = []
            for r in rows:
                out.append({
                    "id": r["id"], "name": r["name"], "base_url": r["base_url"],
                    "api_key": "***" + (decrypt_text(r["api_key_enc"])[-4:] if r["api_key_enc"] else ""),
                    "headers": json.loads(r["headers_json"] or "{}"),
                    "enabled": bool(r["enabled"]),
                    "tags": json.loads(r["tags_json"] or "{}"),
                    "last_health": json.loads(r["last_health_json"] or "{}"),
                })
            return out
        s = self.read_state()
        out = []
        for p in s["providers"]:
            q = dict(p)
            q["api_key"] = "***" + (decrypt_text(p.get("api_key_enc", ""))[-4:] if p.get("api_key_enc") else "")
            out.append(q)
        return out

    def get_provider_raw(self, provider_id: str):
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            r = cur.execute("SELECT * FROM providers WHERE id=?", (provider_id,)).fetchone()
            if not r:
                return None
            return {
                "id": r["id"], "name": r["name"], "base_url": r["base_url"],
                "api_key": decrypt_text(r["api_key_enc"] or ""),
                "headers": json.loads(r["headers_json"] or "{}"), "enabled": bool(r["enabled"]),
                "tags": json.loads(r["tags_json"] or "{}"), "last_health": json.loads(r["last_health_json"] or "{}"),
            }
        s = self.read_state()
        for p in s["providers"]:
            if p["id"] == provider_id:
                q = dict(p)
                q["api_key"] = decrypt_text(q.get("api_key_enc", ""))
                return q
        return None

    def upsert_provider(self, body: Dict[str, Any], provider_id: Optional[str] = None):
        pid = provider_id or body.get("id") or str(uuid.uuid4())[:8]
        now = dt.datetime.utcnow().isoformat()
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            ex = cur.execute("SELECT id, api_key_enc FROM providers WHERE id=?", (pid,)).fetchone()
            api_key_enc = ex["api_key_enc"] if ex else ""
            if body.get("api_key"):
                api_key_enc = encrypt_text(body["api_key"])
            cur.execute(
                """INSERT OR REPLACE INTO providers(id,name,base_url,api_key_enc,headers_json,enabled,tags_json,last_health_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,COALESCE((SELECT created_at FROM providers WHERE id=?),?),?)""",
                (
                    pid,
                    body.get("name", pid),
                    str(body.get("base_url", "")).rstrip("/"),
                    api_key_enc,
                    json.dumps(body.get("headers", {}), ensure_ascii=False),
                    1 if body.get("enabled", True) else 0,
                    json.dumps(body.get("tags", {}), ensure_ascii=False),
                    json.dumps(body.get("last_health", {}), ensure_ascii=False),
                    pid,
                    now,
                    now,
                ),
            )
            self.conn.commit()
            return self.get_provider_raw(pid)

        s = self.read_state()
        cur = None
        for p in s["providers"]:
            if p["id"] == pid:
                cur = p
                break
        if not cur:
            cur = {"id": pid, "created_at": now}
            s["providers"].append(cur)
        cur["name"] = body.get("name", pid)
        cur["base_url"] = str(body.get("base_url", "")).rstrip("/")
        cur["enabled"] = bool(body.get("enabled", True))
        cur["headers"] = body.get("headers", {})
        cur["tags"] = body.get("tags", {})
        cur["last_health"] = body.get("last_health", {})
        if body.get("api_key"):
            cur["api_key_enc"] = encrypt_text(body["api_key"])
        cur["updated_at"] = now
        self.write_state(s)
        return self.get_provider_raw(pid)

    def delete_provider(self, provider_id: str):
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            cur.execute("DELETE FROM providers WHERE id=?", (provider_id,))
            cur.execute("DELETE FROM models WHERE provider_id=?", (provider_id,))
            cur.execute("DELETE FROM deployments WHERE provider_id=?", (provider_id,))
            self.conn.commit()
            return True
        s = self.read_state()
        s["providers"] = [p for p in s["providers"] if p["id"] != provider_id]
        s["models"] = [m for m in s["models"] if m["provider_id"] != provider_id]
        s["deployments"] = [d for d in s["deployments"] if d["provider_id"] != provider_id]
        self.write_state(s)
        return True

    def set_models(self, provider_id: str, models: List[Dict[str, Any]]):
        now = dt.datetime.utcnow().isoformat()
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            cur.execute("DELETE FROM models WHERE provider_id=?", (provider_id,))
            for m in models:
                cur.execute(
                    "INSERT INTO models(id,provider_id,model_name,model_type,context_length,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (str(uuid.uuid4())[:12], provider_id, m.get("model_name"), m.get("model_type", "chat"), m.get("context_length"), m.get("note", ""), now, now),
                )
            self.conn.commit()
            return
        s = self.read_state()
        s["models"] = [m for m in s["models"] if m["provider_id"] != provider_id]
        for m in models:
            s["models"].append({"id": str(uuid.uuid4())[:12], "provider_id": provider_id, **m, "created_at": now, "updated_at": now})
        self.write_state(s)

    def list_models(self, provider_id: str):
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            rows = cur.execute("SELECT * FROM models WHERE provider_id=? ORDER BY model_name", (provider_id,)).fetchall()
            return [dict(r) for r in rows]
        s = self.read_state()
        return [m for m in s["models"] if m["provider_id"] == provider_id]

    def list_groups(self):
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            rows = cur.execute("SELECT * FROM routing_groups ORDER BY name").fetchall()
            return [dict(r) for r in rows]
        return self.read_state()["routing_groups"]

    def upsert_group(self, group: Dict[str, Any]):
        name = group["name"]
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO routing_groups(name,enabled,tag_filter) VALUES(?,?,?)", (name, 1 if group.get("enabled", True) else 0, group.get("tag_filter", "")))
            self.conn.commit()
            return
        s = self.read_state()
        s["routing_groups"] = [g for g in s["routing_groups"] if g["name"] != name] + [group]
        self.write_state(s)

    def delete_group(self, name: str):
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            cur.execute("DELETE FROM routing_groups WHERE name=?", (name,))
            cur.execute("DELETE FROM deployments WHERE group_name=?", (name,))
            self.conn.commit()
            return
        s = self.read_state()
        s["routing_groups"] = [g for g in s["routing_groups"] if g["name"] != name]
        s["deployments"] = [d for d in s["deployments"] if d["group_name"] != name]
        self.write_state(s)

    def list_deployments(self, group_name: Optional[str] = None):
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            if group_name:
                rows = cur.execute("SELECT * FROM deployments WHERE group_name=? ORDER BY priority ASC, weight DESC", (group_name,)).fetchall()
            else:
                rows = cur.execute("SELECT * FROM deployments ORDER BY group_name, priority ASC, weight DESC").fetchall()
            return [dict(r) for r in rows]
        s = self.read_state()
        arr = s["deployments"]
        if group_name:
            arr = [d for d in arr if d["group_name"] == group_name]
        return arr

    def upsert_deployment(self, dep: Dict[str, Any], dep_id: Optional[str] = None):
        did = dep_id or dep.get("id") or str(uuid.uuid4())[:10]
        now = dt.datetime.utcnow().isoformat()
        if self.mode == "sqlite":
            cur = self.conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO deployments(id,group_name,provider_id,model_name,weight,priority,fallback_on,enabled,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,COALESCE((SELECT created_at FROM deployments WHERE id=?),?),?)""",
                (
                    did,
                    dep["group_name"],
                    dep["provider_id"],
                    dep["model_name"],
                    int(dep.get("weight", 100)),
                    int(dep.get("priority", 100)),
                    ",".join(dep.get("fallback_on", ["timeout", "429", "5xx"])),
                    1 if dep.get("enabled", True) else 0,
                    did,
                    now,
                    now,
                ),
            )
            self.conn.commit()
            return did
        s = self.read_state()
        s["deployments"] = [d for d in s["deployments"] if d["id"] != did] + [{"id": did, **dep, "updated_at": now}]
        self.write_state(s)
        return did

    def delete_deployment(self, dep_id: str):
        if self.mode == "sqlite":
            self.conn.execute("DELETE FROM deployments WHERE id=?", (dep_id,))
            self.conn.commit()
            return
        s = self.read_state()
        s["deployments"] = [d for d in s["deployments"] if d["id"] != dep_id]
        self.write_state(s)

    def get_classifier(self):
        if self.mode == "sqlite":
            r = self.conn.execute("SELECT * FROM classifier_config WHERE id=1").fetchone()
            return dict(r)
        return self.read_state()["classifier"]

    def set_classifier(self, cfg: Dict[str, Any]):
        if self.mode == "sqlite":
            self.conn.execute(
                "UPDATE classifier_config SET classifier_provider=?, classifier_model=?, classifier_mode=?, low_confidence_threshold=? WHERE id=1",
                (cfg.get("classifier_provider", ""), cfg.get("classifier_model", ""), cfg.get("classifier_mode", "rules_only"), float(cfg.get("low_confidence_threshold", 0.7))),
            )
            self.conn.commit()
            return
        s = self.read_state()
        s["classifier"] = cfg
        self.write_state(s)

    def log_route(self, row: Dict[str, Any]):
        now = dt.datetime.utcnow().isoformat()
        if self.mode == "sqlite":
            self.conn.execute(
                "INSERT INTO route_logs(id,request_id,group_name,provider_id,model_name,latency_ms,status_code,fallback_count,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4())[:12], row["request_id"], row["group"], row.get("provider_id", ""), row.get("model_name", ""), float(row.get("latency_ms", 0)), int(row.get("status_code", 0)), int(row.get("fallback_count", 0)), now),
            )
            self.conn.commit()
            return
        s = self.read_state()
        s["route_logs"].append({**row, "created_at": now})
        s["route_logs"] = s["route_logs"][-2000:]
        self.write_state(s)

    def rpm_allow(self, key: str) -> bool:
        now = time.time()
        q = self._rpm.get(key, [])
        q = [x for x in q if x >= now - 60]
        if len(q) >= RPM_LIMIT:
            self._rpm[key] = q
            return False
        q.append(now)
        self._rpm[key] = q
        return True


store = Storage()

# -------- classifier runtime --------
async def classifier_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = store.get_classifier()
    groups = [g["name"] for g in store.list_groups() if g.get("enabled", 1)]
    rule = detect_intent_rule(payload, groups)
    mode = cfg.get("classifier_mode", "rules_only")
    if mode == "rules_only":
        return {**rule, "final": rule["group"], "reason": "rules_only"}

    def need_llm():
        if mode == "classifier_only":
            return True
        return rule.get("confidence", 0) < float(cfg.get("low_confidence_threshold", 0.7))

    if not need_llm():
        return {**rule, "final": rule["group"], "reason": "hybrid-rule-hit"}

    provider = store.get_provider_raw(cfg.get("classifier_provider", ""))
    if not provider:
        return {**rule, "final": rule["group"], "reason": "classifier-provider-missing"}

    headers = {"Content-Type": "application/json", **(provider.get("headers") or {})}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    prompt = (
        "你是路由分类器。仅输出 JSON，如 {\"route\":\"GENERAL\",\"confidence\":0.9,\"reason\":\"...\"}。"
        "route 只能是以下之一: " + ",".join(groups) + "。\n"
        "用户消息：" + flatten_text(payload.get("messages", []))
    )
    body = {
        "model": cfg.get("classifier_model", ""),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            r = await client.post(f"{provider['base_url']}/v1/chat/completions", headers=headers, json=body)
            if r.status_code >= 400:
                return {**rule, "final": rule["group"], "reason": f"classifier-http-{r.status_code}"}
            content = (((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                obj = json.loads(m.group(0))
                route = str(obj.get("route", "")).upper().strip()
                if route in groups:
                    return {
                        "group": route,
                        "source": "classifier",
                        "confidence": float(obj.get("confidence", 0.7)),
                        "reason": obj.get("reason", "classifier"),
                        "final": route,
                    }
    except Exception:
        pass
    return {**rule, "final": rule["group"], "reason": "classifier-fallback"}


# ---------- config generation ----------
def generate_configs() -> Dict[str, Any]:
    groups = store.list_groups()
    deps = store.list_deployments()
    providers = {p["id"]: store.get_provider_raw(p["id"]) for p in store.list_providers()}

    model_list = []
    for g in groups:
        gname = g["name"]
        gd = [d for d in deps if d["group_name"] == gname and d.get("enabled", 1)]
        gl = []
        for d in gd:
            p = providers.get(d["provider_id"])
            if not p:
                continue
            gl.append(
                {
                    "model_name": gname,
                    "litellm_params": {
                        "model": f"openai/{d['model_name']}",
                        "api_base": p["base_url"],
                        "api_key": p["api_key"] or "",
                        "timeout": REQUEST_TIMEOUT_SECONDS,
                    },
                }
            )
        model_list.extend(gl)

    litellm_yaml = yaml.safe_dump({"model_list": model_list, "router_settings": {"routing_strategy": "simple-shuffle"}}, sort_keys=False, allow_unicode=True)

    rules = {"groups": groups, "keywords": {"CODE": CODE_PATTERNS, "IMAGE": IMAGE_PATTERNS, "VIDEO": VIDEO_PATTERNS}}
    settings = {
        "classifier": store.get_classifier(),
        "bypass_litellm": BYPASS_LITELLM,
        "upstream_base_url": UPSTREAM_BASE_URL,
        "raw_upstream_base_url": RAW_UPSTREAM_BASE_URL,
        "rpm_limit": RPM_LIMIT,
    }

    return {
        "litellm/config.yaml": litellm_yaml,
        "router/rules.json": json.dumps(rules, ensure_ascii=False, indent=2),
        "router/settings.json": json.dumps(settings, ensure_ascii=False, indent=2),
    }


# ---------------- app ----------------
app = FastAPI(title="sparouter-control-plane")


@app.get("/", response_class=HTMLResponse)
async def root_page():
    return """<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/><title>AI Router API</title></head><body><h2>AI Router API is running</h2><p>WebUI: deploy router-nextjs to access full console.</p><ul><li><a href='/healthz'>/healthz</a></li><li><a href='/status.json'>/status.json</a></li></ul></body></html>"""


@app.get("/status.json")
async def status_json():
    return {"status": "ok", "storage_mode": store.mode}


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "storage_mode": store.mode}


# ---- Providers API
@app.get("/api/providers")
async def api_list_providers():
    return {"providers": store.list_providers()}


@app.post("/api/providers")
async def api_create_provider(req: Request):
    body = await req.json()
    if not body.get("base_url"):
        raise HTTPException(status_code=400, detail="base_url is required")
    p = store.upsert_provider(body)
    return {"provider": {k: v for k, v in p.items() if k != "api_key"}}


@app.put("/api/providers/{provider_id}")
async def api_update_provider(provider_id: str, req: Request):
    body = await req.json()
    p = store.upsert_provider(body, provider_id=provider_id)
    return {"provider": {k: v for k, v in p.items() if k != "api_key"}}


@app.delete("/api/providers/{provider_id}")
async def api_delete_provider(provider_id: str):
    store.delete_provider(provider_id)
    return {"deleted": provider_id}


@app.post("/api/providers/{provider_id}/sync-models")
async def api_sync_models(provider_id: str, req: Request):
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    p = store.get_provider_raw(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail="provider not found")

    manual = body.get("manual_models") or []
    models = []
    if manual:
        for m in manual:
            models.append({"model_name": m.get("model_name") if isinstance(m, dict) else str(m), "model_type": (m.get("model_type") if isinstance(m, dict) else "chat") or "chat", "context_length": (m.get("context_length") if isinstance(m, dict) else None), "note": (m.get("note") if isinstance(m, dict) else "manual")})
    else:
        headers = {"Content-Type": "application/json", **(p.get("headers") or {})}
        if p.get("api_key"):
            headers["Authorization"] = f"Bearer {p['api_key']}"
        urls = [f"{p['base_url']}/v1/models", f"{p['base_url']}/models"]
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            for u in urls:
                try:
                    r = await client.get(u, headers=headers)
                    if r.status_code < 400:
                        data = r.json()
                        for item in data.get("data", []):
                            models.append({"model_name": item.get("id"), "model_type": "chat", "context_length": None, "note": "synced"})
                        break
                except Exception:
                    pass

    store.set_models(provider_id, [m for m in models if m.get("model_name")])
    return {"provider_id": provider_id, "models": store.list_models(provider_id)}


@app.get("/api/providers/{provider_id}/models")
async def api_provider_models(provider_id: str):
    return {"models": store.list_models(provider_id)}


# ---- health check provider
@app.post("/api/providers/{provider_id}/health-check")
async def provider_health(provider_id: str):
    p = store.get_provider_raw(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail="provider not found")
    headers = {"Content-Type": "application/json", **(p.get("headers") or {})}
    if p.get("api_key"):
        headers["Authorization"] = f"Bearer {p['api_key']}"
    start = time.perf_counter()
    status = "down"
    err = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            r = await client.get(f"{p['base_url']}/v1/models", headers=headers)
            if r.status_code < 500:
                status = "up"
            else:
                err = f"http-{r.status_code}"
    except Exception as e:
        err = str(e)
    latency = round((time.perf_counter() - start) * 1000, 2)
    health = {"status": status, "latency_ms": latency, "error": err, "checked_at": dt.datetime.utcnow().isoformat()}
    p2 = store.upsert_provider({**p, "api_key": p.get("api_key", ""), "last_health": health}, provider_id)
    return {"health": p2.get("last_health", health)}


# ---- Routing API
@app.get("/api/routing/groups")
async def api_groups():
    groups = store.list_groups()
    for g in groups:
        g["deployments"] = store.list_deployments(g["name"])
    return {"groups": groups}


@app.post("/api/routing/groups")
async def api_create_group(req: Request):
    body = await req.json()
    if not body.get("name"):
        raise HTTPException(status_code=400, detail="name required")
    store.upsert_group({"name": body["name"], "enabled": body.get("enabled", True), "tag_filter": body.get("tag_filter", "")})
    return {"ok": True}


@app.put("/api/routing/groups/{group}")
async def api_update_group(group: str, req: Request):
    body = await req.json()
    store.upsert_group({"name": group, "enabled": body.get("enabled", True), "tag_filter": body.get("tag_filter", "")})
    return {"ok": True}


@app.delete("/api/routing/groups/{group}")
async def api_delete_group(group: str):
    store.delete_group(group)
    return {"deleted": group}


@app.post("/api/routing/groups/{group}/deployments")
async def api_create_deployment(group: str, req: Request):
    body = await req.json()
    did = store.upsert_deployment({"group_name": group, **body})
    return {"id": did}


@app.put("/api/routing/deployments/{dep_id}")
async def api_update_deployment(dep_id: str, req: Request):
    body = await req.json()
    store.upsert_deployment(body, dep_id=dep_id)
    return {"ok": True}


@app.delete("/api/routing/deployments/{dep_id}")
async def api_delete_deployment(dep_id: str):
    store.delete_deployment(dep_id)
    return {"deleted": dep_id}


# ---- Classifier API
@app.get("/api/classifier/config")
async def api_get_classifier():
    return store.get_classifier()


@app.put("/api/classifier/config")
async def api_put_classifier(req: Request):
    body = await req.json()
    store.set_classifier(
        {
            "classifier_provider": body.get("classifier_provider", ""),
            "classifier_model": body.get("classifier_model", ""),
            "classifier_mode": body.get("classifier_mode", "rules_only"),
            "low_confidence_threshold": float(body.get("low_confidence_threshold", 0.7)),
        }
    )
    return store.get_classifier()


@app.post("/api/classifier/test")
async def api_classifier_test(req: Request):
    body = await req.json()
    result = await classifier_predict(body)
    group = result["final"]
    deps = store.list_deployments(group)
    deps = [d for d in deps if d.get("enabled", 1)]
    return {"classification": result, "candidate_deployments": deps}


# ---- Generate / Apply
@app.post("/api/generate")
async def api_generate():
    return {"files": generate_configs()}


@app.post("/api/apply")
async def api_apply(req: Request):
    token = req.headers.get("X-Admin-Token", "")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    files = generate_configs()
    Path("generated/litellm").mkdir(parents=True, exist_ok=True)
    Path("generated/router").mkdir(parents=True, exist_ok=True)
    Path("generated/litellm/config.yaml").write_text(files["litellm/config.yaml"])
    Path("generated/router/rules.json").write_text(files["router/rules.json"])
    Path("generated/router/settings.json").write_text(files["router/settings.json"])
    return {"applied": True, "paths": ["generated/litellm/config.yaml", "generated/router/rules.json", "generated/router/settings.json"]}


# ---- Runtime route endpoint
@app.post("/v1/chat/completions")
async def route_chat(req: Request):
    payload = await req.json()
    req_id = req.headers.get("X-Request-Id") or str(uuid.uuid4())[:12]
    client_id = req.headers.get("X-Client-Id") or (req.client.host if req.client else "anonymous")
    auth = req.headers.get("Authorization", "")
    rate_key = hashlib.sha256(f"{client_id}:{auth}".encode()).hexdigest()[:16]
    if not store.rpm_allow(rate_key):
        raise HTTPException(status_code=429, detail=f"RPM limit exceeded ({RPM_LIMIT})")

    decision = await classifier_predict(payload)
    group = decision["final"]
    deployments = [d for d in store.list_deployments(group) if d.get("enabled", 1)]
    if not deployments:
        raise HTTPException(status_code=503, detail=f"no deployment available for group {group}")

    # weighted candidate order
    weighted = []
    for d in sorted(deployments, key=lambda x: int(x.get("priority", 100))):
        weighted.extend([d] * max(1, int(d.get("weight", 100)) // 10))
    random.shuffle(weighted)
    candidate_order = []
    seen = set()
    for x in weighted:
        if x["id"] not in seen:
            candidate_order.append(x)
            seen.add(x["id"])

    is_stream = bool(payload.get("stream", False))
    fallback_count = 0
    last_error = None
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0)) as client:
        for dep in candidate_order:
            provider = store.get_provider_raw(dep["provider_id"])
            if not provider or not provider.get("enabled", True):
                fallback_count += 1
                continue
            headers = {"Content-Type": "application/json", **(provider.get("headers") or {})}
            if BYPASS_LITELLM:
                if provider.get("api_key"):
                    headers["Authorization"] = f"Bearer {provider['api_key']}"
                target = f"{provider['base_url']}/v1/chat/completions"
                out_payload = dict(payload)
                out_payload["model"] = dep["model_name"]
            else:
                target = f"{UPSTREAM_BASE_URL}/v1/chat/completions"
                if LITELLM_MASTER_KEY:
                    headers["Authorization"] = f"Bearer {LITELLM_MASTER_KEY}"
                out_payload = dict(payload)
                out_payload["model"] = group

            headers["X-Route-Group"] = group
            headers["X-Request-Id"] = req_id

            try:
                if is_stream:
                    upstream = await client.stream("POST", target, headers=headers, json=out_payload)
                    if upstream.status_code >= 400:
                        last_error = HTTPException(status_code=upstream.status_code, detail=(await upstream.aread()).decode(errors="ignore")[:500])
                        fallback_count += 1
                        await upstream.aclose()
                        continue

                    async def iter_bytes():
                        try:
                            async for chunk in upstream.aiter_bytes():
                                if chunk:
                                    yield chunk
                        finally:
                            await upstream.aclose()

                    latency = round((time.perf_counter() - start) * 1000, 2)
                    store.log_route({"request_id": req_id, "group": group, "provider_id": provider["id"], "model_name": dep["model_name"], "latency_ms": latency, "status_code": 200, "fallback_count": fallback_count})
                    return StreamingResponse(iter_bytes(), media_type="text/event-stream", headers={"X-Selected-Group": group, "X-Selected-Provider": provider["id"], "X-Selected-Model": dep["model_name"], "X-Fallback-Count": str(fallback_count), "X-Decision-Reason": decision.get("reason", "")})

                r = await client.post(target, headers=headers, json=out_payload)
                if r.status_code in {429, 500, 502, 503, 504}:
                    last_error = HTTPException(status_code=r.status_code, detail=r.text[:500])
                    fallback_count += 1
                    continue

                latency = round((time.perf_counter() - start) * 1000, 2)
                store.log_route({"request_id": req_id, "group": group, "provider_id": provider["id"], "model_name": dep["model_name"], "latency_ms": latency, "status_code": r.status_code, "fallback_count": fallback_count})
                try:
                    content = r.json()
                except Exception:
                    content = {"raw": r.text}
                return JSONResponse(status_code=r.status_code, content=content, headers={"X-Selected-Group": group, "X-Selected-Provider": provider["id"], "X-Selected-Model": dep["model_name"], "X-Fallback-Count": str(fallback_count), "X-Decision-Reason": decision.get("reason", "")})
            except Exception as e:
                last_error = HTTPException(status_code=502, detail=f"upstream error: {e}")
                fallback_count += 1
                continue

    raise last_error or HTTPException(status_code=503, detail="all candidates failed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=ROUTER_LISTEN_PORT)

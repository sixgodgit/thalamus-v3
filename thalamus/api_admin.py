"""管理面：登录、key 分发与每日额度、渠道增删改与探测、请求日志、路由试算、热重载。"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import os
import re
import secrets
import time
from collections import deque
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import __version__, totp
from .adapters import CallContext, make_adapter
from .api_v1 import client_ip as _ip  # 同一套"可信来源 IP"口径，见那边的说明
from .config import DIALECTS, MANAGED_DIR, ConfigError, load_settings
from .engine import Engine
from .quota import seconds_to_reset
from .secrets import mask
from .types import GatewayError, UpstreamError

log = logging.getLogger("thalamus.admin")

STATIC = Path(__file__).resolve().parent.parent / "static"
COOKIE = "thl_admin"
SESSION_TTL = 86400
LOGIN_WINDOW, LOGIN_MAX_FAILS = 600, 8


def _allowed_networks() -> list[ipaddress._BaseNetwork]:
    """THALAMUS_ADMIN_ALLOW：逗号分隔的 IP 或 CIDR。留空 = 不限（靠反代白名单）。"""
    out = []
    for part in os.getenv("THALAMUS_ADMIN_ALLOW", "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            log.warning("THALAMUS_ADMIN_ALLOW 里的 %r 不是合法 IP/CIDR，已忽略", part)
    return out


def admin_network_guard(request: Request) -> None:
    """整个 /admin（含 HTML 页面与登录）先过来源 IP 白名单。

    纵深防御：反代的白名单可能忘配（审计发现过控制台公网可达），这一层不依赖反代。
    不合规返回 404 而不是 403 —— 不告诉扫描者这里有东西。
    """
    nets = _allowed_networks()
    if not nets:
        return
    try:
        ip = ipaddress.ip_address(_ip(request))
    except ValueError:
        raise GatewayError(404, "not_found", "not found") from None
    if not any(ip in n for n in nets):
        log.warning("admin 访问被来源白名单拒绝：%s %s", ip, request.url.path)
        raise GatewayError(404, "not_found", "not found")


router = APIRouter(dependencies=[Depends(admin_network_guard)])


def require_admin(request: Request) -> None:
    token = request.app.state.settings.admin_token
    if not token:
        raise GatewayError(503, "admin_disabled", "THALAMUS_ADMIN_TOKEN is not set; admin console disabled")
    sessions: dict[str, float] = request.app.state.admin_sessions
    c = request.cookies.get(COOKIE)
    if c and sessions.get(c, 0) > time.time():
        return
    st = request.app.state.settings
    if st.totp_secret and st.totp_only:
        # only 模式：口令（含 Bearer 头）不再是一种凭据，只能靠 TOTP 换到的会话 cookie
        raise GatewayError(401, "unauthorized", "admin login required")
    auth = request.headers.get("authorization", "")
    given = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-admin-token", "")
    if given and hmac.compare_digest(given, token):
        return
    raise GatewayError(401, "unauthorized", "admin login required")


async def _json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise GatewayError(400, "invalid_request_error", "JSON body required") from None
    if not isinstance(body, dict):
        raise GatewayError(400, "invalid_request_error", "JSON object required")
    return body


def _engine(request: Request) -> Engine:
    return request.app.state.engine


# ───────────────────────────── 页面与登录 ─────────────────────────────


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def page(request: Request):
    # 登录卡渲染哪种输入框由运行期配置决定：静态 HTML 里的 JS 常量做不到这件事，
    # 所以用占位符在服务端替换（TOTP 是否启用、是否 only 模式）。
    html = (STATIC / "admin.html").read_text(encoding="utf-8")
    st = request.app.state.settings
    enabled = bool(st.totp_secret)
    only = enabled and bool(st.totp_only)
    html = html.replace("__TOTP_ENABLED__", "true" if enabled else "false")
    html = html.replace("__TOTP_ONLY__", "true" if only else "false")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@router.post("/api/login")
async def login(request: Request):
    token = request.app.state.settings.admin_token
    if not token:
        raise GatewayError(503, "admin_disabled", "THALAMUS_ADMIN_TOKEN is not set")
    ip = _ip(request)
    fails: dict[str, deque[float]] = request.app.state.login_failures
    q = fails.setdefault(ip, deque())
    now = time.time()
    while q and q[0] < now - LOGIN_WINDOW:
        q.popleft()
    if len(q) >= LOGIN_MAX_FAILS:
        raise GatewayError(429, "too_many_attempts", "too many failed logins, try later", {"Retry-After": str(LOGIN_WINDOW)})
    body = await _json(request)
    st = request.app.state.settings
    totp_ok = False
    if st.totp_secret and str(body.get("totp", "")).strip():
        # 防重放：同一个验证码在它自己的时间窗内只接受一次（被截获的 6 位数字不能再用）
        last = getattr(request.app.state, "totp_last_counter", 0)
        hit = totp.counter_for(st.totp_secret, str(body["totp"]), min_counter=last)
        if hit is not None:
            request.app.state.totp_last_counter = hit
            totp_ok = True
    # only 模式下口令彻底失效（含 Bearer 头，见 require_admin）
    token_ok = (not (st.totp_secret and st.totp_only)) and hmac.compare_digest(
        str(body.get("token", "")), token)
    if not (totp_ok or token_ok):
        q.append(now)
        raise GatewayError(401, "unauthorized",
                           "wrong TOTP code" if st.totp_secret else "wrong admin token")
    q.clear()
    sid = secrets.token_urlsafe(32)
    sessions: dict[str, float] = request.app.state.admin_sessions
    for k in [k for k, exp in sessions.items() if exp < now]:
        sessions.pop(k, None)
    sessions[sid] = now + SESSION_TTL
    resp = JSONResponse({"ok": True})
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    resp.set_cookie(COOKIE, sid, max_age=SESSION_TTL, httponly=True, samesite="strict", secure=secure, path="/admin")
    return resp


@router.post("/api/logout")
async def logout(request: Request):
    request.app.state.admin_sessions.pop(request.cookies.get(COOKIE, ""), None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/admin")
    return resp


@router.get("/api/me")
async def me(request: Request):
    require_admin(request)
    st = request.app.state.settings
    enabled = bool(st.totp_secret)
    return {"ok": True, "version": __version__,
            "totp": {"enabled": enabled, "only": enabled and bool(st.totp_only), "period": totp.PERIOD}}


# ───────────────────────────── 概览 ─────────────────────────────


def _day_start_ts(engine: Engine) -> float:
    return time.time() - (86400 - seconds_to_reset(engine.settings.policy.timezone))


@router.get("/api/overview")
async def overview(request: Request):
    require_admin(request)
    e = _engine(request)
    s = e.settings
    await e.writer.flush()
    stats = await e.store.stats_since(_day_start_ts(e))
    outcomes = await e.store.outcome_stats(_day_start_ts(e))
    keys = await e.store.list_keys()
    usage = await e.ledger.usage_for_day(e.ledger.today())
    return {
        "version": __version__,
        "day": e.ledger.today(),
        "timezone": s.policy.timezone,
        "reset_in_s": seconds_to_reset(s.policy.timezone),
        "today": stats,
        "outcomes": outcomes,
        "metered_usd_today": round(await e.metered_spent(), 4),
        "metered_cap_usd": s.policy.daily_metered_usd_cap,
        "keys_total": len(keys),
        "keys_enabled": sum(1 for k in keys if k.enabled),
        "keys_active_today": len(usage),
        "sessions": len(e.sessions),
        "inflight": sum(e.channel_load.values()),
        "capacity": e.capacity.snapshot(),
        "warnings": s.warnings,
        "uptime_s": int(time.time() - e.started_at),
    }


# ───────────────────────────── Key ─────────────────────────────

_PLANS = ("standard", "pro")
_MODES = ("economy", "reject")


def _validate_key_fields(e: Engine, body: dict[str, Any], creating: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "name" in body or creating:
        name = str(body.get("name", "")).strip()
        if not name or len(name) > 80:
            raise GatewayError(400, "invalid_request_error", "name 必填，最多 80 字")
        out["name"] = name
    for f_, typ in (("daily_credits", float), ("rpm", int), ("max_concurrency", int), ("max_output_tokens", int)):
        if body.get(f_) is not None:
            try:
                v = typ(body[f_])
            except (TypeError, ValueError):
                raise GatewayError(400, "invalid_request_error", f"{f_} 必须是数字") from None
            if v < 0:
                raise GatewayError(400, "invalid_request_error", f"{f_} 不能为负")
            if f_ in ("rpm", "max_concurrency") and v < 1:
                raise GatewayError(400, "invalid_request_error", f"{f_} 至少为 1（不提供不限速选项）")
            out[f_] = v
    if body.get("plan") is not None:
        if body["plan"] not in _PLANS:
            raise GatewayError(400, "invalid_request_error", "plan 只能是 standard/pro")
        out["plan"] = body["plan"]
    if body.get("exhaust_mode") is not None:
        if body["exhaust_mode"] not in _MODES:
            raise GatewayError(400, "invalid_request_error", "exhaust_mode 只能是 economy/reject")
        out["exhaust_mode"] = body["exhaust_mode"]
    if "allowed_models" in body:
        am = body["allowed_models"] or None
        if am is not None:
            if not isinstance(am, list) or any(m not in e.settings.models for m in am):
                raise GatewayError(400, "invalid_request_error", "allowed_models 含未知模型")
        out["allowed_models"] = am
    if "allowed_regions" in body:
        ar = body["allowed_regions"] or None
        if ar is not None:
            known = e.settings.regions()
            if not isinstance(ar, list) or any(r not in known for r in ar):
                raise GatewayError(400, "invalid_request_error", f"allowed_regions 只能取 {sorted(known)}")
            live = {o.channel.region for o in e.settings.usable_offers()}
            if not set(ar) & live:
                raise GatewayError(
                    400, "invalid_request_error", f"区域 {sorted(set(ar))} 当前没有可用渠道，这样的 key 一次请求都发不出去"
                )
            ar = sorted(set(ar))
        out["allowed_regions"] = ar
    if "enabled" in body and body["enabled"] is not None:
        out["enabled"] = bool(body["enabled"])
    if "note" in body and body["note"] is not None:
        out["note"] = str(body["note"])[:500]
    if "expires_days" in body:
        d = body["expires_days"]
        out["expires_at"] = (time.time() + float(d) * 86400) if d not in (None, "", 0) else None
    return out


@router.get("/api/keys")
async def list_keys(request: Request):
    require_admin(request)
    e = _engine(request)
    usage = await e.ledger.usage_for_day(e.ledger.today())
    rows = []
    for k in await e.store.list_keys():
        u = usage.get(k.id, {})
        d = k.public()
        d.update(
            used_today=round(u.get("used", 0.0), 3),
            reserved_today=round(u.get("reserved", 0.0), 3),
            requests_today=u.get("requests", 0),
            inflight=e.limiter.inflight(k.id),
        )
        rows.append(d)
    return {"day": e.ledger.today(), "keys": rows}


@router.post("/api/keys", status_code=201)
async def create_key(request: Request):
    require_admin(request)
    e = _engine(request)
    body = await _json(request)
    fields = _validate_key_fields(e, body, creating=True)
    k = e.settings.policy.keys
    defaults = {
        "daily_credits": k.daily_credits,
        "rpm": k.rpm,
        "max_concurrency": k.max_concurrency,
        "max_output_tokens": k.max_output_tokens,
        "plan": k.plan,
        "exhaust_mode": k.exhaust_mode,
        "allowed_regions": k.allowed_regions,
    }
    name = fields.pop("name")
    for f_ in ("allowed_models", "allowed_regions"):
        if f_ in fields:  # 显式传 null 表示"不限"，要能压过策略默认值
            defaults[f_] = fields.pop(f_)
    key, raw = await e.store.create_key(name, defaults, **fields)
    return JSONResponse({"key": key.public(), "api_key": raw}, status_code=201)


@router.patch("/api/keys/{kid}")
async def update_key(kid: str, request: Request):
    require_admin(request)
    e = _engine(request)
    fields = _validate_key_fields(e, await _json(request), creating=False)
    key = await e.store.update_key(kid, fields)
    if key is None:
        raise GatewayError(404, "not_found", "key not found")
    e.invalidate_keys()
    return {"key": key.public()}


@router.post("/api/keys/{kid}/rotate")
async def rotate_key(kid: str, request: Request):
    require_admin(request)
    e = _engine(request)
    raw = await e.store.rotate_key(kid)
    if raw is None:
        raise GatewayError(404, "not_found", "key not found")
    e.invalidate_keys()
    return {"api_key": raw}


@router.post("/api/keys/{kid}/reset-today")
async def reset_today(kid: str, request: Request):
    require_admin(request)
    e = _engine(request)
    day = e.ledger.today()
    await e.ledger.usage(kid, day)  # 确保已加载，避免稍后加载时用库里的旧值覆盖
    e.ledger.reset_used(kid, day)
    await e.writer.flush()
    return {"ok": True}


@router.delete("/api/keys/{kid}")
async def delete_key(kid: str, request: Request):
    require_admin(request)
    e = _engine(request)
    await e.writer.flush()
    e.ledger.forget(kid)
    if not await e.store.delete_key(kid):
        raise GatewayError(404, "not_found", "key not found")
    e.invalidate_keys()
    return {"deleted": kid}


@router.get("/api/keys/{kid}/usage")
async def key_usage(kid: str, request: Request):
    require_admin(request)
    e = _engine(request)
    await e.writer.flush()
    return {
        "history": await e.store.usage_history(kid, 14),
        "requests": await e.store.recent_requests(50, kid),
    }


# ───────────────────────────── 渠道 / 日志 / 学习 ─────────────────────────────


@router.get("/api/channels")
async def channels(request: Request):
    require_admin(request)
    e = _engine(request)
    s = e.settings
    health = e.health.snapshot()
    cap = e.capacity.snapshot()
    spend = e.metered_by_channel()
    out = []
    for ch in s.channels.values():
        offers = [o for o in s.offers if o.channel.id == ch.id]
        out.append(
            {
                "id": ch.id,
                "kind": ch.kind,
                "region": ch.region,
                "base_url": ch.base_url,
                "configured": ch.configured,
                "enabled": ch.enabled,
                "key_env": ch.api_key_env,
                "managed": ch.id in s.managed,  # channels.d/ 里的才能在控制台改删
                "key_source": ("runtime" if request.app.state.secrets.get(ch.api_key_env)
                               else "env" if ch.api_key else "none"),
                "dialect": ch.dialect,
                "load": e.channel_load.get(ch.id, 0),
                "max_concurrency": ch.max_concurrency,
                "metered_usd_today": round(spend.get(ch.id, 0.0), 5),
                "budget_unit": ch.budget.unit if ch.budget else None,
                "capacity": {k: v for k, v in cap.items() if k == ch.id or k.endswith("@" + ch.id)},
                "offers": [
                    {
                        "key": o.key,
                        "model": o.model.key,
                        "display": o.model.display,
                        "upstream": o.upstream,
                        "dialect": o.dialect,
                        "tier": o.model.tier,
                        "context": o.model.context,
                        "max_output": o.model.max_output,
                        "price": {"input": o.price.input, "cached": o.price.cached, "output": o.price.output},
                        "time_multiplier_now": o.pricing_time.at(),
                        "health": health.get(o.key, {"state": "closed", "ok": 0, "failed": 0}),
                        "ttft_s": round(e.health.ttft(o), 3),
                        "tps": round(e.health.tps(o), 1),
                    }
                    for o in offers
                ],
            }
        )
    return {"channels": out, "pool": request.app.state.pool_status}


# ───────────────────────────── 渠道增删改 ─────────────────────────────
#
# 控制台加的渠道写成 config/channels.d/<id>.yaml，一个渠道一个文件：
# 不去改 upstreams.yaml（会冲掉里面的注释），也仍然是可手工编辑、可进版本库的 YAML。
# API key 存运行时密钥库，不需要改 .env 重启。

_CID = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
_MODEL_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,63}$")
_REGION = re.compile(r"^[a-z][a-z0-9-]{1,15}$")
_UPSTREAM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,95}$")


def _secret_name(cid: str) -> str:
    return f"THALAMUS_KEY_{cid.upper().replace('-', '_')}"


def _managed_path(e: Engine, cid: str) -> Path:
    if not _CID.match(cid):
        raise GatewayError(400, "invalid_request_error", "渠道 id 只能是小写字母开头的 2-32 位字母数字/下划线/连字符")
    return Path(e.settings.config_dir) / MANAGED_DIR / f"{cid}.yaml"


def _num(body: dict[str, Any], key: str, lo: float, hi: float, default: float | None = None) -> float:
    v = body.get(key, default)
    if v is None:
        raise GatewayError(400, "invalid_request_error", f"{key} 必填")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise GatewayError(400, "invalid_request_error", f"{key} 必须是数字") from None
    if not (lo <= f <= hi):
        raise GatewayError(400, "invalid_request_error", f"{key} 必须在 {lo:g}–{hi:g} 之间")
    return f


def _base_url(raw: Any) -> str:
    url = str(raw or "").strip().rstrip("/")
    if not re.match(r"^https?://[^\s/$.?#][^\s]{0,300}$", url):
        raise GatewayError(400, "invalid_request_error", "base_url 必须是 http(s):// 开头的完整地址")
    return url


async def _probe_models(e: Engine, base_url: str, dialect: str, api_key: str, user_agent: str) -> dict[str, Any]:
    """调上游的 /models，把能认出来的模型 id 与上下文长度带回来。"""
    headers = {"User-Agent": user_agent or "thalamus-gateway/3.0", "Accept": "application/json"}
    if dialect == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    client = e.pool.temporary(base_url)
    try:
        r = await client.get("/models", headers=headers, timeout=15.0)
    except Exception as ex:  # noqa: BLE001 - 探测失败要原样告诉用户，不是服务器错误
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
    finally:
        await client.aclose()
    if r.status_code >= 400:
        return {"ok": False, "status": r.status_code, "error": r.text[:400]}
    try:
        data = r.json()
    except ValueError:
        return {"ok": False, "status": r.status_code, "error": "返回的不是 JSON，可能 base_url 不对"}
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {"ok": False, "status": r.status_code, "error": "返回里没有 data 数组，可能不是 OpenAI 兼容端点"}
    out = []
    for it in items:
        if isinstance(it, str):
            out.append({"id": it})
            continue
        if not isinstance(it, dict):
            continue
        mid = it.get("id") or it.get("name") or it.get("model")
        if not mid:
            continue
        # 各家把上下文长度放在不同字段里，能认出来就预填，认不出来让用户自己填
        ctx = next(
            (it[k] for k in ("context_length", "context_window", "max_context_length", "max_model_len", "max_input_tokens")
             if isinstance(it.get(k), (int, float)) and it[k] > 0),
            None,
        )
        out.append({
            "id": str(mid)[:120],
            "display": str(it.get("display_name") or it.get("name") or mid)[:120],
            "context": int(ctx) if ctx else None,
            "owned_by": str(it.get("owned_by") or "")[:60] or None,
        })
    out.sort(key=lambda m: m["id"])
    return {"ok": True, "status": r.status_code, "count": len(out), "models": out}


@router.post("/api/channels/probe")
async def probe_channel(request: Request):
    """探测一个上游：连通性 + 鉴权 + 可用模型列表。不写任何配置。"""
    require_admin(request)
    e = _engine(request)
    body = await _json(request)
    dialect = body.get("dialect", "openai_chat")
    if dialect not in DIALECTS:
        raise GatewayError(400, "invalid_request_error", f"dialect 只能是 {sorted(DIALECTS)}")
    key = str(body.get("api_key") or "").strip()
    cid = str(body.get("channel_id") or "").strip()
    if not key and cid:  # 复探已有渠道时不必重新输入 key
        ch = e.settings.channels.get(cid)
        key = ch.api_key if ch else ""
    if not key:
        raise GatewayError(400, "invalid_request_error", "需要 api_key（或给一个已配置好 key 的 channel_id）")
    return await _probe_models(e, _base_url(body.get("base_url")), dialect, key, str(body.get("user_agent") or ""))


def _build_channel_doc(e: Engine, body: dict[str, Any], cid: str) -> tuple[dict[str, Any], list[str]]:
    """把表单变成 channels.d/<id>.yaml 的内容。"""
    dialect = body.get("dialect", "openai_chat")
    if dialect not in DIALECTS:
        raise GatewayError(400, "invalid_request_error", f"dialect 只能是 {sorted(DIALECTS)}")
    if body.get("kind", "metered") != "metered":
        raise GatewayError(
            400, "invalid_request_error",
            "控制台只支持添加按量渠道；订阅渠道需要积分系数与窗口上限，请参照 upstreams.yaml 手写",
        )
    region = str(body.get("region") or "global").strip().lower()
    if not _REGION.match(region):
        raise GatewayError(400, "invalid_request_error", "region 只能是小写字母/数字/连字符")
    models = body.get("models")
    if not isinstance(models, list) or not models:
        raise GatewayError(400, "invalid_request_error", "至少要勾选一个模型")
    if len(models) > 50:
        raise GatewayError(400, "invalid_request_error", "一次最多添加 50 个模型")

    model_defs: dict[str, Any] = {}
    offers: dict[str, Any] = {}
    notes: list[str] = []
    for m in models:
        if not isinstance(m, dict):
            raise GatewayError(400, "invalid_request_error", "models 的每一项必须是对象")
        mkey = str(m.get("key") or m.get("id") or "").strip()
        if not _MODEL_KEY.match(mkey):
            raise GatewayError(400, "invalid_request_error", f"模型 key 不合法：{mkey[:40]}")
        upstream = str(m.get("upstream") or mkey).strip()
        if not _UPSTREAM.match(upstream):
            raise GatewayError(400, "invalid_request_error", f"上游模型名不合法：{upstream[:40]}")
        if mkey in offers:
            raise GatewayError(400, "invalid_request_error", f"模型 {mkey} 重复")
        price = m.get("price") or {}
        offer: dict[str, Any] = {
            "upstream": upstream,
            "price": {
                "input": _num(price, "input", 0, 10_000),
                "cached": _num(price, "cached", 0, 10_000, price.get("input", 0)),
                "output": _num(price, "output", 0, 10_000),
            },
        }
        if m.get("min_max_tokens"):
            offer["min_max_tokens"] = int(_num(m, "min_max_tokens", 0, 1_000_000))
        if m.get("think_tags"):
            offer["think_tags"] = True
        offers[mkey] = offer

        if mkey in e.settings.models:  # 已有的模型定义直接复用，只是多挂一个渠道
            notes.append(f"{mkey}：沿用已有的模型能力与质量先验")
            continue
        q = m.get("quality", 7.0)
        quality = {t: float(q) for t in ("utility", "chat", "code", "agent", "reason", "vision", "long")} \
            if isinstance(q, (int, float)) else {str(k): float(v) for k, v in dict(q).items()}
        reasoning = str(m.get("reasoning") or "none")
        if reasoning not in ("none", "hybrid", "always"):
            raise GatewayError(400, "invalid_request_error", "reasoning 只能是 none/hybrid/always")
        tier = str(m.get("tier") or "standard")
        if tier not in ("standard", "pro"):
            raise GatewayError(400, "invalid_request_error", "tier 只能是 standard/pro")
        model_defs[mkey] = {
            "display": str(m.get("display") or mkey)[:80],
            "context": int(_num(m, "context", 1024, 20_000_000)),
            "max_output": int(_num(m, "max_output", 256, 2_000_000)),
            "vision": bool(m.get("vision")),
            "tools": bool(m.get("tools", True)),
            "json_mode": bool(m.get("json_mode", True)),
            "reasoning": reasoning,
            "tier": tier,
            "ttft_s": _num(m, "ttft_s", 0.05, 120, 1.5),
            "tps": _num(m, "tps", 1, 10_000, 50),
            "quality": quality,
        }
        if not isinstance(q, (int, float)):
            notes.append(f"{mkey}：质量先验按你填的值")
        else:
            notes.append(f"{mkey}：质量先验统一取 {float(q):g}，建议之后用 scripts/calibrate.py 校准")

    doc: dict[str, Any] = {
        "channels": {
            cid: {
                "kind": "metered",
                "base_url": _base_url(body.get("base_url")),
                "api_key_env": _secret_name(cid),
                "dialect": dialect,
                "region": region,
                "max_concurrency": int(_num(body, "max_concurrency", 1, 1024, 32)),
                "connect_timeout_s": _num(body, "connect_timeout_s", 1, 60, 8),
                "enabled": bool(body.get("enabled", True)),
                "offers": offers,
            }
        }
    }
    if body.get("user_agent"):
        doc["channels"][cid]["user_agent"] = str(body["user_agent"])[:120]
    if model_defs:
        doc = {"models": model_defs, **doc}
    return doc, notes


async def _reload(request: Request) -> Any:
    """按当前 config_dir 重新加载并热应用；配置有错就抛 400，不改变现状。"""
    e = _engine(request)
    try:
        new = load_settings(e.settings.config_dir or None)
    except Exception as ex:  # noqa: BLE001 - ConfigError / YAML / IO 都按"配置有误"处理
        raise GatewayError(400, "config_error", f"配置有误，未生效：{ex}") from None
    new.db_path = e.settings.db_path
    new.admin_token = e.settings.admin_token
    await e.apply_settings(new)
    request.app.state.settings = new
    return new


@router.post("/api/channels", status_code=201)
async def create_channel(request: Request):
    require_admin(request)
    e = _engine(request)
    body = await _json(request)
    cid = str(body.get("id") or "").strip().lower()
    path = _managed_path(e, cid)
    if cid in e.settings.channels:
        raise GatewayError(409, "conflict", f"渠道 {cid} 已存在")
    api_key = str(body.get("api_key") or "").strip()
    if not api_key:
        raise GatewayError(400, "invalid_request_error", "api_key 必填")
    doc, notes = _build_channel_doc(e, body, cid)

    request.app.state.secrets.set(_secret_name(cid), api_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# 由 Thalamus 控制台创建，可手工编辑；改完在控制台点\"重载配置\"\n"
        + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    try:
        new = await _reload(request)
    except GatewayError:
        path.unlink(missing_ok=True)  # 回滚：坏配置不留在盘上
        request.app.state.secrets.delete(_secret_name(cid))
        await _reload(request)
        raise
    ch = new.channels[cid]
    return JSONResponse(
        {"id": cid, "configured": ch.configured, "offers": [o.key for o in new.offers if o.channel.id == cid],
         "notes": notes, "warnings": new.warnings, "file": str(path)},
        status_code=201,
    )


@router.patch("/api/channels/{cid}")
async def update_channel(cid: str, request: Request):
    """只允许改控制台管得着的部分：启用开关与 API key。其余请改 YAML 后重载。"""
    require_admin(request)
    e = _engine(request)
    body = await _json(request)
    if cid not in e.settings.channels:
        raise GatewayError(404, "not_found", f"渠道 {cid} 不存在")
    changed = []
    if body.get("api_key"):
        name = e.settings.channels[cid].api_key_env
        request.app.state.secrets.set(name, str(body["api_key"]).strip())
        changed.append("api_key")
    if "enabled" in body:
        if cid not in e.settings.managed:
            raise GatewayError(400, "not_managed", f"渠道 {cid} 来自 upstreams.yaml，请直接改文件里的 enabled")
        path = _managed_path(e, cid)
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        doc["channels"][cid]["enabled"] = bool(body["enabled"])
        path.write_text(
            "# 由 Thalamus 控制台管理\n" + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100),
            encoding="utf-8",
        )
        changed.append("enabled")
    new = await _reload(request)
    ch = new.channels[cid]
    return {"id": cid, "changed": changed, "enabled": ch.enabled, "configured": ch.configured}


@router.delete("/api/channels/{cid}")
async def delete_channel(cid: str, request: Request):
    require_admin(request)
    e = _engine(request)
    if cid not in e.settings.managed:
        raise GatewayError(
            400, "not_managed", f"渠道 {cid} 来自 upstreams.yaml，控制台不删主配置里的渠道"
        )
    path = _managed_path(e, cid)
    path.unlink(missing_ok=True)
    request.app.state.secrets.delete(_secret_name(cid))
    await _reload(request)
    return {"deleted": cid}


@router.post("/api/channels/{cid}/test")
async def test_channel(cid: str, request: Request):
    """向这个渠道真发一条最小请求：验证模型名、鉴权、usage 字段、错误文案。会产生真实费用。"""
    require_admin(request)
    e = _engine(request)
    body = await _json(request)
    offer = e.settings.offer(f"{body.get('model')}@{cid}")
    if offer is None:
        raise GatewayError(404, "not_found", f"{body.get('model')}@{cid} 不存在")
    if not offer.channel.api_key:
        raise GatewayError(400, "no_key", "这个渠道还没有 API key")
    prompt = str(body.get("prompt") or "回复两个字：收到")[:500]
    ctx = CallContext("admin-test", f"admtest-{secrets.token_hex(6)}", 64)
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 64}
    text, reasoning, usage, ttft, finish = "", "", None, None, None
    t0 = time.perf_counter()
    agen = make_adapter(offer, e.pool).stream(payload, ctx)
    try:
        async with asyncio.timeout(float(body.get("timeout_s") or 40)):
            async for d in agen:
                if ttft is None and d.meaningful:
                    ttft = time.perf_counter() - t0
                text += d.content
                reasoning += d.reasoning
                if d.usage:
                    usage = d.usage if usage is None else usage.merge(d.usage)
                if d.finish_reason:
                    finish = d.finish_reason  # 不在这里停：usage 常常在 finish 之后单独一帧
                if len(text) > 600:
                    break
    except UpstreamError as err:
        return {"ok": False, "failure": err.failure.value, "status": err.status,
                "retryable": err.retryable, "message": err.message[:500]}
    except (TimeoutError, asyncio.TimeoutError):
        return {"ok": False, "failure": "first_token_timeout", "message": "超时内没有收到任何输出"}
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "failure": "gateway_error", "message": f"{type(ex).__name__}: {str(ex)[:300]}"}
    finally:
        await agen.aclose()
    return {
        "ok": True,
        "upstream_model": offer.upstream,
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "finish_reason": finish,
        "text": text[:600],
        "had_reasoning": bool(reasoning),
        "usage": usage.__dict__ if usage else None,
        "usage_missing": usage is None or usage.empty,
    }


@router.get("/api/secrets")
async def list_secrets(request: Request):
    """只回显掩码，完整值永远不返回。"""
    require_admin(request)
    e = _engine(request)
    store = request.app.state.secrets
    masked = store.masked()
    return {
        "secrets": [
            {"name": ch.api_key_env, "channel": ch.id, "source":
                ("runtime" if masked.get(ch.api_key_env) else ("env" if ch.api_key else "none")),
             "value": masked.get(ch.api_key_env) or (mask(ch.api_key) if ch.api_key else "")}
            for ch in e.settings.channels.values()
        ]
    }


@router.get("/api/requests")
async def requests_log(request: Request, limit: int = 100, key_id: str | None = None):
    require_admin(request)
    e = _engine(request)
    await e.writer.flush()
    return {"requests": await e.store.recent_requests(min(max(limit, 1), 500), key_id)}


@router.get("/api/learning")
async def learning(request: Request):
    require_admin(request)
    e = _engine(request)
    return {"deltas": e.learner.snapshot(), "bad_rates": e.outcomes.snapshot()}


@router.get("/api/config")
async def config_summary(request: Request):
    require_admin(request)
    s = _engine(request).settings
    return {
        "models": [
            {"key": m.key, "display": m.display, "tier": m.tier, "vision": m.vision, "context": m.context, "quality": m.quality}
            for m in s.models.values()
        ],
        "key_defaults": s.policy.keys.__dict__,
        "regions": sorted(s.regions()),
        "live_regions": sorted({o.channel.region for o in s.usable_offers()}),
        "credit_usd": s.policy.credit_usd,
        "tasks": {k: v.__dict__ for k, v in s.policy.tasks.items()},
    }


@router.post("/api/route-preview")
async def route_preview(request: Request):
    require_admin(request)
    e = _engine(request)
    body = await _json(request)
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise GatewayError(400, "invalid_request_error", "messages required")
    key = await e.store.get_key(body["key_id"]) if body.get("key_id") else None
    return await e.preview(body, key)


@router.post("/api/reload")
async def reload_config(request: Request):
    require_admin(request)
    new = await _reload(request)
    return {"reloaded": True, "warnings": new.warnings, "offers": [o.key for o in new.usable_offers()]}

"""OpenAI 兼容数据面。"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ModelSpec
from .engine import Engine
from .intent import TASK_ALIASES
from .store import ApiKey
from .types import GatewayError, UpstreamError

router = APIRouter()


def raw_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return request.headers.get("x-api-key") or request.headers.get("api-key") or None


def client_ip(request: Request) -> str:
    """客户端 IP，用于按来源限流。

    只用 uvicorn 已按 THALAMUS_TRUSTED_PROXIES 校验过的 request.client：自己去读
    X-Forwarded-For 的话，客户端可以伪造最左边那一项（Caddy 只在链尾追加真实 IP），
    于是轮换假 IP 就能绕过鉴权失败限流和管理台登录限流。
    """
    return request.client.host if request.client else "?"


async def current_key(request: Request) -> ApiKey:
    engine: Engine = request.app.state.engine
    return await engine.authenticate(raw_key(request), client_ip(request))


async def read_json_body(request: Request, limit: int) -> dict[str, Any]:
    """带上限地读请求体：不设上限的话，一个超大 body 会被整体读进内存。"""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise GatewayError(413, "payload_too_large", f"request body exceeds {limit} bytes")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise GatewayError(413, "payload_too_large", f"request body exceeds {limit} bytes")
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise GatewayError(400, "invalid_request_error", "request body must be JSON") from None
    if not isinstance(body, dict):
        raise GatewayError(400, "invalid_request_error", "request body must be a JSON object")
    return body


@router.get("/models")
async def list_models(request: Request):
    key = await current_key(request)
    engine: Engine = request.app.state.engine
    regions = set(key.allowed_regions) if key.allowed_regions else None

    usable: list[ModelSpec] = []
    seen: set[str] = set()
    for o in engine.settings.eligible_offers():
        m = o.model
        if m.key in seen or (m.tier == "pro" and key.plan != "pro"):
            continue
        if key.allowed_models and m.key not in key.allowed_models:
            continue
        if regions is not None and o.channel.region not in regions:
            continue
        seen.add(m.key)
        usable.append(m)

    def caps(pool: list[ModelSpec], ctx: int | None = None, out: int | None = None) -> dict[str, Any]:
        """别名也要报出上下文与输出上限。

        不报的话，客户端只能用它自己的默认值（很多是 131072 = 128×1024），于是把输入
        提前截断在 131.1k —— 明明后面接的是 1M 的模型。同一个数字用两个常见字段名给出，
        因为各家客户端读的键不一样。
        """
        if ctx is None:
            ctx = max((m.context for m in pool), default=0)
        if out is None:
            out = max((m.max_output for m in pool), default=0)
        if key.max_output_tokens > 0:
            out = min(out, key.max_output_tokens)
        if not ctx:
            return {}
        return {"context_length": ctx, "context_window": ctx, "max_output_tokens": out}

    def alias(name: str, pool: list[ModelSpec]) -> dict[str, Any]:
        return {"id": name, "object": "model", "owned_by": "thalamus", **caps(pool)}

    vision = [m for m in usable if m.vision]
    data: list[dict[str, Any]] = [alias("thalamus/auto", usable), alias("thalamus/fast", usable)]
    if key.plan == "pro":
        data.append(alias("thalamus/pro", usable))
    data += [alias(f"thalamus/{t}", vision if t == "vision" else usable) for t in sorted(TASK_ALIASES)]
    data += [
        {"id": m.key, "object": "model", "owned_by": "thalamus", **caps([], m.context, m.max_output)}
        for m in usable
    ]
    return {"object": "list", "data": data}


@router.post("/chat/completions")
async def chat_completions(request: Request):
    key = await current_key(request)
    engine: Engine = request.app.state.engine
    body = await read_json_body(request, engine.settings.policy.max_request_bytes)
    stream = bool(body.get("stream"))
    session = (
        request.headers.get("x-session-id")
        or request.headers.get("x-opencode-session")
        or request.headers.get("x-conversation-id")
        or (body.get("session_id") if isinstance(body.get("session_id"), str) else None)
    )
    call = await engine.open(
        body, key, stream=stream, session_header=session, task_header=request.headers.get("x-thalamus-task")
    )
    headers = call.headers()

    if stream:
        so = body.get("stream_options") or {}
        include_usage = bool(so.get("include_usage")) if isinstance(so, dict) else False
        return StreamingResponse(
            call.sse(include_usage),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        payload = await call.collect()
    except UpstreamError as e:
        return JSONResponse(
            {"error": {"message": f"upstream interrupted ({e.failure.value})", "type": e.failure.value}},
            status_code=502,
            headers=headers,
        )
    return JSONResponse(payload, headers=headers)

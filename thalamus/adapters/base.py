"""适配器公共部分：连接池、SSE 切帧、错误分类、<think> 拆分。"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from ..config import Channel, Offer
from ..types import Delta, FailureClass, UpstreamError

log = logging.getLogger("thalamus.adapter")


# ───────────────────────────── 连接池 ─────────────────────────────


class ConnectionPool:
    """每个 base_url 一个长连接客户端，所有 key 共享。跨洲握手 400ms+，热连接直接省掉。"""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()

    async def client(self, channel: Channel) -> httpx.AsyncClient:
        c = self._clients.get(channel.base_url)
        if c is not None:
            return c
        async with self._lock:
            c = self._clients.get(channel.base_url)
            if c is None:
                c = httpx.AsyncClient(
                    base_url=channel.base_url,
                    transport=self._transport,
                    limits=httpx.Limits(max_connections=256, max_keepalive_connections=64, keepalive_expiry=120),
                    # 读超时由引擎的首 token / 空闲超时接管，这里不设
                    timeout=httpx.Timeout(connect=channel.connect_timeout_s, read=None, write=30.0, pool=10.0),
                )
                self._clients[channel.base_url] = c
            return c

    def temporary(self, base_url: str, connect_timeout: float = 8.0) -> httpx.AsyncClient:
        """一次性客户端：管理台探测任意地址时用，不进连接池（否则乱填的地址会留下客户端）。"""
        return httpx.AsyncClient(
            base_url=base_url,
            transport=self._transport,
            timeout=httpx.Timeout(connect=connect_timeout, read=20.0, write=20.0, pool=5.0),
        )

    async def warm(self, channels: list[Channel]) -> dict[str, str]:
        result: dict[str, str] = {}

        async def one(ch: Channel) -> None:
            try:
                c = await self.client(ch)
                await c.get(
                    "/models",
                    headers={"Authorization": f"Bearer {ch.api_key}", "User-Agent": ch.user_agent},
                    timeout=8.0,
                )
                result[ch.id] = "warm"
            except Exception as e:  # noqa: BLE001 - 预热失败不阻塞启动
                result[ch.id] = f"cold ({type(e).__name__})"

        await asyncio.gather(*(one(ch) for ch in channels if ch.configured))
        return result

    async def aclose(self) -> None:
        await asyncio.gather(*(c.aclose() for c in self._clients.values()), return_exceptions=True)
        self._clients.clear()


# ───────────────────────────── 错误分类 ─────────────────────────────

_RATE_HINTS = ("rate limit", "rate_limit", "too many requests", "concurrency", "并发", "频率", "速率", "请求过多", "1302", "1303")
_QUOTA_HINTS = (
    "insufficient", "quota", "balance", "exceeded your", "usage limit", "limit reached", "billing",
    "余额", "额度", "上限", "套餐", "欠费", "1113", "1308", "1310",
)
_CONTEXT_HINTS = (
    "context length", "context_length", "maximum context", "too many tokens", "reduce the length",
    "prompt is too long", "context window", "上下文", "长度超",
)
# "请求没问题，是这个模型吃不下它"。这些 400 换一个模型就能成，不该当死路交给客户端。
_INCOMPATIBLE_HINTS = (
    "reasoning_content",            # GLM：思考模式要求把上一轮的 reasoning_content 回传
    "must be passed back",
    "thinking mode",
    "only 1 is allowed for this model",  # kimi-k2.7-code 只接受 temperature=1
    "does not support",
    "is not supported with",
    "unsupported parameter",
    "unsupported value",
)


def _retry_after(headers: httpx.Headers | None) -> float | None:
    if not headers:
        return None
    v = headers.get("retry-after")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def classify_http(status: int, body: str, headers: httpx.Headers | None = None) -> UpstreamError:
    low = body.lower()
    ra = _retry_after(headers)
    msg = body[:500]
    if status in (401, 403):
        if any(h in low for h in _QUOTA_HINTS):
            return UpstreamError(FailureClass.QUOTA_EXHAUSTED, msg, status, ra)
        return UpstreamError(FailureClass.AUTH_FAILED, msg, status, ra)
    if status == 402:
        return UpstreamError(FailureClass.QUOTA_EXHAUSTED, msg, status, ra)
    if status == 429:
        if any(h in low for h in _RATE_HINTS):
            return UpstreamError(FailureClass.RATE_LIMITED, msg, status, ra)
        if any(h in low for h in _QUOTA_HINTS):
            return UpstreamError(FailureClass.QUOTA_EXHAUSTED, msg, status, ra)
        return UpstreamError(FailureClass.RATE_LIMITED, msg, status, ra)
    if status in (400, 413, 422):
        if any(h in low for h in _CONTEXT_HINTS):
            return UpstreamError(FailureClass.CONTEXT_OVERFLOW, msg, status, ra)
        if any(h in low for h in _QUOTA_HINTS):
            return UpstreamError(FailureClass.QUOTA_EXHAUSTED, msg, status, ra)
        if any(h in low for h in _INCOMPATIBLE_HINTS):
            return UpstreamError(FailureClass.MODEL_INCOMPATIBLE, msg, status, ra)
        return UpstreamError(FailureClass.CLIENT_ERROR, msg, status, ra)
    if status in (404, 405):
        # 模型名在该渠道不存在 / 路径不对：换渠道可能成功，并且应当熔断这个候选
        return UpstreamError(FailureClass.PROVIDER_DOWN, msg, status, ra)
    if 400 <= status < 500:
        return UpstreamError(FailureClass.CLIENT_ERROR, msg, status, ra)
    return UpstreamError(FailureClass.PROVIDER_DOWN, msg, status, ra)


def classify_exception(exc: BaseException) -> UpstreamError:
    if isinstance(exc, UpstreamError):
        return exc
    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
        return UpstreamError(FailureClass.TIMEOUT, f"{type(exc).__name__}: {exc}")
    if isinstance(exc, httpx.HTTPError):
        return UpstreamError(FailureClass.PROVIDER_DOWN, f"{type(exc).__name__}: {exc}")
    return UpstreamError(FailureClass.INVALID_RESPONSE, f"{type(exc).__name__}: {exc}")


# ───────────────────────────── SSE ─────────────────────────────


async def iter_sse(resp: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    event, data_lines = "", []
    async for raw in resp.aiter_lines():
        line = raw.rstrip("\r")
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = "", []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield event, "\n".join(data_lines)


def safe_json(s: str) -> dict[str, Any] | None:
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


# ───────────────────────────── <think> 拆分 ─────────────────────────────


def _partial_suffix(s: str, tag: str) -> int:
    for k in range(min(len(tag) - 1, len(s)), 0, -1):
        if tag.startswith(s[-k:]):
            return k
    return 0


class ThinkSplitter:
    """把正文里的 <think>…</think> 挪到 reasoning，跨分片安全。"""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.inside = False
        self.buf = ""

    def feed(self, text: str) -> tuple[str, str]:
        s, self.buf = self.buf + text, ""
        content: list[str] = []
        reasoning: list[str] = []
        while s:
            tag = self.CLOSE if self.inside else self.OPEN
            i = s.find(tag)
            target = reasoning if self.inside else content
            if i < 0:
                keep = _partial_suffix(s, tag)
                target.append(s[: len(s) - keep])
                self.buf = s[len(s) - keep :]
                break
            target.append(s[:i])
            s = s[i + len(tag) :]
            self.inside = not self.inside
        return "".join(content), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        s, self.buf = self.buf, ""
        return ("", s) if self.inside else (s, "")


# ───────────────────────────── 适配器基类 ─────────────────────────────


@dataclass
class CallContext:
    session_key: str
    request_id: str
    max_output_cap: int = 0
    context_room: int = 0  # 该模型窗口里输入之外还剩多少 token；0 = 未知/不限


def resolve_max_tokens(
    body: dict[str, Any], offer: Offer, cap: int, default: int | None = None, room: int = 0
) -> int | None:
    req = body.get("max_completion_tokens") or body.get("max_tokens")
    mt = int(req) if isinstance(req, (int, float)) and req > 0 else None
    if cap > 0:
        mt = min(mt, cap) if mt else cap
    if mt is None:
        mt = default
    if mt and offer.min_max_tokens and mt < offer.min_max_tokens and (cap <= 0 or cap >= offer.min_max_tokens):
        # reasoning 模型预算太小时正文会是空的
        mt = offer.min_max_tokens
    if mt:
        mt = min(mt, offer.model.max_output)
        if room > 0:
            # 长上下文里输入和输出共用一个窗口：再大的 max_tokens 也不能越过剩余空间。
            # 客户端没给 max_tokens 时不替它设，交给上游默认值（路由已保证 room 够用）。
            mt = min(mt, room)
    return mt


class Adapter:
    def __init__(self, offer: Offer, pool: ConnectionPool):
        self.offer = offer
        self.pool = pool

    def headers(self, ctx: CallContext) -> dict[str, str]:
        ch = self.offer.channel
        h = {
            "Authorization": f"Bearer {ch.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": ch.user_agent,
        }
        for k, v in ch.headers.items():
            h[k] = v.replace("{session}", ctx.session_key).replace("{request_id}", ctx.request_id)
        return h

    def build(self, body: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        raise NotImplementedError

    def parse(self, event: str, data: dict[str, Any]) -> list[Delta]:
        raise NotImplementedError

    def finish(self) -> list[Delta]:
        return []

    async def stream(self, body: dict[str, Any], ctx: CallContext) -> AsyncIterator[Delta]:
        client = await self.pool.client(self.offer.channel)
        payload = self.build(body, ctx)
        req = client.build_request("POST", self.offer.path, json=payload, headers=self.headers(ctx))
        try:
            resp = await client.send(req, stream=True)
        except Exception as exc:  # noqa: BLE001
            raise classify_exception(exc) from exc
        try:
            if resp.status_code >= 400:
                raw = (await resp.aread()).decode("utf-8", "replace")
                raise classify_http(resp.status_code, raw, resp.headers)
            async for event, data_str in iter_sse(resp):
                if data_str.strip() == "[DONE]":
                    break
                data = safe_json(data_str)
                if data is None:
                    continue
                err = data.get("error")
                if err and not data.get("choices"):
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    code = err.get("code") if isinstance(err, dict) else None
                    status = int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit() and len(code) == 3) else 500
                    raise classify_http(status, f"{msg} {code or ''}")
                for d in self.parse(event, data):
                    yield d
            for d in self.finish():
                yield d
        except UpstreamError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise classify_exception(exc) from exc
        finally:
            await resp.aclose()

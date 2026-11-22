"""Anthropic Messages 方言（OpenCode Go 上的 MiniMax / Qwen 等走 /messages 的模型）。

客户端永远说 OpenAI 格式；这里双向翻译：
  请求  system → system；tool 角色 → tool_result；assistant.tool_calls → tool_use；
        image_url(data:/http) → image；tools/tool_choice 转换；相邻同角色消息合并
  响应  text_delta → content；thinking_delta → reasoning；
        tool_use + input_json_delta → OpenAI tool_calls 增量（保持 index/id/name）
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from ..features import content_text
from ..types import Delta, FailureClass, UpstreamError, Usage
from .base import Adapter, CallContext, ThinkSplitter, resolve_max_tokens

_STOP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "stop",
}


def _image_block(url: str) -> dict[str, Any] | None:
    if not url:
        return None
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        media = head[5:].split(";")[0] or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    out: list[dict[str, Any]] = []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text") and p.get("text"):
                out.append({"type": "text", "text": str(p["text"])})
            elif t == "image_url":
                iu = p.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                b = _image_block(str(url or ""))
                if b:
                    out.append(b)
    return out


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            text, _ = content_text(m.get("content"))
            if text:
                system.append({"type": "text", "text": text})
        elif role == "tool":
            text, _ = content_text(m.get("content"))
            push("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "", "content": text or ""}])
        elif role == "assistant":
            blocks = _blocks(m.get("content"))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                try:
                    inp = json.loads(args) if isinstance(args, str) else args
                except (json.JSONDecodeError, TypeError):
                    inp = {"_raw": args}
                if not isinstance(inp, dict):
                    inp = {"value": inp}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                        "name": fn.get("name", ""),
                        "input": inp,
                    }
                )
            push("assistant", blocks)
        else:
            push("user", _blocks(m.get("content")))

    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "."}]})
    return system, out


def to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    res = []
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type", "function") != "function":
            continue
        fn = t.get("function") or {}
        res.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return res


def to_anthropic_tool_choice(tc: Any, parallel: Any) -> dict[str, Any] | None:
    choice: dict[str, Any] | None = None
    if tc == "auto":
        choice = {"type": "auto"}
    elif tc == "required":
        choice = {"type": "any"}
    elif tc == "none":
        choice = {"type": "none"}
    elif isinstance(tc, dict) and tc.get("type") == "function":
        choice = {"type": "tool", "name": (tc.get("function") or {}).get("name", "")}
    if parallel is False:
        choice = choice or {"type": "auto"}
        if choice["type"] != "none":
            choice["disable_parallel_tool_use"] = True
    return choice


class AnthropicAdapter(Adapter):
    def __init__(self, offer, pool):
        super().__init__(offer, pool)
        self._think = ThinkSplitter() if offer.think_tags else None
        self._tool_index: dict[int, int] = {}

    def headers(self, ctx: CallContext) -> dict[str, str]:
        h = super().headers(ctx)
        h["x-api-key"] = self.offer.channel.api_key
        h["anthropic-version"] = "2023-06-01"
        return h

    def build(self, body: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        system, messages = to_anthropic_messages(body.get("messages") or [])
        p: dict[str, Any] = {
            "model": self.offer.upstream,
            "messages": messages,
            "stream": True,
            "max_tokens": resolve_max_tokens(
                body, self.offer, ctx.max_output_cap,
                default=min(8192, self.offer.model.max_output), room=ctx.context_room,
            ),
        }
        if system:
            p["system"] = system
        tools = to_anthropic_tools(body.get("tools"))
        if tools:
            p["tools"] = tools
            choice = to_anthropic_tool_choice(body.get("tool_choice"), body.get("parallel_tool_calls"))
            if choice:
                p["tool_choice"] = choice
        if isinstance(body.get("temperature"), (int, float)):
            p["temperature"] = max(0.0, min(float(body["temperature"]), 1.0))
        if isinstance(body.get("top_p"), (int, float)):
            p["top_p"] = body["top_p"]
        stop = body.get("stop")
        if stop:
            p["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        p.update(self.offer.extra_body)
        return p

    @staticmethod
    def _usage(u: dict[str, Any] | None) -> Usage | None:
        if not isinstance(u, dict):
            return None
        g = lambda k: int(u.get(k) or 0)  # noqa: E731
        return Usage(
            input_miss=g("input_tokens") + g("cache_creation_input_tokens"),
            input_cached=g("cache_read_input_tokens"),
            output=g("output_tokens"),
        )

    def _text(self, text: str) -> Delta:
        if self._think:
            c, r = self._think.feed(text)
            return Delta(content=c, reasoning=r)
        return Delta(content=text)

    def parse(self, event: str, data: dict[str, Any]) -> list[Delta]:
        etype = data.get("type") or event
        if etype == "message_start":
            u = self._usage((data.get("message") or {}).get("usage"))
            return [Delta(usage=u)] if u else []
        if etype == "content_block_start":
            cb = data.get("content_block") or {}
            if cb.get("type") == "tool_use":
                idx = len(self._tool_index)
                self._tool_index[int(data.get("index", idx))] = idx
                return [
                    Delta(
                        tool_calls=[
                            {
                                "index": idx,
                                "id": cb.get("id") or f"call_{uuid.uuid4().hex[:20]}",
                                "type": "function",
                                "function": {"name": cb.get("name", ""), "arguments": ""},
                            }
                        ]
                    )
                ]
            if cb.get("type") == "text" and cb.get("text"):
                return [self._text(cb["text"])]
            return []
        if etype == "content_block_delta":
            d = data.get("delta") or {}
            dt = d.get("type")
            if dt == "text_delta":
                return [self._text(d.get("text", ""))]
            if dt == "thinking_delta":
                return [Delta(reasoning=d.get("thinking", ""))]
            if dt == "input_json_delta":
                idx = self._tool_index.get(int(data.get("index", -1)))
                if idx is None or not d.get("partial_json"):
                    return []
                return [Delta(tool_calls=[{"index": idx, "function": {"arguments": d["partial_json"]}}])]
            return []
        if etype == "message_delta":
            stop = (data.get("delta") or {}).get("stop_reason")
            u = self._usage(data.get("usage"))
            if stop or u:
                return [Delta(finish_reason=_STOP.get(stop, stop) if stop else None, usage=u)]
            return []
        if etype == "error":
            err = data.get("error") or {}
            et, msg = err.get("type", ""), err.get("message", "")
            if et == "rate_limit_error":
                raise UpstreamError(FailureClass.RATE_LIMITED, msg, 429)
            if et in ("overloaded_error", "api_error"):
                raise UpstreamError(FailureClass.PROVIDER_DOWN, msg, 529)
            if et == "authentication_error":
                raise UpstreamError(FailureClass.AUTH_FAILED, msg, 401)
            raise UpstreamError(FailureClass.INVALID_RESPONSE, f"{et}: {msg}")
        return []

    def finish(self) -> list[Delta]:
        if not self._think:
            return []
        c, r = self._think.flush()
        return [Delta(content=c, reasoning=r)] if (c or r) else []

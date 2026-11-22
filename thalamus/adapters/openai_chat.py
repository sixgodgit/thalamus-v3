"""OpenAI Chat Completions 方言（DeepSeek、GLM Coding Plan、OpenCode Go 的大部分模型）。

策略是"近乎原样透传"：tools、tool_choice、图片、response_format 等字段不做有损转换，
只改 model、强制流式并要求返回 usage、按 key 限制 max_tokens。
"""
from __future__ import annotations

from typing import Any

from ..types import Delta, Usage
from .base import Adapter, CallContext, ThinkSplitter, resolve_max_tokens

_DROP = {
    "model", "stream", "stream_options", "max_tokens", "max_completion_tokens",
    "policy", "session_id", "thalamus",
    "n",  # 多份候选回答会成倍消耗，而网关只转发第一份
}
_DELTA_KNOWN = {"role", "content", "reasoning_content", "reasoning", "tool_calls"}


def parse_usage(u: dict[str, Any] | None) -> Usage | None:
    if not isinstance(u, dict):
        return None

    def num(d: dict[str, Any], *names: str, default: int = 0) -> int:
        for n in names:
            v = d.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return default

    prompt = num(u, "prompt_tokens", "input_tokens")
    output = num(u, "completion_tokens", "output_tokens")
    details_out = u.get("completion_tokens_details") or {}
    reasoning = num(details_out, "reasoning_tokens") if isinstance(details_out, dict) else 0
    hit = num(u, "prompt_cache_hit_tokens", default=-1)
    miss = num(u, "prompt_cache_miss_tokens", default=-1)
    if hit >= 0 and miss >= 0:  # DeepSeek
        return Usage(input_miss=miss, input_cached=hit, output=output, reasoning=reasoning)
    details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    cached = num(details, "cached_tokens") if isinstance(details, dict) else 0
    cached = cached or num(u, "cached_tokens")
    cached = min(cached, prompt)
    return Usage(input_miss=max(prompt - cached, 0), input_cached=cached, output=output, reasoning=reasoning)


class OpenAIChatAdapter(Adapter):
    def __init__(self, offer, pool):
        super().__init__(offer, pool)
        self._think = ThinkSplitter() if offer.think_tags else None

    def build(self, body: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        p = {k: v for k, v in body.items() if k not in _DROP}
        p.update(self.offer.extra_body)
        p["model"] = self.offer.upstream
        p["stream"] = True
        p["stream_options"] = {"include_usage": True}
        mt = resolve_max_tokens(body, self.offer, ctx.max_output_cap, room=ctx.context_room)
        if mt:
            p["max_tokens"] = mt
        return p

    def parse(self, event: str, data: dict[str, Any]) -> list[Delta]:
        usage = parse_usage(data.get("usage"))
        choices = data.get("choices") or []
        if not choices:
            return [Delta(usage=usage)] if usage else []
        c0 = choices[0] or {}
        d = c0.get("delta") or c0.get("message") or {}
        content = d.get("content") or ""
        reasoning = d.get("reasoning_content") or d.get("reasoning") or ""
        if not isinstance(reasoning, str):
            reasoning = ""
        if self._think and content:
            content, r2 = self._think.feed(content)
            reasoning += r2
        extra = {k: v for k, v in d.items() if k not in _DELTA_KNOWN and v is not None}
        out = Delta(
            content=content if isinstance(content, str) else "",
            reasoning=reasoning,
            tool_calls=d.get("tool_calls") or None,
            finish_reason=c0.get("finish_reason"),
            usage=usage,
            extra=extra,
        )
        if not (out.meaningful or out.finish_reason or out.usage or out.extra):
            return []
        return [out]

    def finish(self) -> list[Delta]:
        if not self._think:
            return []
        c, r = self._think.flush()
        return [Delta(content=c, reasoning=r)] if (c or r) else []

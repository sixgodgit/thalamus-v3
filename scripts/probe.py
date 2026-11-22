"""上线前核对真实上游。会产生少量真实费用 / 订阅额度消耗。

    python scripts/probe.py                      # 所有已配置候选：基础流式对话
    python scripts/probe.py --tools --cache      # 额外测工具调用与前缀缓存命中
    python scripts/probe.py --offer glm-5.3@glm_plan
    python scripts/probe.py --models             # 列出各渠道 /models 返回的模型名

逐项检查：模型名是否存在、首 token 时间、正文是否为空（reasoning 模型预算不足时常见）、
usage 字段能否被解析（记账是否准确）、工具调用能否还原、缓存是否命中。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thalamus.env import load_env  # noqa: E402

load_env(ROOT / ".env")

from thalamus.adapters import CallContext, ConnectionPool, make_adapter  # noqa: E402
from thalamus.config import Offer, load_settings  # noqa: E402
from thalamus.types import UpstreamError, Usage  # noqa: E402

STABLE_PREFIX = (
    "你是一个严谨的助手。以下是一份固定的参考资料，用于测试前缀缓存，请忽略其内容本身。\n"
    + "\n".join(f"第 {i} 条：Thalamus 是一个模型路由网关，负责在多个上游之间按成本、质量与延迟选择模型。" for i in range(120))
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


async def call(pool: ConnectionPool, offer: Offer, body: dict, session: str) -> dict:
    ad = make_adapter(offer, pool)
    ctx = CallContext(session, "probe-" + uuid.uuid4().hex[:8])
    t0 = time.perf_counter()
    out = {"ttft": None, "text": "", "reasoning": "", "tools": {}, "usage": Usage(), "finish": None, "error": None}
    try:
        async for d in ad.stream(body, ctx):
            if d.meaningful and out["ttft"] is None:
                out["ttft"] = time.perf_counter() - t0
            out["text"] += d.content
            out["reasoning"] += d.reasoning
            for tc in d.tool_calls or []:
                cur = out["tools"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                cur["name"] = cur["name"] or fn.get("name", "")
                cur["arguments"] += fn.get("arguments") or ""
            if d.usage:
                out["usage"] = out["usage"].merge(d.usage)
            if d.finish_reason:
                out["finish"] = d.finish_reason
    except UpstreamError as e:
        out["error"] = f"{e.failure.value} (HTTP {e.status}) {e.message[:300]}"
    out["total"] = time.perf_counter() - t0
    return out


def show(label: str, r: dict) -> bool:
    if r["error"]:
        print(f"  ✗ {label}: {r['error']}")
        return False
    u: Usage = r["usage"]
    usage = f"入 {u.input_miss} / 缓存 {u.input_cached} / 出 {u.output}" if not u.empty else "⚠ 没有解析到 usage（记账将用估算）"
    ttft = f"{r['ttft']:.2f}s" if r["ttft"] else "—"
    print(f"  ✓ {label}: 首token {ttft} · 总 {r['total']:.2f}s · {usage} · finish={r['finish']}")
    if r["text"]:
        print(f"      正文: {r['text'][:80]!r}")
    elif not r["tools"]:
        print("      ⚠ 正文为空" + ("（只有 reasoning，考虑调大 min_max_tokens）" if r["reasoning"] else ""))
    for t in r["tools"].values():
        print(f"      工具调用: {t['name']}({t['arguments'][:80]})")
    return True


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offer", help="只测某个候选，如 glm-5.3@glm_plan")
    ap.add_argument("--channel", help="只测某个渠道")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--models", action="store_true")
    args = ap.parse_args()

    s = load_settings()
    pool = ConnectionPool()
    try:
        if args.models:
            for ch in s.channels.values():
                if not ch.configured:
                    continue
                client = await pool.client(ch)
                try:
                    r = await client.get("/models", headers={"Authorization": f"Bearer {ch.api_key}", "User-Agent": ch.user_agent}, timeout=20)
                    ids = sorted(m.get("id", "") for m in r.json().get("data", [])) if r.status_code == 200 else []
                    print(f"[{ch.id}] HTTP {r.status_code}: {', '.join(ids) or r.text[:200]}")
                    for o in s.offers:
                        if o.channel.id == ch.id and ids and o.upstream not in ids:
                            print(f"   ⚠ 配置里的 {o.upstream} 不在列表中")
                except Exception as e:  # noqa: BLE001
                    print(f"[{ch.id}] 失败: {e}")
            return

        offers = [o for o in s.usable_offers()
                  if (not args.offer or o.key == args.offer) and (not args.channel or o.channel.id == args.channel)]
        if not offers:
            print("没有可测试的候选：检查 .env 里的 API key 或 --offer/--channel 参数")
            return
        for o in offers:
            print(f"\n[{o.key}]  upstream={o.upstream}  dialect={o.dialect}")
            session = "probe-" + uuid.uuid4().hex[:12]
            ok = show("对话", await call(pool, o, {"messages": [{"role": "user", "content": "用一句话介绍你自己"}], "max_tokens": 200}, session))
            if not ok:
                continue
            if args.tools:
                show("工具", await call(pool, o, {
                    "messages": [{"role": "user", "content": "巴黎现在天气怎么样？请调用工具。"}],
                    "tools": TOOLS, "tool_choice": "auto", "max_tokens": 300,
                }, session))
            if args.cache:
                body = {"messages": [{"role": "system", "content": STABLE_PREFIX}, {"role": "user", "content": "回复 OK"}], "max_tokens": 20}
                first = await call(pool, o, body, session)
                await asyncio.sleep(2)
                second = await call(pool, o, body, session)
                show("缓存-第1次", first)
                show("缓存-第2次", second)
                u = second["usage"]
                if not second["error"]:
                    if u.input_cached > 0:
                        print(f"      ✓ 缓存命中 {u.input_cached}/{u.input_total}")
                    else:
                        print("      ⚠ 第二次没有缓存命中：若长期如此，把该候选的 cached 价格改成与 input 相同，否则成本估算偏低")
    finally:
        await pool.aclose()


if __name__ == "__main__":
    asyncio.run(main())

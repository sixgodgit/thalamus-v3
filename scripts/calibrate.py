"""模型质量校准：让各模型真实回答评测题，由裁判模型打分，给出 quality 建议值。

⚠ 会产生真实费用 / 消耗订阅额度。题目 12 道 × 模型数 × 2 次调用（回答 + 裁判）。

两种模式：

  ① 直连上游（在网关那台机器上跑，需要 .env 里的上游 key）
     python scripts/calibrate.py --judge deepseek-v4-pro@deepseek
     python scripts/calibrate.py --judge deepseek-v4-pro@deepseek --offers kimi-k2.7-code@opencode_go,glm-5.3@glm_plan

  ② 经网关（任何机器都能跑，只要一把 Thalamus key —— 给 agent 用的就是这个）
     python scripts/calibrate.py --via https://api.example.com --key sk-thl-xxx --judge deepseek-v4-pro
     模型名从网关的 /v1/models 取；花费受那把 key 的每日额度约束，跑飞了也有上限。

输出：
  eval/reports/calibration-<时间>.json   每题的分数、延迟、成本与裁判理由
  终端打印按 (模型, 任务) 汇总的均分，以及可以粘贴进 config/upstreams.yaml 的 quality 建议值
不会自动修改配置。裁判模型也会犯错，建议人工抽查报告里的低分样本后再采纳。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thalamus.env import load_env  # noqa: E402

load_env(ROOT / ".env")

from thalamus.adapters import CallContext, ConnectionPool, make_adapter  # noqa: E402
from thalamus.config import Offer, load_settings  # noqa: E402
from thalamus.evaluation import aggregate, judge_messages, load_jsonl, parse_score, quality_yaml, suggest_quality  # noqa: E402
from thalamus.types import UpstreamError, Usage  # noqa: E402


# ⚠ 这三个常量必须在用到它们做默认参数的函数之前定义：默认值在 def 执行时就求值，
# 放到下面会在 import 阶段直接 NameError（连 --help 都跑不起来）。
MIN_SAMPLES = 2  # 单个样本就改先验太激进：12 道题每类只有 1-2 道，不足就保持原值
ANSWER_TOKENS = 4096  # 思考型模型要留够预算，否则正文被思考挤没
JUDGE_TOKENS = 1500  # 同理：裁判也可能是思考型模型，400 会让它只输出思考、正文为空


async def ask(pool: ConnectionPool, offer: Offer, messages: list[dict], max_tokens: int = ANSWER_TOKENS) -> dict:
    """问一次。

    三件事是 2026-10-04 第一次真实校准踩出来的：
      · 不发 temperature —— kimi-k2.7-code 只接受 1，发别的值直接 400，一题都答不了
      · 要收集 reasoning —— 思考型模型（glm-5.3 是 always）可能把预算全花在思考上，
        只看 content 会以为它返回了空
      · 预算要给够 —— 2048 对思考型模型不够，正文会被挤没
    """
    ad = make_adapter(offer, pool)
    t0 = time.perf_counter()
    text, think, usage, err = "", "", Usage(), None
    try:
        async for d in ad.stream({"messages": messages, "max_tokens": max_tokens},
                                 CallContext("calibrate-" + uuid.uuid4().hex[:8], "calibrate")):
            text += d.content
            think += d.reasoning
            if d.usage:
                usage = usage.merge(d.usage)
    except UpstreamError as e:
        err = f"{e.failure.value}: {e.message[:200]}"
    cost = offer.list_usd(usage.input_miss, usage.input_cached, usage.output)
    return {"text": text, "reasoning": think, "latency_s": time.perf_counter() - t0,
            "cost_usd": cost, "error": err}


def report(rows: list[dict], models: dict[str, Any], blend: float, priors: dict[str, Any]) -> None:
    agg = aggregate(rows)
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"rows": rows, "aggregate": {f"{m}|{t}": v for (m, t), v in agg.items()}},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {path}\n")
    print(f"{'模型':<22}{'任务':<9}{'样本':>5}{'均分':>6}{'先验':>6}{'裁判废':>7}{'延迟':>7}{'成本$':>10}")
    for (m, t), a in sorted(agg.items()):
        prior = priors[m].q(t) if m in priors else float("nan")
        mean = "  —" if a["mean"] != a["mean"] else f"{a['mean']:.1f}"
        print(f"{m:<22}{t:<9}{int(a['n']):>5}{mean:>6}{prior:>6.1f}{int(a.get('skipped', 0)):>7}"
              f"{a['latency_s']:>6.1f}s{a['cost_usd']:>10.5f}")
    print()
    print(quality_yaml(suggest_quality(models, agg, blend, min_n=MIN_SAMPLES)))

    thin = [(m, t) for (m, t), a in agg.items() if a["n"] < MIN_SAMPLES]
    if thin:
        print(f"\n样本不足（< {MIN_SAMPLES}）因此保持先验不变：" + ", ".join(f"{m}/{t}" for m, t in sorted(thin)))

    unscored = [r for r in rows if r.get("score") is None]
    if unscored:
        kinds: Counter = Counter(r.get("unscored") or "未说明" for r in unscored)
        print(f"\n⚠ {len(unscored)}/{len(rows)} 题没有打到分，**全部已排除**（不算 0 分）："
              + "，".join(f"{k} {n} 次" for k, n in kinds.most_common()))
        for r in unscored[:12]:
            print(f"  {r['model']:<22}{r['id']:<18}{r.get('unscored', ''):<10}{str(r.get('judge'))[:70]}")
        print("  —— 没打到分不代表模型差：请求参数不被接受、思考挤掉正文、裁判自己空输出，"
              "都与被测模型的质量无关。先把这些修掉再看分数。")


async def run_via_gateway(args: Any) -> None:
    """经网关模式：不需要上游 key，模型名从 /v1/models 取。"""
    if not args.key:
        sys.exit("--via 模式需要 --key（Thalamus API key）")
    s = load_settings()  # 只用来取标价做成本估算；没有上游 key 也能加载
    first_offer = {}
    for o in s.offers:
        first_offer.setdefault(o.model.key, o)
    gw = ViaGateway(args.via, args.key, first_offer)
    try:
        available = await gw.models_from_gateway()
        models = [m.strip() for m in args.models.split(",")] if args.models else available
        missing = [m for m in models if m not in available]
        if missing:
            sys.exit(f"这把 key 用不了这些模型：{missing}\n可用：{available}")
        if args.judge not in available:
            sys.exit(f"裁判模型 {args.judge} 不在可用列表里：{available}")
        tasks = load_jsonl(args.tasks)
        print(f"经 {args.via} 评测 {len(models)} 个模型 × {len(tasks)} 题，裁判 {args.judge}。"
              f"会产生真实费用，受这把 key 的每日额度约束。")

        sem = asyncio.Semaphore(args.concurrency)
        rows: list[dict] = []

        async def one(model: str, t: dict) -> None:
            async with sem:
                ans = await gw.ask(model, [{"role": "user", "content": t["prompt"]}])
                score, reason, why = None, ans["error"], "请求失败" if ans["error"] else ""
                if not ans["error"]:
                    if not ans["text"].strip():
                        why = "正文为空"
                        reason = "正文为空（思考型模型可能把预算花在 reasoning 上）"
                    else:
                        j = await gw.ask(args.judge, judge_messages(t["prompt"], t["rubric"], ans["text"]),
                                         JUDGE_TOKENS)
                        score = parse_score(j["text"])
                        if score is None:
                            why = "裁判无输出"
                            reason = f"裁判输出无法解析: {(j['error'] or j['text'] or '(空)')[:200]}"
                        else:
                            reason = j["text"][:300]
                if ans["served_by"] and ans["served_by"] != model:
                    reason = f"⚠ 点名 {model} 实际由 {ans['served_by']} 服务；" + (reason or "")
                rows.append({
                    "offer": f"{model}@gateway", "model": model, "task": t["task"], "id": t["id"],
                    "score": score, "unscored": why, "error": ans["error"],
                    "latency_s": round(ans["latency_s"], 2), "cost_usd": ans["cost_usd"], "judge": reason,
                })
                print(f"  {model:<24} {t['id']:<20} {'—' if score is None else score:>4}  {ans['latency_s']:.1f}s")

        await asyncio.gather(*(one(m, t) for m in models for t in tasks))
    finally:
        await gw.aclose()
    specs = {m: s.models[m] for m in {r["model"] for r in rows} if m in s.models}
    report(rows, specs, args.blend, specs)


class ViaGateway:
    """经网关调用：只需要一把 Thalamus key，不需要上游 key。

    点名模型走的是真实数据面（鉴权 → 路由硬过滤 → 适配器），所以顺带也验证了
    "点名这个模型确实能用"。花费由那把 key 的每日额度兜住。
    """

    def __init__(self, base_url: str, api_key: str, models: dict[str, Offer | None]):
        import httpx

        self.prices = models
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=10.0),
        )

    async def models_from_gateway(self) -> list[str]:
        r = await self.client.get("/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"] if not str(m["id"]).startswith("thalamus/")]

    async def ask(self, model: str, messages: list[dict], max_tokens: int = ANSWER_TOKENS) -> dict:
        """不发 temperature：kimi-k2.7-code 只接受 1，发别的值直接 400。
        网关侧虽然有 extra_body 覆盖兜着，但校准脚本不该依赖那个 —— 和直连模式保持一致。
        默认预算用 ANSWER_TOKENS 而不是 2048：2048 会让思考型模型的正文被挤没，
        2026-10-04 那次 glm-5.3 大面积"正文为空"就是这么来的。"""
        t0 = time.perf_counter()
        try:
            r = await self.client.post("/v1/chat/completions", json={
                "model": model, "messages": messages, "max_tokens": max_tokens, "stream": False,
            })
        except Exception as e:  # noqa: BLE001
            return {"text": "", "latency_s": time.perf_counter() - t0, "cost_usd": 0.0,
                    "error": f"{type(e).__name__}: {str(e)[:200]}", "served_by": ""}
        latency = time.perf_counter() - t0
        served = r.headers.get("x-thalamus-model", "")
        if r.status_code >= 400:
            return {"text": "", "latency_s": latency, "cost_usd": 0.0,
                    "error": f"HTTP {r.status_code}: {r.text[:200]}", "served_by": served}
        body = r.json()
        text = (body["choices"][0]["message"].get("content") or "")
        u = body.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        miss = max(u.get("prompt_tokens", 0) - cached, 0)
        o = self.prices.get(model)
        cost = o.list_usd(miss, cached, u.get("completion_tokens", 0)) if o else 0.0
        return {"text": text, "latency_s": latency, "cost_usd": cost, "error": None, "served_by": served}

    async def aclose(self) -> None:
        await self.client.aclose()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="裁判：直连模式写 offer（model@channel），经网关模式写模型名")
    ap.add_argument("--offers", help="逗号分隔的候选；默认每个模型取第一个可用候选")
    ap.add_argument("--via", help="网关地址，如 https://api.example.com（给出则走经网关模式）")
    ap.add_argument("--key", help="Thalamus API key，经网关模式必填")
    ap.add_argument("--models", help="经网关模式下逗号分隔的模型名；默认取 /v1/models 里的全部真实模型")
    ap.add_argument("--tasks", default=str(ROOT / "eval" / "quality_tasks.jsonl"))
    ap.add_argument("--blend", type=float, default=0.5, help="实测分数在建议值中的权重 0-1")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    if args.via:
        await run_via_gateway(args)
        return

    s = load_settings()
    judge = s.offer(args.judge)
    if judge is None or not judge.usable:
        sys.exit(f"裁判候选 {args.judge} 不存在或未配置 key")
    if args.offers:
        offers = [s.offer(k) for k in args.offers.split(",")]
        if any(o is None or not o.usable for o in offers):
            sys.exit("--offers 中有不存在或未配置的候选")
    else:
        seen, offers = set(), []
        # 用 eligible_offers：被 min_context_tokens 门槛排除的模型路由永远不会选，
        # 花钱去测它没有意义（第一次跑时白测了 minimax-m3）
        for o in s.eligible_offers():
            if o.model.key not in seen:
                seen.add(o.model.key)
                offers.append(o)
    tasks = load_jsonl(args.tasks)
    print(f"将评测 {len(offers)} 个候选 × {len(tasks)} 题，裁判 {judge.key}。这会产生真实费用。")

    pool = ConnectionPool()
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    async def one(o: Offer, t: dict) -> None:
        async with sem:
            ans = await ask(pool, o, [{"role": "user", "content": t["prompt"]}])
            score, reason, why = None, ans["error"], "请求失败" if ans["error"] else ""
            if not ans["error"]:
                if not ans["text"].strip():
                    why = "正文为空"
                    reason = (f"只有思考没有正文（reasoning {len(ans['reasoning'])} 字符）"
                              if ans["reasoning"] else "正文与 reasoning 都为空")
                else:
                    j = await ask(pool, judge, judge_messages(t["prompt"], t["rubric"], ans["text"]), JUDGE_TOKENS)
                    score = parse_score(j["text"])
                    if score is None:
                        why = "裁判无输出"
                        reason = f"裁判输出无法解析: {(j['error'] or j['text'] or '(空)')[:200]}"
                    else:
                        reason = j["text"][:300]
            rows.append({
                "offer": o.key, "model": o.model.key, "task": t["task"], "id": t["id"], "score": score,
                "unscored": why, "error": ans["error"],
                "latency_s": round(ans["latency_s"], 2), "cost_usd": ans["cost_usd"], "judge": reason,
            })
            print(f"  {o.key:<34} {t['id']:<20} {'—' if score is None else score:>4}  {ans['latency_s']:.1f}s")

    try:
        await asyncio.gather(*(one(o, t) for o in offers for t in tasks))
    finally:
        await pool.aclose()

    report(rows, {o.model.key: o.model for o in offers}, args.blend, s.models)


if __name__ == "__main__":
    asyncio.run(main())

"""离线评测：路由分类准确率 + 模型质量校准的纯函数部分。

  scripts/eval_routing.py   用标注集评测分类（免费、离线、秒级）
  scripts/calibrate.py      让各模型真实回答评测题、由裁判模型打分，给出 quality 建议值（会产生费用）

评测集格式（JSONL，一行一个用例）：
  {"id": "nl-code-1", "lang": "nl", "text": "...", "expect": "code", "difficulty": [2, 3]}
  或用 "body" 给出完整请求（多轮、工具、图片），"model" 可写别名。
注意：仓库里自带的评测集是人工编写的种子数据，规则也是照着它调的。上线后请用真实（脱敏、经同意的）
请求补充标注，否则准确率数字只能说明"没有退化"，不能说明"对真实用户准确"。
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import TASKS, ModelSpec
from .features import extract
from .intent import classify, parse_alias

# ───────────────────────────── 路由评测 ─────────────────────────────


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: {e}") from None
    return out


def case_body(case: dict[str, Any]) -> dict[str, Any]:
    if "body" in case:
        return case["body"]
    text = case["text"]
    if isinstance(case.get("repeat"), int):
        text = text * case["repeat"]
    return {"messages": [{"role": "user", "content": text}], **(case.get("params") or {})}


def run_routing_eval(cases: list[dict[str, Any]], long_threshold: int = 120_000) -> dict[str, Any]:
    confusion: dict[str, Counter] = defaultdict(Counter)
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    misses: list[dict[str, Any]] = []
    diff_total = diff_ok = task_ok = 0
    for c in cases:
        body = case_body(c)
        at, ad, boost = parse_alias(body.get("model"), None)
        it = classify(extract(body, "eval"), long_threshold, at, ad, boost)
        exp = c["expect"]
        ok = it.task == exp
        task_ok += ok
        confusion[exp][it.task] += 1
        lang = c.get("lang", "?")
        by_lang[lang][0] += ok
        by_lang[lang][1] += 1
        by_task[exp][0] += ok
        by_task[exp][1] += 1
        d_ok = True
        if "difficulty" in c:
            lo, hi = c["difficulty"]
            d_ok = lo <= it.difficulty <= hi
            diff_total += 1
            diff_ok += d_ok
        if not ok or not d_ok:
            misses.append({
                "id": c.get("id"), "expect": exp, "got": it.task,
                "difficulty": it.difficulty, "want_difficulty": c.get("difficulty"), "signals": it.signals,
            })
    n = len(cases) or 1
    return {
        "total": len(cases),
        "task_accuracy": task_ok / n,
        "difficulty_accuracy": (diff_ok / diff_total) if diff_total else None,
        "by_lang": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in sorted(by_lang.items())},
        "by_task": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in sorted(by_task.items())},
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "misses": misses,
    }


def format_routing_report(r: dict[str, Any]) -> str:
    lines = [
        f"用例 {r['total']}  任务准确率 {r['task_accuracy']:.1%}"
        + (f"  难度准确率 {r['difficulty_accuracy']:.1%}" if r["difficulty_accuracy"] is not None else ""),
        "",
        "按语言: " + "  ".join(f"{k} {v['correct']}/{v['total']}" for k, v in r["by_lang"].items()),
        "按任务: " + "  ".join(f"{k} {v['correct']}/{v['total']}" for k, v in r["by_task"].items()),
        "",
        "混淆矩阵（行=期望，列=判定）:",
    ]
    cols = [t for t in TASKS if any(t in row for row in r["confusion"].values())]
    lines.append("          " + "".join(f"{c:>9}" for c in cols))
    for exp, row in r["confusion"].items():
        lines.append(f"{exp:>9} " + "".join(f"{row.get(c, 0):>9}" for c in cols))
    if r["misses"]:
        lines += ["", "未命中:"]
        for m in r["misses"]:
            lines.append(
                f"  {m['id']}: 期望 {m['expect']}{m['want_difficulty'] or ''} → 判为 {m['got']}/{m['difficulty']}  {m['signals']}"
            )
    return "\n".join(lines)


# ───────────────────────────── 质量校准 ─────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict evaluator of AI assistant answers. Score the ANSWER to the TASK on a 0-10 scale using the RUBRIC. "
    "10 = excellent and fully correct, 7 = good with minor issues, 5 = partially correct or incomplete, "
    "2 = mostly wrong, 0 = empty or irrelevant. Judge correctness first, then completeness, then clarity. "
    "Reply with ONLY a JSON object: {\"score\": <number>, \"reason\": \"<one sentence>\"}"
)


def judge_messages(prompt: str, rubric: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"TASK:\n{prompt}\n\nRUBRIC:\n{rubric}\n\nANSWER:\n{answer[:12000]}"},
    ]


_SCORE_JSON = re.compile(r"\{[^{}]*\"score\"\s*:\s*(-?\d+(?:\.\d+)?)[^{}]*\}", re.S)
_SCORE_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10")


def parse_score(text: str) -> float | None:
    if not text:
        return None
    m = _SCORE_JSON.search(text) or _SCORE_TEXT.search(text)
    if not m:
        return None
    return max(0.0, min(10.0, float(m.group(1))))


def aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """rows: {model, task, score|None, latency_s, cost_usd}。

    **只平均真正打到分的那些**；`score is None` 一律不计入，而不是算 0 分。
    2026-10-04 第一次真实校准证明了这条的必要性 —— 当时"失败按 0 分计"把三类
    与模型质量无关的东西混进均分，直接把先验改坏：

      · 我们发的 temperature=0.2 被某个模型拒绝（400）→ 它一题没答却被判 0 分
      · 裁判自己输出为空（思考型模型 + 预算太小）→ 被判 0 分
      · 模型把预算花在 reasoning 上、正文为空（而当时没收集 reasoning）→ 被判 0 分

    质量 = "答出来的东西有多好"。答不出来属于可靠性，由熔断器的错误率与
    OutcomeBook 的坏比率单独计量，不该在这里重复计一次。
    """
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped: Counter = Counter()
    for r in rows:
        k = (r["model"], r["task"])
        if r.get("score") is None:
            skipped[k] += 1
            continue
        acc[k]["score"].append(float(r["score"]))
        acc[k]["latency"].append(float(r.get("latency_s") or 0))
        acc[k]["cost"].append(float(r.get("cost_usd") or 0))
    out = {
        k: {
            "n": len(v["score"]),
            "mean": sum(v["score"]) / len(v["score"]),
            "latency_s": sum(v["latency"]) / len(v["latency"]),
            "cost_usd": sum(v["cost"]) / len(v["cost"]),
            "skipped": float(skipped.get(k, 0)),
        }
        for k, v in acc.items()
    }
    for k, n in skipped.items():  # 全部被裁判吞掉的组合也要出现，否则报告里看不见
        out.setdefault(k, {"n": 0, "mean": float("nan"), "latency_s": 0.0, "cost_usd": 0.0, "skipped": float(n)})
    return out


def suggest_quality(
    models: dict[str, ModelSpec], agg: dict[tuple[str, str], dict[str, float]], blend: float = 0.5, min_n: int = 2
) -> dict[str, dict[str, float]]:
    """把先验和实测分数按 blend 混合。样本太少的保持先验。"""
    out: dict[str, dict[str, float]] = {}
    for mkey, spec in models.items():
        q = dict(spec.quality)
        for (m, task), a in agg.items():
            if m != mkey or a["n"] < min_n or a["mean"] != a["mean"]:  # NaN = 一个有效样本都没有
                continue
            prior = spec.q(task)
            q[task] = round(prior * (1 - blend) + a["mean"] * blend, 1)
        out[mkey] = q
    return out


def quality_yaml(suggested: dict[str, dict[str, float]]) -> str:
    lines = ["# 建议值：复制到 config/upstreams.yaml 对应模型的 quality 字段"]
    for m, q in suggested.items():
        inner = ", ".join(f"{t}: {q[t]}" for t in TASKS if t in q)
        lines.append(f"{m}:\n  quality: {{{inner}}}")
    return "\n".join(lines)

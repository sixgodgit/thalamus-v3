"""评测工具本身的测试 + 用种子评测集做分类回归门槛。"""
import importlib.util
import sys

import pytest
from helpers import ROOT
from thalamus.config import load_settings
from thalamus.evaluation import aggregate, load_jsonl, parse_score, quality_yaml, run_routing_eval, suggest_quality

MIN_TASK_ACCURACY = 0.9


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "scripts").glob("*.py")))
def test_scripts_import_cleanly(script, monkeypatch):
    """scripts/ 一直没被任何测试 import 过，于是 calibrate.py 里
    `max_tokens: int = ANSWER_TOKENS` 写在常量定义之前（默认值在 def 执行时就求值）
    带着 NameError 躺了两天 —— 连 --help 都跑不起来，而它是给 agent 用的入口。
    import 一遍是最便宜的护栏：语法、顶层求值顺序、模块级副作用都能兜住。"""
    monkeypatch.setattr(sys, "argv", [script])
    spec = importlib.util.spec_from_file_location(f"_script_{script[:-3]}", ROOT / "scripts" / script)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_routing_eval_meets_threshold():
    cases = load_jsonl(ROOT / "eval" / "routing_cases.jsonl")
    assert len(cases) >= 40
    r = run_routing_eval(cases)
    assert r["task_accuracy"] >= MIN_TASK_ACCURACY, r["misses"]
    assert set(r["by_lang"]) >= {"zh", "en", "nl"}
    for lang, v in r["by_lang"].items():
        assert v["accuracy"] >= 0.8, (lang, r["misses"])


def test_quality_tasks_file_is_valid():
    tasks = load_jsonl(ROOT / "eval" / "quality_tasks.jsonl")
    s = load_settings(ROOT / "config")
    assert {t["task"] for t in tasks} <= set(s.policy.tasks)
    assert all(t["prompt"] and t["rubric"] and t["id"] for t in tasks)
    assert len({t["id"] for t in tasks}) == len(tasks)


def test_parse_score():
    assert parse_score('{"score": 7.5, "reason": "ok"}') == 7.5
    assert parse_score('```json\n{"reason": "x", "score": 11}\n```') == 10.0
    assert parse_score("I'd give it 6/10.") == 6.0
    assert parse_score("great answer") is None


def test_aggregate_and_suggest():
    s = load_settings(ROOT / "config")
    rows = [
        {"model": "glm-5.3", "task": "code", "score": 9, "latency_s": 2, "cost_usd": 0.001},
        {"model": "glm-5.3", "task": "code", "score": 4, "latency_s": 1, "cost_usd": 0},
        {"model": "glm-5.3", "task": "chat", "score": 5, "latency_s": 1, "cost_usd": 0},
    ]
    agg = aggregate(rows)
    assert agg[("glm-5.3", "code")]["mean"] == 6.5 and agg[("glm-5.3", "code")]["n"] == 2
    sug = suggest_quality({"glm-5.3": s.models["glm-5.3"]}, agg, blend=0.5, min_n=2)
    assert sug["glm-5.3"]["code"] == round(8.6 * 0.5 + 6.5 * 0.5, 1)
    assert sug["glm-5.3"]["chat"] == 8.3  # 样本不足，保持先验
    assert "glm-5.3:" in quality_yaml(sug)


def test_unscored_rows_never_count_as_zero():
    """第一次真实校准（2026-10-04）的教训：三类"没打到分"都与模型质量无关，
    按 0 分计入会直接把先验改坏 ——
      · 我们发的 temperature 被上游拒绝（400），模型一题没答
      · 思考型模型把预算花在 reasoning 上，正文为空
      · 裁判自己输出为空
    """
    s = load_settings(ROOT / "config")
    agg = aggregate([
        {"model": "glm-5.3", "task": "code", "score": 9, "latency_s": 2, "cost_usd": 0},
        {"model": "glm-5.3", "task": "code", "score": None, "unscored": "请求失败",
         "error": "client_error: invalid temperature", "latency_s": 1, "cost_usd": 0},
        {"model": "glm-5.3", "task": "code", "score": None, "unscored": "正文为空", "latency_s": 1, "cost_usd": 0},
        {"model": "glm-5.3", "task": "code", "score": None, "unscored": "裁判无输出", "latency_s": 1, "cost_usd": 0},
    ])
    assert agg[("glm-5.3", "code")]["n"] == 1
    assert agg[("glm-5.3", "code")]["mean"] == 9.0  # 不是 9/4
    assert agg[("glm-5.3", "code")]["skipped"] == 3

    # 一个有效样本都没有 → 均分是 NaN，而且绝不改先验
    only_bad = aggregate([
        {"model": "kimi-k2.7-code", "task": "code", "score": None, "unscored": "请求失败",
         "latency_s": 1, "cost_usd": 0},
    ])
    a = only_bad[("kimi-k2.7-code", "code")]
    assert a["n"] == 0 and a["mean"] != a["mean"] and a["skipped"] == 1
    spec = s.models["kimi-k2.7-code"]
    sug = suggest_quality({"kimi-k2.7-code": spec}, only_bad, blend=0.5, min_n=1)
    assert sug["kimi-k2.7-code"]["code"] == spec.q("code")

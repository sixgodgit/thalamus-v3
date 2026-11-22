import math
import time

import pytest
from helpers import ROOT
from thalamus.capacity import Pressure
from thalamus.capacity import CapacityTracker
from thalamus.config import load_settings
from thalamus.features import extract
from thalamus.health import HealthBook, QualityLearner, SessionState
from thalamus.intent import Intent, classify
from thalamus.router import KeyView, Router

STD = KeyView("k", "standard", None, 0)
PRO = KeyView("k", "pro", None, 0)


def make(min_context: int = 0, exempt: frozenset[str] = frozenset()):
    s = load_settings(ROOT / "config")
    s.policy.min_context_tokens = min_context  # 见 helpers.load_test_settings 的说明
    s.policy.min_context_exempt_tasks = exempt  # 同上：豁免默认关掉，要测的用例自己打开
    cap = CapacityTracker(s)
    health = HealthBook(s.policy)
    r = Router(s, cap, health, QualityLearner(s.policy), {})
    return s, cap, health, r


def plan_for(r, body, key=STD, intent=None, **kw):
    f = extract(body, key.id)
    it = intent or classify(f, 120_000)
    return r.plan(f, it, key, **kw)


def test_vision_only_vision_models():
    _, _, _, r = make()
    p = plan_for(r, {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "这是什么"}, {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]})
    assert p.ranked and all(c.offer.model.vision for c in p.ranked)
    assert any(c.reject == "不支持图片" for c in p.rejected)


def test_pro_models_need_pro_plan():
    _, _, _, r = make()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert any(c.offer.model.key == "kimi-k3" and c.reject == "需要 pro 套餐" for c in plan_for(r, body).rejected)
    assert any(c.offer.model.key == "kimi-k3" for c in plan_for(r, body, PRO).ranked)


def test_region_whitelist_filters_candidates():
    _, _, _, r = make()
    key = KeyView("k", "standard", None, 0, {"cn"})
    p = plan_for(r, {"messages": [{"role": "user", "content": "写个快排"}]}, key)
    assert p.ranked and all(c.offer.channel.region == "cn" for c in p.ranked)
    assert any(c.reject == "key 不允许此区域(us)" for c in p.rejected)


def test_min_context_gate_excludes_short_context_offers():
    s, _, _, r = make(1_000_000)
    p = plan_for(r, {"messages": [{"role": "user", "content": "写个快排"}]})
    assert p.ranked and all(c.offer.model.context >= 1_000_000 for c in p.ranked)
    short = [o.key for o in s.usable_offers() if o.model.context < 1_000_000]
    assert short, "配置里已经没有低于 1M 的 offer，这个用例失去意义，请换一个构造"
    for key in short:
        c = next(c for c in p.rejected if c.offer.key == key)
        assert c.reject.startswith("原生上下文 ") and "< 要求 1M" in c.reject, c.reject


def test_min_context_exemption_gives_small_tasks_the_fast_tier():
    """"要多大窗口"是每个请求自己的事。闲聊输入只有几 K，对它要求 1M 等于白扔掉
    又快又便宜的候选 —— 而 1M 以内装得下的保证由 room 检查提供，不靠这道全局门槛。"""
    s, _, _, r = make(1_000_000, exempt=frozenset({"chat"}))
    short = {o.key for o in s.usable_offers() if o.model.context < 1_000_000}
    assert short, "配置里已经没有低于 1M 的 offer，这个用例失去意义，请换一个构造"

    chat = plan_for(r, {"messages": [{"role": "user", "content": "今天天气怎么样"}]})
    assert chat.intent.task == "chat"
    assert {c.offer.key for c in chat.ranked} & short, "豁免没生效"
    # 没被豁免的任务，门槛照旧
    code = plan_for(r, {"messages": [{"role": "user", "content": "用 python 写个 LRU 缓存，要带单元测试"}]})
    assert code.intent.task == "code"
    assert not {c.offer.key for c in code.ranked} & short
    for key in short:
        assert "原生上下文" in next(c for c in code.rejected if c.offer.key == key).reject


def test_min_context_exemption_never_applies_to_a_pinned_model():
    """对外承诺是"/v1/models 里每个能点的模型都有 1M"。豁免只给自动路由，
    否则点名就能用到没列出来的 offer，那条承诺就漏了。"""
    s, _, _, r = make(1_000_000, exempt=frozenset({"chat"}))
    for o in [o for o in s.usable_offers() if o.model.context < 1_000_000]:
        p = plan_for(r, {"messages": [{"role": "user", "content": "今天天气怎么样"}]},
                     pinned_model=o.model.key)
        assert not p.ranked, f"点名 {o.model.key} 绕过了 1M 门槛"
        assert "原生上下文" in next(c for c in p.rejected if c.offer.key == o.key).reject


def test_exempt_tasks_can_never_exceed_the_short_window():
    """豁免安全的前提：输入一大就不再是 chat/utility 了。
    est_tokens ≥ long_context_tokens(12 万) 时结构信号直接判 long，而 long 没被豁免 ——
    所以被豁免的任务实际上永远不会超过 12 万 token，低于最短窗口（minimax 20 万）。"""
    s = load_settings(ROOT / "config")
    assert s.policy.long_context_tokens <= min(
        o.model.context for o in s.usable_offers() if o.model.context < s.policy.min_context_tokens
    ), "有豁免 offer 的窗口比 long 阈值还小，豁免就不再安全了"
    assert not s.policy.min_context_exempt_tasks & {"long", "code", "agent", "reason", "vision"}
    f = extract({"messages": [{"role": "user", "content": "聊聊天气。" + "闲" * 200_000}]}, "k")
    assert classify(f, s.policy.long_context_tokens).task == "long"


def _price(cap, u: float, before: float | None = None, lam: float = 1.0) -> float:
    pr = Pressure(ratio=u, window_hours=5, exhausted=False,
                  before=u if before is None else before, scope="x", lam=lam)
    return cap.shadow_multiplier(u, 0.0, pr)


def test_shadow_price_curve_matches_theory_anchors_and_is_convex():
    """形状改成 exp(κ(u−full))：κ 由 floor/knee/full 反解，所以旧口径的两个锚点仍然成立。"""
    s, cap, _, _ = make()
    p = s.policy
    assert _price(cap, p.pacing_knee) == pytest.approx(p.pacing_floor, rel=0.02)
    assert _price(cap, p.pacing_full) == pytest.approx(1.0, rel=0.02)
    assert _price(cap, 0.0) == p.pacing_floor  # 闲置被地板托住
    assert _price(cap, 0.90) > 1.0 and _price(cap, 0.95) > _price(cap, 0.90)
    assert math.isinf(_price(cap, p.pacing_safety))
    # 凸性：一次吃掉 0.5→0.85 这段的平均价，必须高于起点的边际价、低于终点的边际价
    avg = _price(cap, 0.85, before=0.5)
    assert _price(cap, 0.5) < avg < _price(cap, 0.85), avg
    # λ 是线性缩放
    assert _price(cap, 0.8, lam=2.0) == pytest.approx(2 * _price(cap, 0.8, lam=1.0), rel=1e-6)


def test_dual_price_learns_from_consumption_rate():
    """对偶镜像下降：消耗快于窗口可持续速率就涨价，慢就降价，且有界、可关闭。"""
    s, cap, _, _ = make()
    scope = "glm_plan"
    limit, w = min(cap._windows[scope], key=lambda lw: lw[1].seconds)
    rate = limit / w.seconds  # 把窗口刚好用满的速率（单位/秒）
    t = 1_000_000.0

    for i in range(30):  # 以 50 倍可持续速率猛用
        cap.learn(scope, rate * 50 * 10, t + i * 10)
    hot = cap.lam(scope, w.seconds)
    assert hot > 1.3, hot

    for i in range(40):  # 之后长时间几乎不用
        cap.learn(scope, rate * 0.01 * 600, t + 100_000 + i * 600)
    assert cap.lam(scope, w.seconds) < hot

    for i in range(600):  # 上下界
        cap.learn(scope, rate * 1000 * 10, t + 500_000 + i * 10)
    assert cap.lam(scope, w.seconds) == pytest.approx(s.policy.pacing_lam_max)

    s.policy.pacing_adaptive = False
    frozen = cap.lam(scope, w.seconds)
    cap.learn(scope, rate * 1e6, t + 900_000)
    assert cap.lam(scope, w.seconds) == frozen


def test_latency_deadline_filters_then_degrades_gracefully():
    """延迟当约束用：超死线的先剔掉；全都超线时放回来按常规打分，不把请求判死。"""
    s, _, _, r = make()
    body = {"messages": [{"role": "user", "content": "今天天气怎么样"}]}
    s.policy.tasks["chat"].deadline_s = 6.0
    p = plan_for(r, body)
    assert p.ranked and all(c.latency_s <= 6.0 for c in p.ranked)
    assert any("超过死线" in c.reject for c in p.rejected)

    s.policy.tasks["chat"].deadline_s = 0.5  # 没有候选能满足
    p2 = plan_for(r, body)
    assert p2.ranked and not any("超过死线" in c.reject for c in p2.rejected)

    # thalamus/fast 会把死线再收紧 3 倍
    s.policy.tasks["chat"].deadline_s = 18.0
    f = extract(body, "k")
    loose = r.plan(f, classify(f, 120_000), STD)
    tight = r.plan(f, classify(f, 120_000, "chat", 1, 3.0), STD)
    assert len(tight.ranked) < len(loose.ranked)
    s.policy.tasks["chat"].deadline_s = 0.0


def test_single_huge_request_does_not_buy_the_whole_subscription_window():
    """一次 90 万 token 的请求不该以地板价吃掉 GLM 的 5 小时积分窗口。"""
    s, cap, _, r = make(1_000_000)
    body = {"messages": [{"role": "user", "content": "帮我把这个仓库整体重构一遍"}, {"role": "system", "content": "x" * 3_400_000}]}
    p = plan_for(r, body)
    assert p.ranked
    glm = next(c for c in p.ranked if c.offer.key == "glm-5.3@glm_plan")
    # 窗口是空的（累计压力≈单次占比），但单次占比已经超过阈值，所以不给地板价
    assert glm.pressure is not None and glm.pressure > s.policy.pacing_single_share
    assert glm.shadow > 1.0, (glm.pressure, glm.shadow)
    assert p.ranked[0].offer.metered or p.ranked[0].offer.channel.id == "opencode_go", p.reason()

    # 常规大小的请求不受影响，仍然是地板价
    small = plan_for(r, {"messages": [{"role": "user", "content": "帮我写个快排"}]})
    g2 = next(c for c in small.ranked if c.offer.key == "glm-5.3@glm_plan")
    assert g2.shadow == s.policy.pacing_floor, (g2.pressure, g2.shadow)
    assert cap.shadow_multiplier(0.1, 0.30) > cap.shadow_multiplier(0.1, 0.05)


def test_near_1m_input_keeps_room_for_output():
    _, _, _, r = make(1_000_000)
    # 约 90 万 token 的输入：1M 模型仍然可用，但预估输出被压到剩余空间以内
    p = plan_for(r, {"messages": [{"role": "user", "content": "x" * 3_400_000}]})
    assert p.ranked, [c.reject for c in p.rejected]
    top = p.ranked[0]
    assert 0 < top.est_out <= top.offer.model.context - top.est_miss - top.est_cached

    # 输入本身超过窗口：全部拒绝，且理由里带上实际估算值
    p2 = plan_for(r, {"messages": [{"role": "user", "content": "x" * 4_400_000}]})
    assert not p2.ranked
    ctx_rejects = [c.reject for c in p2.rejected if c.offer.model.context >= 1_000_000 and c.offer.model.tier == "standard"]
    assert ctx_rejects and all(r.startswith("上下文不足(1M，输入约 ") for r in ctx_rejects), ctx_rejects


def test_shipped_policy_requires_1m_context():
    """产品口径：只对接原生 1M 的模型，且三家上游在这道门槛下都还有可用 offer。"""
    s = load_settings(ROOT / "config")
    assert s.policy.min_context_tokens == 1_000_000
    eligible = s.eligible_offers()
    assert all(o.model.context >= 1_000_000 for o in eligible)
    chans = {o.channel.id for o in eligible}
    assert chans == {"deepseek", "glm_plan", "opencode_go"}, sorted(chans)
    # standard 档也必须每家都有，否则平价用户实际只有一家可用
    std = {o.channel.id for o in eligible if o.model.tier == "standard"}
    assert std == {"deepseek", "glm_plan", "opencode_go"}, sorted(std)


def test_default_region_whitelist_is_validated_at_startup():
    from thalamus.config import validate

    s = load_settings(ROOT / "config")
    s.policy.keys.allowed_regions = ["eu"]
    warns = validate(s)
    assert any("未知区域" in w for w in warns), warns
    assert any("没有任何可用渠道" in w for w in warns), warns

    s.policy.keys.allowed_regions = ["cn"]
    warns = validate(s)
    assert any("默认区域白名单生效" in w for w in warns), warns


def test_hard_agent_prefers_strong_model_when_subscriptions_idle():
    _, _, _, r = make()
    body = {
        "messages": [
            {"role": "system", "content": "x" * 120_000},
            {"role": "user", "content": "重构整个项目的鉴权模块"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "file content"},
        ],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }
    p = plan_for(r, body)
    assert p.intent.task == "agent" and p.intent.difficulty >= 2
    top = p.ranked[0].offer
    # 订阅额度闲置时，高价值的 agent 任务应该用订阅里够强的模型，而不是按量的便宜模型
    assert not top.metered, p.reason()
    assert top.model.q("agent") >= 8.0


def test_pacing_moves_easy_chat_to_metered_when_subscriptions_busy():
    s, cap, _, r = make()
    body = {"messages": [{"role": "user", "content": "给我讲个笑话"}]}
    now = time.time()
    for scope in cap.scopes():
        for limit, w in cap._windows[scope]:
            w.add(now - 60, limit * 0.93)
    p = plan_for(r, body)
    assert p.ranked[0].offer.metered, [c.as_dict() for c in p.ranked[:3]]


def test_exhausted_subscription_is_rejected():
    s, cap, _, r = make()
    go = next(o for o in s.offers if o.channel.id == "opencode_go")
    cap.mark_exhausted(go)
    p = plan_for(r, {"messages": [{"role": "user", "content": "hi"}]}, PRO)
    go_rejects = [c for c in p.rejected if c.offer.channel.id == "opencode_go"]
    assert go_rejects and all("订阅额度" in c.reject for c in go_rejects)


def test_session_cache_makes_same_offer_cheaper():
    s, _, _, r = make()
    body = {"messages": [{"role": "system", "content": "资料" * 30_000}, {"role": "user", "content": "python 代码怎么优化"}]}
    f = extract(body, "k")
    prev = SessionState("code", 2, "deepseek-v4.1-flash@deepseek", "deepseek-v4.1-flash", f.est_tokens - 50, "h", 1, time.time())
    p = r.plan(f, classify(f, 120_000), STD, prev=prev)
    sticky = next(c for c in p.ranked if c.offer.key == "deepseek-v4.1-flash@deepseek")
    other = next(c for c in p.ranked if c.offer.key == "deepseek-v4.1-flash@opencode_go")
    assert sticky.est_cached > 0 and other.est_cached == 0
    assert sticky.list_usd < other.list_usd


def test_economy_mode_limits_cost():
    s, _, _, r = make()
    body = {"messages": [{"role": "user", "content": "分析" * 20_000}]}
    p = plan_for(r, body, PRO, mode="economy")
    assert all(c.credits <= s.policy.keys.economy_max_credits for c in p.ranked)
    assert any(c.reject.startswith("经济档") for c in p.rejected)


def test_metered_cap_blocks_metered_channels():
    s, _, _, r = make()
    s.policy.daily_metered_usd_cap = 1.0
    p = plan_for(r, {"messages": [{"role": "user", "content": "hi"}]}, metered_spent_today=1.5)
    assert p.ranked and all(not c.offer.metered for c in p.ranked)


def test_breaker_open_offer_rejected():
    from thalamus.types import FailureClass

    s, _, health, r = make()
    for _ in range(s.policy.breaker_threshold):
        health.failure("deepseek-v4-pro@deepseek", FailureClass.PROVIDER_DOWN)
    p = plan_for(r, {"messages": [{"role": "user", "content": "hi"}]})
    assert any(c.offer.key == "deepseek-v4-pro@deepseek" and "熔断" in c.reject for c in p.rejected)


def test_regenerate_penalizes_previous_model():
    s, _, _, r = make()
    body = {"messages": [{"role": "user", "content": "写一首关于海的诗"}]}
    f = extract(body, "k")
    base = r.plan(f, classify(f, 120_000), STD)
    top = base.ranked[0].offer
    it = classify(f, 120_000)
    it.regenerate = True
    prev = SessionState("chat", 1, top.key, top.model.key, 10, f.last_user_hash, 1, time.time())
    again = r.plan(f, it, STD, prev=prev)
    assert again.ranked[0].offer.model.key != top.model.key

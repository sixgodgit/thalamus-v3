"""候选打分与排序。

每个候选（某渠道上的某模型）先过硬约束，再算一个统一的"净价值"（单位：美元）：

    score = 质量价值 − 有效成本 − 等待成本 − 风险成本

  质量价值  = 有效质量分 × 该任务每分值多少钱
             有效质量分：够用线以内全额计，超出部分只计 35%，不足部分按 150% 扣
  有效成本  = 按量渠道：实付美元（含峰时倍率，已算入会话缓存命中）
             订阅渠道：标价美元 × 影子价格倍率（额度越紧越贵）
  等待成本  = 预计耗时（首 token + 输出/速度）× 该任务每秒值多少钱
  风险成本  = 近期错误率 × 质量价值

所有项都可以在管理台"路由试算"里看到，模型选择完全可解释。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .capacity import CapacityTracker
from .config import Offer, Settings, fmt_ctx
from .features import Features
from .health import HealthBook, OutcomeBook, QualityLearner, SessionState
from .intent import Intent


@dataclass
class KeyView:
    id: str
    plan: str
    allowed_models: set[str] | None
    max_output_tokens: int
    allowed_regions: set[str] | None = None  # None = 不限区域


@dataclass
class Candidate:
    offer: Offer
    reject: str = ""
    score: float = -math.inf
    quality: float = 0.0
    need: float = 0.0
    value: float = 0.0
    cost: float = 0.0
    latency_s: float = 0.0
    latency_cost: float = 0.0
    risk_cost: float = 0.0
    bad_rate: float = 0.0  # 验证器判坏的比率（含先验收缩）
    est_miss: int = 0
    est_cached: int = 0
    est_out: int = 0
    list_usd: float = 0.0
    real_usd: float = 0.0
    units: float = 0.0
    credits: float = 0.0
    pressure: float | None = None
    shadow: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "offer": self.offer.key,
            "model": self.offer.model.key,
            "channel": self.offer.channel.id,
            "reject": self.reject,
            "score": None if self.reject else round(self.score, 6),
            "quality": round(self.quality, 2),
            "need": self.need,
            "value_usd": round(self.value, 6),
            "cost_usd": round(self.cost, 6),
            "latency_s": round(self.latency_s, 2),
            "latency_cost_usd": round(self.latency_cost, 6),
            "risk_cost_usd": round(self.risk_cost, 6),
            "bad_rate": round(self.bad_rate, 4),
            "est_tokens": {"miss": self.est_miss, "cached": self.est_cached, "out": self.est_out},
            "list_usd": round(self.list_usd, 6),
            "real_usd": round(self.real_usd, 6),
            "units": round(self.units, 3),
            "credits": round(self.credits, 3),
            "pressure": None if self.pressure is None or math.isinf(self.pressure) else round(self.pressure, 3),
            "shadow_multiplier": None if self.shadow is None or math.isinf(self.shadow) else round(self.shadow, 3),
        }


@dataclass
class RoutePlan:
    intent: Intent
    mode: str
    ranked: list[Candidate]
    rejected: list[Candidate] = field(default_factory=list)

    def reason(self) -> str:
        if not self.ranked:
            return "no candidate"
        c = self.ranked[0]
        return (
            f"{self.intent.label()} q{c.quality:.1f}/{c.need:g} "
            f"cost${c.cost:.5f} wait{c.latency_s:.1f}s"
            + (f" p{c.pressure:.0%}" if c.pressure is not None else "")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": {
                "task": self.intent.task,
                "difficulty": self.intent.difficulty,
                "confidence": round(self.intent.confidence, 2),
                "source": self.intent.source,
                "signals": self.intent.signals,
                "regenerate": self.intent.regenerate,
                "correction": self.intent.correction,
            },
            "mode": self.mode,
            "ranked": [c.as_dict() for c in self.ranked],
            "rejected": [c.as_dict() for c in self.rejected],
        }


COST_WEIGHT = {"normal": 1.0, "soft": 3.0, "economy": 10.0}


class Router:
    def __init__(
        self,
        settings: Settings,
        capacity: CapacityTracker,
        health: HealthBook,
        learner: QualityLearner,
        channel_load: dict[str, int],
        outcomes: OutcomeBook | None = None,
    ):
        self.settings = settings
        self.capacity = capacity
        self.health = health
        self.learner = learner
        self.channel_load = channel_load
        self.outcomes = outcomes or OutcomeBook()

    def plan(
        self,
        f: Features,
        intent: Intent,
        key: KeyView,
        *,
        mode: str = "normal",
        prev: SessionState | None = None,
        pinned_model: str | None = None,
        metered_spent_today: float = 0.0,
        exclude: set[str] | None = None,
        now: float | None = None,
    ) -> RoutePlan:
        now = now or time.time()
        pol = self.settings.policy
        tp = pol.tasks[intent.task]
        d = intent.difficulty
        need = tp.at(tp.need, d)
        vpp = tp.at(tp.usd_per_point, d)
        cap = pol.daily_metered_usd_cap
        ranked: list[Candidate] = []
        rejected: list[Candidate] = []
        too_slow: list[Candidate] = []

        for offer in self.settings.offers:
            c = Candidate(offer=offer, need=need)
            m = offer.model
            reject = self._hard_filter(offer, f, key, pinned_model, exclude, metered_spent_today, now, intent.task)

            # 预估 token。room = 窗口里输入之外还剩多少，1M 上下文下这才是真正的约束。
            est_in = f.est_tokens
            room = m.context - est_in
            out = tp.at(tp.est_output, d)
            if f.max_tokens:
                out = min(out, f.max_tokens)
            if key.max_output_tokens > 0:
                out = min(out, key.max_output_tokens)
            if m.reasoning == "always":
                out += 300
            out = min(out, m.max_output)
            if room > 0:
                out = min(out, room)
            out = max(out, 0)
            cached = 0
            if prev is not None and prev.offer_key == offer.key:
                cached = min(prev.prompt_tokens, est_in)
            c.est_miss, c.est_cached, c.est_out = est_in - cached, cached, out

            c.list_usd = offer.list_usd(c.est_miss, cached, out)
            c.real_usd = offer.real_usd(c.est_miss, cached, out)
            c.units = offer.units(c.est_miss, cached, out)
            c.credits = c.list_usd / pol.credit_usd

            # 留给输出的空间必须够这个模型说完一句话（reasoning 模型的 min_max_tokens 更大）
            if not reject and room < max(256, offer.min_max_tokens):
                reject = f"上下文不足({fmt_ctx(m.context)}，输入约 {est_in})"

            # 成本
            if offer.metered:
                cost = c.real_usd
                if cap > 0 and metered_spent_today > 0.8 * cap:
                    cost *= 1 + 10 * (metered_spent_today / cap - 0.8)
            else:
                pr = self.capacity.pressure(offer, c.units, now)
                ratio = pr.ratio if pr else 0.0
                c.pressure = ratio
                c.shadow = self.capacity.shadow_multiplier(ratio, pr.share if pr else 0.0, pr)
                if not reject and (pr and pr.exhausted):
                    reject = "订阅额度已耗尽(上游返回)"
                elif not reject and math.isinf(c.shadow):
                    reject = f"订阅额度不足({pr.window_hours:g}h 窗口将达 {ratio:.0%})" if pr else "订阅额度不足"
                cost = c.list_usd * (0 if math.isinf(c.shadow) else c.shadow)
            cost *= COST_WEIGHT.get(mode, 1.0)

            if not reject and mode == "economy" and c.credits > pol.keys.economy_max_credits:
                reject = f"经济档单次上限 {pol.keys.economy_max_credits:g} credits"

            # 质量
            q = m.q(intent.task) + self.learner.delta(m.key, intent.task)
            c.quality = q
            q_eff = min(q, need) + 0.35 * max(0.0, q - need) - 1.5 * max(0.0, need - q)
            c.value = q_eff * vpp

            c.cost = cost
            c.latency_s = self.health.ttft(offer) + out / max(self.health.tps(offer), 1.0)
            c.latency_cost = tp.usd_per_sec * c.latency_s * intent.latency_boost
            # 风险成本有两项来源，语义不同：
            #   传输失败（上游 5xx/超时）→ 换家重试，损失≈这次的价值
            #   内容不合格（验证器判坏）→ 整轮白做且要重来，损失≈价值 + 再付一次成本
            c.risk_cost = self.health.error_rate(offer) * max(c.value, 0.0)
            c.bad_rate = self.outcomes.rate(m.key, intent.task)
            c.risk_cost += c.bad_rate * (max(c.value, 0.0) + cost + c.latency_cost)
            if intent.regenerate and prev is not None and prev.model_key == m.key:
                c.risk_cost += 1.0  # 用户在重新生成：排到其他可用模型之后，给一个不同的答案
            elif intent.correction and prev is not None and prev.model_key == m.key:
                c.risk_cost += 0.5 * max(c.value, 0.0)  # 用户在纠正：同一个模型打五折，更强/不同的模型更容易胜出

            # 延迟按"死线约束"处理而不是只按线性价格：等待的效用不是线性的
            # （300ms 和 500ms 用户无感，1s 和 3s 天壤之别），线性加权还只能取到
            # Pareto 前沿凸包上的点。死线把不可接受的候选先剔掉，剩下的再比价值。
            deadline = tp.deadline_s / max(intent.latency_boost, 1e-6) if tp.deadline_s > 0 else 0.0
            c.score = c.value - c.cost - c.latency_cost - c.risk_cost
            if reject:
                c.reject = reject
                rejected.append(c)
            elif deadline and c.latency_s > deadline:
                c.reject = f"预计 {c.latency_s:.0f}s 超过死线 {deadline:.0f}s"
                too_slow.append(c)
            else:
                ranked.append(c)

        # 死线是"有牙齿的偏好"，不是正确性要求：所有候选都超线时把它们放回来按常规打分
        # （等待成本本来就在公式里），而不是把请求判死。
        if not ranked and too_slow:
            for c in too_slow:
                c.reject = ""
            ranked, too_slow = too_slow, []
        rejected.extend(too_slow)
        ranked.sort(key=lambda x: x.score, reverse=True)
        return RoutePlan(intent=intent, mode=mode, ranked=ranked, rejected=rejected)

    def _hard_filter(
        self,
        offer: Offer,
        f: Features,
        key: KeyView,
        pinned_model: str | None,
        exclude: set[str] | None,
        metered_spent: float,
        now: float,
        task: str = "",
    ) -> str:
        m = offer.model
        if not offer.enabled or not offer.channel.enabled:
            return "已停用"
        if not offer.channel.api_key:
            return "未配置 API key"
        pol = self.settings.policy
        # 豁免只对自动路由生效。客户端点名时门槛照旧 —— 对外承诺是"/v1/models 里
        # 每个能点的模型都有 1M"，如果点名能用到没列出来的 offer，那条承诺就漏了。
        exempt = not pinned_model and task in pol.min_context_exempt_tasks
        need_ctx = 0 if exempt else pol.min_context_tokens
        if need_ctx and m.context < need_ctx:
            return f"原生上下文 {fmt_ctx(m.context)} < 要求 {fmt_ctx(need_ctx)}"
        if exclude and offer.key in exclude:
            return "本次已尝试"
        if pinned_model and m.key != pinned_model:
            return "非指定模型"
        if key.allowed_models is not None and m.key not in key.allowed_models:
            return "key 不允许此模型"
        if key.allowed_regions is not None and offer.channel.region not in key.allowed_regions:
            return f"key 不允许此区域({offer.channel.region})"
        if m.tier == "pro" and key.plan != "pro":
            return "需要 pro 套餐"
        if f.n_images and not m.vision:
            return "不支持图片"
        if f.has_tools and not m.tools:
            return "不支持工具调用"
        if f.json_mode and not m.json_mode:
            return "不支持结构化输出"
        # 这个客户端不回传 reasoning_content，而这个模型要求回传 —— 发过去就是 400。
        # 和"不支持图片""不支持工具调用"同一类：能力不匹配，属于硬约束而不是偏好。
        if f.reasoning_echo is False and m.needs_reasoning_echo:
            return "思考模型要求回传 reasoning_content，客户端没有回传"
        ok, why = self.health.available(offer.key, now)
        if not ok:
            return why
        if self.channel_load.get(offer.channel.id, 0) >= offer.channel.max_concurrency:
            return "渠道并发已满"
        cap = pol.daily_metered_usd_cap
        if offer.metered and cap > 0 and metered_spent >= cap:
            return "按量渠道今日预算已用完"
        return ""

"""订阅额度追踪与"影子价格"。

订阅渠道的边际成本是 0，但额度按滚动窗口限量（GLM：5 小时 / 每周积分；
OpenCode Go：5 小时 / 每周 / 每月）。窗口里没用掉的额度过期作废，
所以策略是：
  额度闲 → 按标价的 floor（默认 5%）计成本 → 几乎所有任务都愿意用
  额度紧 → 成本平滑升到标价甚至更高 → 只有高价值任务（难代码、agent）还会用
  预计超限 → 直接排除
这样不需要手写"先用订阅、再用按量"的规则，便宜量大自然成立。
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from .config import Offer, Policy, Settings


class RollingWindow:
    """按分钟分桶的滚动求和。add/used 均摊 O(1)。"""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.buckets: deque[list[float]] = deque()
        self.total = 0.0

    def add(self, ts: float, units: float) -> None:
        minute = int(ts // 60)
        if self.buckets and self.buckets[-1][0] == minute:
            self.buckets[-1][1] += units
        else:
            self.buckets.append([minute, units])
        self.total += units

    def used(self, now: float) -> float:
        cutoff = int((now - self.seconds) // 60)
        while self.buckets and self.buckets[0][0] < cutoff:
            self.total -= self.buckets.popleft()[1]
        if not self.buckets:
            self.total = 0.0
        return max(self.total, 0.0)


@dataclass
class Pressure:
    ratio: float  # 最紧窗口的 (已用 + 在途 + 本次) / 上限
    window_hours: float
    exhausted: bool
    share: float = 0.0  # 仅"本次请求"占最紧窗口的比例（1M 上下文下单次就可能很大）
    before: float = 0.0  # 本次请求之前的利用率，用于按区间积分而不是取一点的边际价
    scope: str = ""
    lam: float = 1.0  # 该窗口学到的对偶价格尺度


class CapacityTracker:
    def __init__(self, settings: Settings):
        self.policy: Policy = settings.policy
        self._windows: dict[str, list[tuple[float, RollingWindow]]] = {}
        self._inflight: dict[str, float] = {}
        self._exhausted_until: dict[str, float] = {}
        self._lam: dict[tuple[str, float], float] = {}  # (scope, 窗口秒数) → 对偶价格尺度
        self._last_commit: dict[str, float] = {}
        self.configure(settings)

    # ── 对偶价格（在线学习出来的尺度）────────────────────────────
    #
    # 影子价格 = 尺度 λ × 形状 exp(κ(u − full))：
    #   形状来自理论。在线预算分配的最优折价函数是指数型（AdWords 的 1−1/e 结果），
    #   而不是手画的分段二次。κ 由 floor/knee/full 反解出来，所以配置口径不变：
    #   κ = ln(1/floor) / (full − knee) 使得 u=knee 时正好等于 floor、u=full 时正好等于 1。
    #   λ 来自对偶镜像下降（指数化梯度）：观测到的消耗速率高于窗口可持续速率就涨价，
    #   低于就降价。这样 knee/full 这些"猜的"参数只决定形状，真正的价格水平由流量自己定。

    def lam(self, scope: str, seconds: float) -> float:
        return self._lam.get((scope, seconds), 1.0)

    def _kappa(self) -> float:
        p = self.policy
        span = max(p.pacing_full - p.pacing_knee, 1e-6)
        return math.log(1.0 / max(p.pacing_floor, 1e-9)) / span

    def learn(self, scope: str, units: float, now: float | None = None) -> None:
        """按实际消耗更新该 scope 各窗口的对偶价格（每次 commit 调一次）。

        λ ← clip(λ · exp(η · tanh(观测速率/可持续速率 − 1)))
        tanh 是为了限幅：一个超大请求不该把价格一步打到上限。
        """
        p = self.policy
        if not p.pacing_adaptive:
            return
        now = now or time.time()
        dt = max(now - self._last_commit.get(scope, now - 60.0), 1e-3)
        self._last_commit[scope] = now
        for limit, w in self._windows.get(scope, []):
            if limit <= 0:
                continue
            sustainable = limit / w.seconds          # 单位/秒，把窗口刚好用满的速率
            observed = units / dt
            err = math.tanh(observed / sustainable - 1.0)
            key = (scope, w.seconds)
            self._lam[key] = min(
                p.pacing_lam_max,
                max(p.pacing_lam_min, self._lam.get(key, 1.0) * math.exp(p.pacing_eta * err)),
            )

    def configure(self, settings: Settings) -> None:
        """(重)建窗口。已有且时长未变的窗口保留计数。"""
        self.policy = settings.policy
        fresh: dict[str, list[tuple[float, RollingWindow]]] = {}
        for o in settings.offers:
            b = o.channel.budget
            if b is None:
                continue
            scope = o.budget_scope
            if scope in fresh:
                continue
            old = {w.seconds: w for _, w in self._windows.get(scope, [])}
            fresh[scope] = [
                (bw.limit, old.get(bw.hours * 3600) or RollingWindow(bw.hours * 3600)) for bw in b.windows
            ]
        self._windows = fresh

    def scopes(self) -> list[str]:
        return list(self._windows)

    def max_seconds(self) -> float:
        return max((w.seconds for ws in self._windows.values() for _, w in ws), default=0.0)

    # ── 记录 ─────────────────────────────────────────────────

    def load(self, scope: str, ts: float, units: float) -> None:
        for _, w in self._windows.get(scope, []):
            w.add(ts, units)

    def commit(self, offer: Offer, units: float, ts: float | None = None) -> None:
        if units > 0:
            now = ts or time.time()
            self.load(offer.budget_scope, now, units)
            self.learn(offer.budget_scope, units, now)

    def hold(self, offer: Offer, units: float) -> None:
        if offer.channel.budget is not None:
            s = offer.budget_scope
            self._inflight[s] = self._inflight.get(s, 0.0) + units

    def unhold(self, offer: Offer, units: float) -> None:
        if offer.channel.budget is not None:
            s = offer.budget_scope
            self._inflight[s] = max(0.0, self._inflight.get(s, 0.0) - units)

    def mark_exhausted(self, offer: Offer, seconds: float | None = None) -> None:
        b = offer.channel.budget
        if b is None:
            return
        self._exhausted_until[offer.budget_scope] = time.time() + (seconds or b.exhausted_cooldown_s)

    # ── 查询 ─────────────────────────────────────────────────

    def pressure(self, offer: Offer, units: float, now: float | None = None) -> Pressure | None:
        if offer.channel.budget is None:
            return None
        now = now or time.time()
        scope = offer.budget_scope
        if self._exhausted_until.get(scope, 0) > now:
            return Pressure(math.inf, 0, True)
        inflight = self._inflight.get(scope, 0.0)
        worst = Pressure(ratio=0.0, window_hours=0, exhausted=False, scope=scope)
        for limit, w in self._windows.get(scope, []):
            if limit <= 0:
                return Pressure(ratio=math.inf, window_hours=w.seconds / 3600, exhausted=False,
                                share=math.inf, scope=scope)
            before = (w.used(now) + inflight) / limit
            r = before + units / limit
            # 多个窗口是嵌套的（5 小时 ⊂ 每周 ⊂ 每月），同一批 token 只该被算一次，
            # 所以取最紧的那个，而不是把各窗口的价格相加。
            if r > worst.ratio:
                worst = Pressure(
                    ratio=r, window_hours=w.seconds / 3600, exhausted=False,
                    share=units / limit, before=before, scope=scope, lam=self.lam(scope, w.seconds),
                )
        return worst

    def shadow_multiplier(self, ratio: float, share: float = 0.0, pr: Pressure | None = None) -> float:
        """标价的倍率 = λ × 指数形状，按本次消耗的区间取平均而不是取一点的边际价。

        三件事叠在一起：
        ① 形状：exp(κ(u − full))，κ 由 floor/knee/full 反解。在线预算分配的最优折价
           函数是指数型的（AdWords 的 1−1/e 结果），凸性本身就让"把窗口推得更满"
           的候选更贵。
        ② 区间平均：一次消耗 Δu 的真实成本是 ∫p(u)du，不是 p(u₀)·Δu。曲线是凸的，
           所以大请求自动比同一利用率下的小请求贵 —— 这是数学上正确的做法。
        ③ λ：对偶镜像下降学出来的价格水平（见 learn()）。
        share/single_share 那一项保留：理论在低利用率区把闲置额度定价为≈0，但产品上
        "一次请求不该买走整个窗口"是个明确取舍，而 λ 只在事后才涨得起来。
        """
        p = self.policy
        if ratio >= p.pacing_safety:
            return math.inf
        k = self._kappa()
        lam = pr.lam if pr else 1.0
        u1 = min(ratio, p.pacing_safety)
        u0 = max(0.0, min(pr.before if pr else ratio, u1))
        du = u1 - u0
        if du > 1e-9:  # 区间平均：(1/Δu)∫exp(κ(u−full))du
            m = lam * (math.exp(k * (u1 - p.pacing_full)) - math.exp(k * (u0 - p.pacing_full))) / (k * du)
        else:
            m = lam * math.exp(k * (u1 - p.pacing_full))
        m = max(m, p.pacing_floor)
        if share > 0 and p.pacing_single_share > 0:
            m = max(m, (share / p.pacing_single_share) ** 2)
        return m

    def _price_at(self, scope: str, w: RollingWindow, limit: float, now: float) -> float:
        """当前利用率下的边际价格，给管理台看。"""
        u = w.used(now) / limit if limit else 0.0
        pr = Pressure(ratio=u, window_hours=w.seconds / 3600, exhausted=False, before=u,
                      scope=scope, lam=self.lam(scope, w.seconds))
        m = self.shadow_multiplier(u, 0.0, pr)
        return 0.0 if math.isinf(m) else m

    def snapshot(self, now: float | None = None) -> dict[str, dict]:
        now = now or time.time()
        out: dict[str, dict] = {}
        for scope, ws in self._windows.items():
            out[scope] = {
                "inflight": round(self._inflight.get(scope, 0.0), 3),
                "exhausted_until": self._exhausted_until.get(scope, 0) if self._exhausted_until.get(scope, 0) > now else 0,
                "windows": [
                    {
                        "hours": w.seconds / 3600,
                        "limit": limit,
                        "used": round(w.used(now), 3),
                        "ratio": round(w.used(now) / limit, 4) if limit else 0,
                        "lam": round(self.lam(scope, w.seconds), 3),
                        "price_now": round(self._price_at(scope, w, limit, now), 4),
                    }
                    for limit, w in ws
                ],
            }
        return out

"""上游健康：熔断器 + 延迟/错误率滑动统计 + 会话记忆 + 隐式质量反馈。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Offer, Policy
from .types import FailureClass

EWMA = 0.2


@dataclass
class OfferHealth:
    state: str = "closed"  # closed | open | half_open
    fails: int = 0
    opens: int = 0
    open_until: float = 0.0
    probing: bool = False
    ttft: float | None = None
    tps: float | None = None
    err: float = 0.0
    ok: int = 0
    failed: int = 0
    last_error: str = ""
    last_error_at: float = 0.0


class HealthBook:
    def __init__(self, policy: Policy):
        self.policy = policy
        self._h: dict[str, OfferHealth] = {}

    def get(self, key: str) -> OfferHealth:
        h = self._h.get(key)
        if h is None:
            h = self._h[key] = OfferHealth()
        return h

    def available(self, key: str, now: float | None = None) -> tuple[bool, str]:
        h = self.get(key)
        now = now or time.time()
        if h.state == "closed":
            return True, ""
        if h.state == "open":
            if now < h.open_until:
                return False, f"熔断中({int(h.open_until - now)}s)"
            h.state, h.probing = "half_open", False
        if h.probing:
            return False, "半开探测中"
        return True, "half_open"

    def begin(self, key: str) -> None:
        h = self.get(key)
        if h.state == "half_open":
            h.probing = True

    def success(self, key: str, ttft_s: float | None, tps: float | None) -> None:
        h = self.get(key)
        h.ok += 1
        h.fails = 0
        h.err *= 1 - EWMA
        if h.state != "closed":
            h.state, h.opens, h.probing = "closed", 0, False
        if ttft_s is not None and ttft_s > 0:
            h.ttft = ttft_s if h.ttft is None else h.ttft * (1 - EWMA) + ttft_s * EWMA
        if tps is not None and tps > 0:
            h.tps = tps if h.tps is None else h.tps * (1 - EWMA) + tps * EWMA

    def abort(self, key: str) -> None:
        """请求被客户端取消：不算成功也不算失败，只释放半开探测位。"""
        self.get(key).probing = False

    def failure(self, key: str, failure: FailureClass, message: str = "", retry_after: float | None = None) -> None:
        h = self.get(key)
        now = time.time()
        h.last_error, h.last_error_at = f"{failure.value}: {message[:160]}", now
        # 这三类都不是"上游有毛病"：请求本身的问题、装不下、这个模型吃不下这种请求。
        # 算进错误率会污染路由的风险成本，连续三次还会把一个健康的模型熔断掉。
        if failure in (FailureClass.CLIENT_ERROR, FailureClass.CONTEXT_OVERFLOW,
                       FailureClass.MODEL_INCOMPATIBLE):
            return
        h.failed += 1
        h.err = h.err * (1 - EWMA) + EWMA
        h.probing = False
        if failure is FailureClass.RATE_LIMITED:
            self._open(h, retry_after or 15.0, count=False)
            return
        if failure in (FailureClass.AUTH_FAILED, FailureClass.QUOTA_EXHAUSTED):
            self._open(h, retry_after or self.policy.breaker_max_cooldown_s, count=False)
            return
        h.fails += 1
        if h.state == "half_open" or h.fails >= self.policy.breaker_threshold:
            h.opens += 1
            cooldown = min(
                self.policy.breaker_base_cooldown_s * 2 ** (h.opens - 1),
                self.policy.breaker_max_cooldown_s,
            )
            self._open(h, cooldown, count=True)

    @staticmethod
    def _open(h: OfferHealth, seconds: float, count: bool) -> None:
        h.state = "open"
        h.open_until = max(h.open_until, time.time() + seconds)
        h.probing = False
        if not count:
            h.fails = 0

    def seed(self, rows: list[dict]) -> None:
        """用库里的实测延迟做冷启动（app 启动时调一次）。

        不这么做的话每次重启都退回 upstreams.yaml 里的 ttft_s/tps 先验，而那些先验
        ① 是猜的（glm-5.3 写着 tps 50，实测一次回答 8.5-36.7s）、② 跟模型走、不区分渠道，
        分辨不出"荷兰 → 北京"和"荷兰 → 美国"差的那两百毫秒。路由的等待成本和死线
        全靠这两个数，所以重启就把这部分知识丢掉代价不小。
        """
        for r in rows:
            model, channel = r.get("model"), r.get("channel")
            if not model or not channel:
                continue
            h = self.get(f"{model}@{channel}")
            ttft_ms = float(r.get("ttft_ms") or 0)
            if ttft_ms > 0:
                h.ttft = ttft_ms / 1000.0
            gen_ms, out = float(r.get("gen_ms") or 0), float(r.get("out_tokens") or 0)
            if gen_ms > 0 and out > 0:
                h.tps = out / (gen_ms / 1000.0)

    def ttft(self, offer: Offer) -> float:
        h = self._h.get(offer.key)
        return h.ttft if h and h.ttft else offer.model.ttft_s

    def tps(self, offer: Offer) -> float:
        h = self._h.get(offer.key)
        return h.tps if h and h.tps else offer.model.tps

    def error_rate(self, offer: Offer) -> float:
        h = self._h.get(offer.key)
        return h.err if h else 0.0

    def snapshot(self) -> dict[str, dict]:
        now = time.time()
        return {
            k: {
                "state": h.state if not (h.state == "open" and now >= h.open_until) else "half_open",
                "open_for_s": max(0, int(h.open_until - now)) if h.state == "open" else 0,
                "ttft_s": round(h.ttft, 3) if h.ttft else None,
                "tps": round(h.tps, 1) if h.tps else None,
                "error_rate": round(h.err, 3),
                "ok": h.ok,
                "failed": h.failed,
                "last_error": h.last_error,
                "last_error_at": h.last_error_at,
            }
            for k, h in self._h.items()
        }


# ───────────────────────────── 会话记忆 ─────────────────────────────


@dataclass
class SessionState:
    task: str
    difficulty: int
    offer_key: str
    model_key: str
    prompt_tokens: int
    last_user_hash: str
    n_messages: int
    ts: float
    request_id: str = ""


class SessionMemory:
    """只存哈希与路由元数据，不存内容。用于：缓存粘性、追问沿用任务、重新生成检测。"""

    def __init__(self, ttl_s: float, max_items: int = 200_000):
        self.ttl = ttl_s
        self.max_items = max_items
        self._m: dict[str, SessionState] = {}

    def get(self, key: str, now: float | None = None) -> SessionState | None:
        s = self._m.get(key)
        if s is None:
            return None
        if (now or time.time()) - s.ts > self.ttl:
            self._m.pop(key, None)
            return None
        return s

    def put(self, key: str, state: SessionState) -> None:
        if len(self._m) >= self.max_items and key not in self._m:
            self.sweep(force=True)
        self._m[key] = state

    def sweep(self, force: bool = False) -> int:
        now = time.time()
        dead = [k for k, s in self._m.items() if now - s.ts > self.ttl]
        for k in dead:
            self._m.pop(k, None)
        if force and len(self._m) >= self.max_items:
            for k in sorted(self._m, key=lambda k: self._m[k].ts)[: self.max_items // 10]:
                self._m.pop(k, None)
        return len(dead)

    def __len__(self) -> int:
        return len(self._m)


# ───────────────────────────── 隐式质量反馈 ─────────────────────────────


class OutcomeBook:
    """每（模型，任务）的"回答被验证器判坏"的比率，用来给"答错的代价"定价。

    用带时间衰减的计数 + Beta 先验收缩：样本少时贴近先验（不扰动路由），样本多了
    才开始分化。这比 EWMA 更适合这里 —— 冷启动阶段不会因为一次失败就把某个模型
    判死，而衰减保证几周前的旧表现不会一直压着它。
    """

    def __init__(self, prior_rate: float = 0.03, prior_n: float = 20.0, decay: float = 0.995):
        self.prior_rate = prior_rate
        self.prior_n = prior_n
        self.decay = decay
        self._n: dict[tuple[str, str], float] = {}
        self._bad: dict[tuple[str, str], float] = {}

    def observe(self, model: str, task: str, bad: bool) -> None:
        k = (model, task)
        self._n[k] = self._n.get(k, 0.0) * self.decay + 1.0
        self._bad[k] = self._bad.get(k, 0.0) * self.decay + (1.0 if bad else 0.0)

    def rate(self, model: str, task: str) -> float:
        k = (model, task)
        n, bad = self._n.get(k, 0.0), self._bad.get(k, 0.0)
        return (bad + self.prior_rate * self.prior_n) / (n + self.prior_n)

    def seed(self, rows: list[dict]) -> None:
        """用库里已有的统计做冷启动（app 启动时调一次）。"""
        from .outcome import BAD_FLAGS

        for r in rows:
            model, task, n = r.get("model"), r.get("task"), float(r.get("n") or 0)
            if not model or not task or n <= 0:
                continue
            bad = sum(float(r.get(f) or 0) for f in BAD_FLAGS)
            k = (model, task)
            self._n[k] = self._n.get(k, 0.0) + n
            self._bad[k] = self._bad.get(k, 0.0) + min(bad, n)

    def snapshot(self) -> list[dict]:
        return [
            {"model": m, "task": t, "n": round(self._n.get((m, t), 0.0), 1),
             "bad": round(self._bad.get((m, t), 0.0), 1), "rate": round(self.rate(m, t), 4)}
            for (m, t) in sorted(self._n)
        ]


class QualityLearner:
    """同一会话短时间内重新生成同一问题 → 上次模型在这类任务上扣分。

    有界、缓慢、可审计：单次 -penalty，成功请求缓慢回升，总幅度被 max_down/max_up 限制。
    """

    def __init__(self, policy: Policy):
        self.policy = policy
        self._d: dict[tuple[str, str], float] = {}
        self._dirty: set[tuple[str, str]] = set()

    def delta(self, model: str, task: str) -> float:
        return self._d.get((model, task), 0.0) if self.policy.learning_enabled else 0.0

    def penalize(self, model: str, task: str, weight: float = 1.0) -> None:
        k = (model, task)
        step = self.policy.learn_penalty * weight
        self._d[k] = max(-self.policy.learn_max_down, self._d.get(k, 0.0) - step)
        self._dirty.add(k)

    def reward(self, model: str, task: str) -> None:
        k = (model, task)
        cur = self._d.get(k, 0.0)
        new = min(self.policy.learn_max_up, cur + self.policy.learn_penalty * 0.02)
        if abs(new - cur) > 1e-9:
            self._d[k] = new
            self._dirty.add(k)

    def load(self, rows: list[tuple[str, str, float]]) -> None:
        for model, task, delta in rows:
            self._d[(model, task)] = float(delta)

    def drain_dirty(self) -> list[tuple[str, str, float]]:
        rows = [(m, t, self._d[(m, t)]) for (m, t) in self._dirty]
        self._dirty.clear()
        return rows

    def snapshot(self) -> list[dict]:
        return [
            {"model": m, "task": t, "delta": round(d, 3)}
            for (m, t), d in sorted(self._d.items())
            if abs(d) > 1e-6
        ]

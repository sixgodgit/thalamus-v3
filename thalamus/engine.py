"""请求编排。

  鉴权 → RPM/并发 → 特征与意图（含会话沿用、重新生成与纠正识别）→ 当日配额状态
  → 打分路由 → 原子预留 → 尝试候选直到拿到首个有效输出
        · 首 token 前失败可无感换家
        · 等待敏感的任务在首 token 迟迟不来时对冲到另一个渠道，谁先出字用谁
  → 流式转发（空闲超时、额度实时截断）→ 结果信号检查 → 结算（恰好一次）

响应头在拿到首 token 之后才发出，所以换家/对冲后头里的模型名依然准确，
首 token 前的失败也能返回正确的 HTTP 状态码，而不是 200 里夹一个错误。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from typing import Any, AsyncIterator

from . import outcome as oc
from .adapters import CallContext, ConnectionPool, make_adapter
from .adapters.base import classify_exception
from .capacity import CapacityTracker
from .config import Offer, Settings
from .features import Features, extract
from .health import HealthBook, OutcomeBook, QualityLearner, SessionMemory, SessionState
from .intent import ALIAS_NAMES, Intent, classify, inherit, is_correction, parse_alias
from .metrics import LATENCY_BUCKETS, OVERHEAD_BUCKETS, Metrics
from .persist import WriteBehind
from .quota import Ledger, Limiter
from .router import Candidate, KeyView, RoutePlan, Router
from .store import ApiKey, Store, hash_key
from .types import CHANNEL_WIDE, Delta, FailureClass, GatewayError, UpstreamError, Usage

log = logging.getLogger("thalamus.engine")

KEY_CACHE_S = 10.0


async def _aclose(agen: Any) -> None:
    try:
        await agen.aclose()
    except BaseException:  # noqa: BLE001 - 关闭上游流时的任何异常都无所谓
        pass


class Engine:
    def __init__(self, settings: Settings, store: Store, pool: ConnectionPool):
        self.settings = settings
        self.store = store
        self.pool = pool
        self.capacity = CapacityTracker(settings)
        self.health = HealthBook(settings.policy)
        self.learner = QualityLearner(settings.policy)
        self.outcomes = OutcomeBook()
        self.sessions = SessionMemory(settings.policy.session_ttl_s)
        self.channel_load: dict[str, int] = {}
        self.router = Router(
            settings, self.capacity, self.health, self.learner, self.channel_load, self.outcomes
        )
        self.ledger = Ledger(store, settings)
        self.writer = WriteBehind(store.path, self.ledger)
        self.limiter = Limiter()
        self.metrics = Metrics()
        self.started_at = time.time()
        self._bg: set[asyncio.Task] = set()
        self._calls: set[Call] = set()
        self._key_cache: dict[str, tuple[ApiKey, float]] = {}
        self._key_loading: dict[str, asyncio.Future] = {}
        self._key_miss: dict[str, float] = {}  # 无效 key 的负缓存：别让随机 token 每次都查库
        self._auth_fails: dict[str, deque[float]] = {}
        self._metered_day = ""
        self._metered: dict[str, float] = {}

    # ── 生命周期 ───────────────────────────────────────────────

    async def start(self) -> None:
        await self._load_capacity(self.capacity)
        self.learner.load(await self.store.load_learned())
        # 用库里已有的结果统计给"答错的比率"做冷启动，重启不至于从零开始
        self.outcomes.seed(await self.store.outcome_stats(time.time() - 14 * 86400))
        # 实测的首 token / 出字速度同理：等待成本与死线全靠这两个数，不能每次重启都
        # 退回 upstreams.yaml 里那份跟模型走、不区分渠道的乐观先验
        self.health.seed(await self.store.latency_stats(time.time() - 3 * 86400))
        await self._refresh_metered()
        await self.writer.start()

    async def stop(self) -> None:
        for c in list(self._calls):
            c.status = "shutdown"
            c.finalize()
        await self.drain(timeout=5)
        await self.flush()
        await self.writer.stop()

    async def flush(self) -> None:
        rows = self.learner.drain_dirty()
        if rows:
            await self.store.save_learned(rows)

    async def _load_capacity(self, tracker: CapacityTracker) -> None:
        horizon = tracker.max_seconds()
        if horizon:
            for scope, ts, units in await self.store.load_channel_usage(time.time() - horizon):
                tracker.load(scope, ts, units)

    async def apply_settings(self, s: Settings) -> None:
        """热重载：重建容量窗口（从库里回放），其余组件换引用。"""
        tracker = CapacityTracker(s)
        await self._load_capacity(tracker)
        tracker._inflight = dict(self.capacity._inflight)
        tracker._exhausted_until = dict(self.capacity._exhausted_until)
        self.capacity = tracker
        self.settings = s
        self.router.settings = s
        self.router.capacity = tracker
        self.health.policy = s.policy
        self.learner.policy = s.policy
        self.sessions.ttl = s.policy.session_ttl_s
        self.ledger.settings = s
        self._key_cache.clear()

    def spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._bg.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("background task failed", exc_info=task.exception())

    async def drain(self, timeout: float | None = None) -> None:
        """等待后台任务结束并把写缓冲刷进数据库（测试与管理台读取前调用）。"""
        deadline = time.monotonic() + (timeout or 1e9)
        while self._bg and time.monotonic() < deadline:
            await asyncio.wait(set(self._bg), timeout=max(0.01, deadline - time.monotonic()))
        await self.writer.flush()

    def reap_calls(self) -> int:
        """兜底：长时间没有任何动静的调用强制结算，防止额度/并发位泄漏。"""
        limit = self.settings.policy.stream_idle_timeout_s + 60
        now = time.monotonic()
        n = 0
        for c in list(self._calls):
            if not c.finalized and now - c.last_activity > limit:
                c.status = "abandoned"
                c.finalize()
                n += 1
        return n

    # ── 鉴权 / 账本辅助 ─────────────────────────────────────────

    def _note_auth_fail(self, client: str | None) -> None:
        """同一来源鉴权失败过多就直接拒，避免未鉴权流量打到数据库。"""
        if not client:
            return
        limit = self.settings.policy.auth_fail_per_min
        now = time.monotonic()
        q = self._auth_fails.setdefault(client, deque())
        while q and q[0] <= now - 60:
            q.popleft()
        q.append(now)
        if len(self._auth_fails) > 10_000:
            for k in [k for k, v in self._auth_fails.items() if not v or v[-1] <= now - 60]:
                self._auth_fails.pop(k, None)
        if len(q) > limit:
            raise GatewayError(429, "too_many_failed_auth", "too many failed authentications", {"Retry-After": "60"})

    def _check_auth_throttle(self, client: str | None) -> None:
        if not client:
            return
        q = self._auth_fails.get(client)
        if q and len([t for t in q if t > time.monotonic() - 60]) > self.settings.policy.auth_fail_per_min:
            raise GatewayError(429, "too_many_failed_auth", "too many failed authentications", {"Retry-After": "60"})

    async def authenticate(self, raw: str | None, client: str | None = None) -> ApiKey:
        self._check_auth_throttle(client)
        if not raw:
            self._note_auth_fail(client)
            raise GatewayError(401, "invalid_api_key", "missing API key")
        h = hash_key(raw)
        now = time.time()
        miss_at = self._key_miss.get(h)
        if miss_at and now - miss_at < KEY_CACHE_S:
            self._note_auth_fail(client)
            raise GatewayError(401, "invalid_api_key", "invalid API key")
        hit = self._key_cache.get(h)
        if hit and now - hit[1] < KEY_CACHE_S:
            key = hit[0]
        else:
            # 单飞：缓存过期瞬间涌入的并发请求只查一次库（压测里冷启动 50 并发曾各查一次，平均 62ms）
            fut = self._key_loading.get(h)
            if fut is None:
                fut = asyncio.get_running_loop().create_future()
                self._key_loading[h] = fut
                try:
                    found = await self.store.get_key_by_raw(raw)
                    fut.set_result(found)
                except BaseException as ex:
                    fut.set_exception(ex)
                    fut.exception()  # 标记为已读取，避免无人等待时告警
                    raise
                finally:
                    self._key_loading.pop(h, None)
                key = found
            else:
                key = await asyncio.shield(fut)
            if key is None:
                if len(self._key_miss) > 10_000:
                    self._key_miss.clear()
                self._key_miss[h] = now
                self._note_auth_fail(client)
                raise GatewayError(401, "invalid_api_key", "invalid API key")
            self._key_cache[h] = (key, now)
        if not key.enabled:
            raise GatewayError(403, "key_disabled", "API key is disabled")
        if key.expires_at and key.expires_at < now:
            raise GatewayError(403, "key_expired", "API key has expired")
        return key

    def invalidate_keys(self) -> None:
        self._key_cache.clear()
        self._key_miss.clear()

    async def _refresh_metered(self) -> None:
        self._metered_day = self.ledger.today()
        self._metered = await self.store.metered_for_day(self._metered_day)

    async def metered_spent(self) -> float:
        if self.ledger.today() != self._metered_day:
            await self._refresh_metered()
        return sum(self._metered.values())

    def metered_by_channel(self) -> dict[str, float]:
        return dict(self._metered)

    def note_metered(self, day: str, channel: str, usd: float) -> None:
        if day == self._metered_day:
            self._metered[channel] = self._metered.get(channel, 0.0) + usd

    @staticmethod
    def key_view(key: ApiKey) -> KeyView:
        return KeyView(
            id=key.id,
            plan=key.plan,
            allowed_models=set(key.allowed_models) if key.allowed_models else None,
            max_output_tokens=key.max_output_tokens,
            allowed_regions=set(key.allowed_regions) if key.allowed_regions else None,
        )

    def model_known(self, name: str) -> bool:
        if name in self.settings.models:
            return True
        low = name.strip().lower()
        if low.startswith("thalamus/"):
            return low.split("/", 1)[1].partition("-")[0] in ALIAS_NAMES
        return False

    def check_model_name(self, raw: Any) -> None:
        """点名未知模型默认报 404 而不是静默退回自动路由。

        静默退回对 OpenAI 客户端友好，但配置写错（比如把视觉模型写成一个不存在的名字）
        会被吞掉：请求照样 200，只是换了个模型答，排查起来很难。要 drop-in 兼容的话
        把 policy.yaml 的 global.unknown_model 设成 route。
        """
        if self.settings.policy.unknown_model != "reject":
            return
        if not isinstance(raw, str) or not raw.strip() or self.model_known(raw):
            return
        known = sorted(f"thalamus/{a}" for a in ALIAS_NAMES)[:4] + sorted(self.settings.models)[:4]
        raise GatewayError(
            404, "model_not_found",
            f"unknown model {raw.strip()[:80]!r}; GET /v1/models lists what this key can use, e.g. {', '.join(known)}",
        )

    # ── 理解请求 ───────────────────────────────────────────────

    def understand(
        self, body: dict[str, Any], key_id: str, session_header: str | None, task_header: str | None
    ) -> tuple[Features, Intent, SessionState | None, str | None]:
        pol = self.settings.policy
        f = extract(body, key_id, session_header)
        alias_task, alias_d, boost = parse_alias(body.get("model"), task_header)
        m = body.get("model")
        pinned = m if isinstance(m, str) and m in self.settings.models else None
        intent = classify(f, pol.long_context_tokens, alias_task, alias_d, boost)
        prev = self.sessions.get(f.session_key)
        if prev is None:
            return f, intent, prev, pinned

        intent = inherit(intent, prev.task, prev.difficulty)
        age = time.time() - prev.ts
        if prev.last_user_hash == f.last_user_hash and prev.n_messages == f.n_messages and age <= pol.regenerate_window_s:
            intent.regenerate = True
            intent.signals.append("regenerate")
        elif f.n_messages > prev.n_messages and age <= pol.correction_window_s and is_correction(f.last_user):
            # 上一轮没答好：沿用那一轮的任务，并提高一级难度，让更强的模型有机会接手
            intent.correction = True
            intent.signals.append("correction")
            if intent.source == "rule" and intent.task == "chat" and prev.task != "chat":
                intent.task, intent.source = prev.task, "session"
            intent.difficulty = min(3, max(intent.difficulty, prev.difficulty) + 1)
        return f, intent, prev, pinned

    async def preview(self, body: dict[str, Any], key: ApiKey | None) -> dict[str, Any]:
        """管理台路由试算：只打分不执行。"""
        kv = self.key_view(key) if key else KeyView("preview", "pro", None, 0)
        f, intent, prev, pinned = self.understand(body, kv.id, None, None)
        mode = "normal"
        if key is not None:
            mode = (await self.ledger.state(key)).mode
            mode = "economy" if mode == "reject" else mode
        plan = self.router.plan(
            f, intent, kv, mode=mode, prev=prev, pinned_model=pinned,
            metered_spent_today=await self.metered_spent(),
        )
        out = plan.as_dict()
        out["features"] = {
            "est_tokens": f.est_tokens,
            "n_images": f.n_images,
            "has_tools": f.has_tools,
            "n_tools": f.n_tools,
            "in_tool_loop": f.in_tool_loop,
            "loop_depth": f.loop_depth,
            "n_user_turns": f.n_user_turns,
            "json_mode": f.json_mode,
            "pinned_model": pinned,
        }
        pol = self.settings.policy
        out["hedge"] = pol.hedge_enabled and intent.task in pol.hedge_tasks
        return out

    # ── 执行 ───────────────────────────────────────────────────

    async def open(
        self,
        body: dict[str, Any],
        key: ApiKey,
        *,
        stream: bool,
        session_header: str | None = None,
        task_header: str | None = None,
    ) -> "Call":
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise GatewayError(400, "invalid_request_error", "messages must be a non-empty array")
        self.check_model_name(body.get("model"))
        self.limiter.check_rpm(key)
        await self.limiter.acquire(key)
        call = Call(self, key, body, stream)
        try:
            await call.prepare(session_header, task_header)
            await call.connect()
            await call.resolve()
        except BaseException as e:
            if isinstance(e, GatewayError):
                call.status, call.failure = "rejected", call.failure or e.code
            elif isinstance(e, asyncio.CancelledError):
                call.status = "client_closed"
            else:
                call.status = "error"
            call.finalize()
            raise
        return call


class Call:
    def __init__(self, engine: Engine, key: ApiKey, body: dict[str, Any], stream: bool):
        self.engine = engine
        self.key = key
        self.body = body
        self.stream = stream
        self.request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.t0 = time.perf_counter()
        self.t_dispatch: float | None = None
        self.last_activity = time.monotonic()
        self.f: Features | None = None
        self.intent: Intent | None = None
        self.prev: SessionState | None = None
        self.plan: RoutePlan | None = None
        self.mode = "normal"
        self.day = ""
        self.rid: str | None = None
        self.decision: Candidate | None = None
        self.attempts = 0
        self.hedged = False
        self.hedge_cost_usd = 0.0
        self.failures: list[str] = []
        self.status = "ok"
        self.failure: str | None = None
        self.usage = Usage()
        self.final_usage = Usage()
        self.out_bytes = 0
        self.budget: float | None = None  # 本次请求最多可用的 credits，流式中实时截断
        self.finish_reason: str | None = None
        self.ttft: float | None = None
        self.attempt_ttft: float | None = None
        self.credits = 0.0
        self.real_usd = 0.0
        self.units = 0.0
        self.estimated = False
        self.flags: list[str] = []
        self.verify_retried = False
        self.verify_flags: list[str] = []
        self._pending_error: UpstreamError | None = None
        self._replay: list[Delta] = []
        self.outcome = oc.OutcomeTracker(False)
        self.finalized = False
        self._agen: Any = None
        self._pending: list[Delta] = []
        self._open_attempts: set[str] = set()
        self._attempt_started: dict[str, float] = {}
        engine._calls.add(self)

    # ── 规划 ───────────────────────────────────────────────────

    async def prepare(self, session_header: str | None, task_header: str | None) -> None:
        e = self.engine
        self.f, self.intent, self.prev, pinned = e.understand(self.body, self.key.id, session_header, task_header)
        # 代码块语法检查只对 code/agent 开（其他任务里的代码块多是片段，误报率高），
        # 顺带也限制了为检查而缓存输出文本的内存开销。
        self.outcome = oc.OutcomeTracker(
            self.f.json_mode,
            tools=self.body.get("tools"),
            check_code=self.intent.task in ("code", "agent"),
        )
        if self.prev is not None:
            if self.intent.regenerate:
                self._feedback(self.prev, oc.REGENERATED, 1.0)
            elif self.intent.correction:
                self._feedback(self.prev, oc.USER_CORRECTION, oc.LEARN_WEIGHT[oc.USER_CORRECTION])
        self.day = e.ledger.today()
        quota = await e.ledger.state(self.key, self.day)
        if quota.mode == "reject":
            raise e.ledger.exhausted_error(self.key)
        spent = await e.metered_spent()
        kv = e.key_view(self.key)
        self.mode = quota.mode

        def make(mode: str) -> RoutePlan:
            return e.router.plan(
                self.f, self.intent, kv, mode=mode, prev=self.prev,
                pinned_model=pinned, metered_spent_today=spent,
            )

        plan = make(self.mode)
        rid = None
        if plan.ranked:
            rid = await e.ledger.reserve(self.key, self._reserve_amount(plan), self.mode, self.day)
        if rid is None and plan.ranked and self.mode != "economy" and self.key.exhaust_mode == "economy":
            eplan = make("economy")  # 剩余额度不够这次预估：降到经济档再试
            if eplan.ranked:
                rid = await e.ledger.reserve(self.key, self._reserve_amount(eplan), "economy", self.day)
                if rid is not None:
                    plan, self.mode = eplan, "economy"
        self.plan = plan
        if not plan.ranked:
            raise self._no_route(plan, pinned)
        if rid is None:
            raise e.ledger.exhausted_error(self.key)
        self.rid = rid
        # 预留只是预估；真实输出长度要到结束才知道。给本次请求一个硬预算（剩余额度按在途请求均分），
        # 流式过程中超出就截断，防止用超大 max_tokens / 并发把每日额度打穿。
        ceiling = e.ledger.ceiling(self.key, self.mode)
        remaining = ceiling - quota.used - quota.reserved
        share = max(1, e.limiter.inflight(self.key.id))
        self.budget = max(remaining / share, self._reserve_amount(plan))

    def _feedback(self, prev: SessionState, flag: str, weight: float) -> None:
        """把用户对上一轮的负面反馈记到上一轮的模型与那条请求记录上。"""
        e = self.engine
        e.learner.penalize(prev.model_key, prev.task, weight)
        e.metrics.inc("thalamus_outcome_flags_total", flag=flag, model=prev.model_key)
        if prev.request_id:
            e.writer.add_flag(prev.request_id, flag)

    @staticmethod
    def _reserve_amount(plan: RoutePlan) -> float:
        top = plan.ranked[:3]
        return max(c.credits for c in top) * 1.2 + 0.01

    def _no_route(self, plan: RoutePlan, pinned: str | None) -> GatewayError:
        ignore = {"非指定模型", "未配置 API key", "已停用"}
        # "原生上下文不达标"是配置层的排除（和未配置 key 同类），不该被当成客户端输入太长
        reasons = [c.reject for c in plan.rejected if c.reject not in ignore and not c.reject.startswith("原生上下文")]
        if not reasons:
            if pinned:
                return GatewayError(404, "model_not_available", f"model {pinned} is not available")
            return GatewayError(503, "no_upstream", "no upstream is configured", {"Retry-After": "30"})
        gate = ("需要 pro 套餐", "key 不允许此模型", "key 不允许此区域")
        if all(r in gate for r in reasons):
            return GatewayError(403, "model_not_allowed", "this key is not allowed to use the requested model")
        # 这把 key 本来就用不了的模型（套餐/白名单）不参与"能力不匹配"的判断，
        # 否则一个 pro 模型在场就会把"没有模型支持图片/结构化输出"这类可行动的错误盖成 503。
        rest = [r for r in reasons if r not in gate]
        if all(r.startswith("经济档") for r in rest):
            return self.engine.ledger.exhausted_error(self.key)
        for reason, message in (
            ("不支持图片", "no available model supports image input"),
            ("不支持工具调用", "no available model supports tool calling"),
            ("不支持结构化输出", "no available model supports response_format; drop it or pin a model that does"),
        ):
            if all(r == reason for r in rest):
                return GatewayError(400, "unsupported_input", message)
        if all(r.startswith("上下文不足") for r in rest):
            est = self.f.est_tokens if self.f else 0
            longest = max((o.model.context for o in self.engine.settings.eligible_offers()), default=0)
            return GatewayError(
                400,
                "context_length_exceeded",
                f"input is about {est} tokens; the largest available context is {longest}",
            )
        return GatewayError(
            503, "upstream_unavailable", "all upstream candidates are busy or unavailable, retry shortly",
            {"Retry-After": "10"},
        )

    # ── 连接（首 token 前可换家 / 对冲）───────────────────────────

    def _begin_attempt(self, c: Candidate) -> None:
        e = self.engine
        self._open_attempts.add(c.offer.key)
        e.channel_load[c.offer.channel.id] = e.channel_load.get(c.offer.channel.id, 0) + 1
        e.health.begin(c.offer.key)
        e.capacity.hold(c.offer, c.units)

    def _end_attempt(self, c: Candidate) -> None:
        if c.offer.key not in self._open_attempts:
            return  # 幂等：取消路径与结算路径可能都会调用
        self._open_attempts.discard(c.offer.key)
        e = self.engine
        e.channel_load[c.offer.channel.id] = max(0, e.channel_load.get(c.offer.channel.id, 0) - 1)
        e.capacity.unhold(c.offer, c.units)

    def _on_failure(self, c: Candidate, err: UpstreamError) -> None:
        e = self.engine
        offer = c.offer
        e.health.failure(offer.key, err.failure, err.message, err.retry_after)
        if err.failure in CHANNEL_WIDE:
            if err.failure is FailureClass.QUOTA_EXHAUSTED and offer.channel.budget is not None:
                e.capacity.mark_exhausted(offer, err.retry_after)
            else:
                for o in e.settings.offers:
                    if o.channel.id == offer.channel.id and o.key != offer.key:
                        e.health.failure(o.key, err.failure, err.message, err.retry_after)
        self.failures.append(f"{offer.key}:{err.failure.value}")
        log.warning("attempt failed %s (%s): %s", offer.key, err.failure.value, err.message[:200])

    def _charge_loser(self, c: Candidate) -> None:
        """对冲落败的请求已经送达上游：输入成本由网关承担（记入实付/订阅窗口），不计入用户额度。"""
        e = self.engine
        o = c.offer
        real = o.real_usd(c.est_miss, c.est_cached, 0)
        units = o.units(c.est_miss, c.est_cached, 0)
        now = time.time()
        if units:
            e.capacity.commit(o, units, now)
            e.writer.add_channel_usage(o.budget_scope, now, units)
        if real:
            e.note_metered(self.day, o.channel.id, real)
            e.writer.add_metered(self.day, o.channel.id, real)
            e.metrics.inc("thalamus_upstream_usd_total", real, channel=o.channel.id, kind="hedge_loser")
        self.hedge_cost_usd += real

    async def _first(self, agen: Any) -> list[Delta]:
        buf: list[Delta] = []
        async for d in agen:
            buf.append(d)
            if d.meaningful or d.finish_reason:
                return buf
        if buf:
            return buf
        raise UpstreamError(FailureClass.INVALID_RESPONSE, "empty response from upstream")

    async def _try(self, c: Candidate, timeout: float) -> tuple[Any, list[Delta], float]:
        e = self.engine
        assert self.f is not None
        self._begin_attempt(c)
        t = self._attempt_started.get(c.offer.key) or time.perf_counter()
        agen = None
        try:
            ctx = CallContext(
                self.f.session_key,
                self.request_id,
                self.key.max_output_tokens,
                context_room=max(0, c.offer.model.context - self.f.est_tokens),
            )
            agen = make_adapter(c.offer, e.pool).stream(self.body, ctx)
            first = await asyncio.wait_for(self._first(agen), timeout=timeout)
        except asyncio.TimeoutError:
            err = UpstreamError(FailureClass.FIRST_TOKEN_TIMEOUT, f"no first token within {timeout:g}s")
        except UpstreamError as ex:
            err = ex
        except Exception as ex:  # noqa: BLE001 - 适配器自身的 bug 也不应拖垮整个请求
            log.exception("adapter error on %s", c.offer.key)
            err = classify_exception(ex)
        except BaseException:
            # 被取消：对冲落败，或客户端断开
            self._end_attempt(c)
            e.health.abort(c.offer.key)
            if agen is not None:
                e.spawn(_aclose(agen))
            raise
        else:
            return agen, first, time.perf_counter() - t
        self._end_attempt(c)
        if agen is not None:
            e.spawn(_aclose(agen))
        self._on_failure(c, err)
        raise err

    async def _connect_to(self, c: Candidate) -> None:
        """只连一个指定候选（验证失败后的定向重试用）。"""
        pol = self.engine.settings.policy
        assert self.intent is not None and self.f is not None
        timeout = pol.tasks[self.intent.task].first_token_timeout_s + self.f.est_tokens / pol.prefill_tps
        self.attempts += 1
        self._attempt_started[c.offer.key] = time.perf_counter()
        agen, first, attempt_ttft = await self._try(c, timeout)
        self.decision, self._agen, self._pending = c, agen, first
        self.attempt_ttft = attempt_ttft
        self.last_activity = time.monotonic()

    async def _cancel(self, tasks: dict[asyncio.Task, Candidate], charge: bool) -> None:
        for t in tasks:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (t, c), r in zip(tasks.items(), results):
            if isinstance(r, tuple):  # 取消前恰好已经成功：关掉它
                self._end_attempt(c)
                self.engine.spawn(_aclose(r[0]))
            if charge and not isinstance(r, UpstreamError):
                self._charge_loser(c)

    async def connect(self) -> None:
        e = self.engine
        assert self.plan is not None and self.intent is not None and self.f is not None
        pol = e.settings.policy
        # 首 token 预算要随输入长度放大：1M token 的预填充本身就要几十秒，
        # 否则长上下文请求会在等待中被判为"上游超时"，然后在每个候选上重试一遍。
        timeout = pol.tasks[self.intent.task].first_token_timeout_s + self.f.est_tokens / pol.prefill_tps
        can_hedge = pol.hedge_enabled and self.intent.task in pol.hedge_tasks
        bad_channels: set[str] = set()
        bad_models: set[str] = set()  # 这个模型吃不下这种请求，换渠道也一样
        tried: set[str] = set()
        pending: dict[asyncio.Task, Candidate] = {}
        last: UpstreamError | None = None

        def next_candidate(exclude_channel: str | None = None, max_credits: float | None = None) -> Candidate | None:
            for c in self.plan.ranked:
                o = c.offer
                if o.key in tried or o.channel.id in bad_channels or o.channel.id == exclude_channel:
                    continue
                if o.model.key in bad_models:
                    continue
                if max_credits is not None and c.credits > max_credits:
                    continue
                ok, _ = e.health.available(o.key)
                if not ok or e.channel_load.get(o.channel.id, 0) >= o.channel.max_concurrency:
                    continue
                return c
            return None

        def start(c: Candidate) -> None:
            tried.add(c.offer.key)
            self.attempts += 1
            now = time.perf_counter()
            self._attempt_started[c.offer.key] = now  # 同步记录：任务可能还没被调度，对冲等待要从现在算
            if self.t_dispatch is None:
                self.t_dispatch = now
            pending[asyncio.ensure_future(self._try(c, timeout))] = c

        try:
            while True:
                if not pending:
                    if self.attempts >= pol.max_attempts:
                        break
                    c = next_candidate()
                    if c is None:
                        break
                    start(c)

                wait_s = None
                if can_hedge and len(pending) == 1 and not self.hedged and self.attempts < pol.max_attempts:
                    only = next(iter(pending.values()))
                    delay = max(pol.hedge_min_delay_s, e.health.ttft(only.offer) * pol.hedge_ttft_factor)
                    wait_s = max(0.0, delay - (time.perf_counter() - self._attempt_started[only.offer.key]))

                done, _ = await asyncio.wait(set(pending), timeout=wait_s, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    only = next(iter(pending.values()))
                    c2 = next_candidate(exclude_channel=only.offer.channel.id, max_credits=pol.hedge_max_credits)
                    if c2 is None:
                        can_hedge = False
                    else:
                        self.hedged = True
                        log.info("hedging %s -> %s", only.offer.key, c2.offer.key)
                        start(c2)
                    continue

                wins: list[tuple[Candidate, Any, list[Delta], float]] = []
                for t in done:
                    c = pending.pop(t)
                    try:
                        agen, first, attempt_ttft = t.result()
                    except UpstreamError as err:
                        last = err
                        if err.failure in CHANNEL_WIDE:
                            bad_channels.add(c.offer.channel.id)
                        if err.failure is FailureClass.MODEL_INCOMPATIBLE:
                            # 同一个模型在别的渠道上还是同一个模型，别再白试一遍
                            bad_models.add(c.offer.model.key)
                        if not err.retryable:
                            self.failure = err.failure.value
                            await self._cancel(pending, charge=False)
                            pending.clear()
                            raise GatewayError(400, "upstream_rejected_request", err.message[:300]) from None
                        continue
                    wins.append((c, agen, first, attempt_ttft))
                if not wins:
                    continue

                c, agen, first, attempt_ttft = wins[0]
                for xc, xagen, _, _ in wins[1:]:
                    self._end_attempt(xc)
                    e.spawn(_aclose(xagen))
                    self._charge_loser(xc)
                losers = dict(pending)
                pending.clear()
                if losers:
                    await self._cancel(losers, charge=True)
                now = time.perf_counter()
                self.decision, self._agen, self._pending = c, agen, first
                self.ttft, self.attempt_ttft = now - self.t0, attempt_ttft
                self.last_activity = time.monotonic()
                return
        except GatewayError:
            raise
        except BaseException:
            # 客户端在首 token 前断开：请求已送达上游并可能产生输入费用，按输入估算计费（防"发完就断"）
            if pending and self.decision is None:
                self.decision = next(iter(pending.values()))
            for t in pending:
                t.cancel()
            raise

        if last is None:
            raise GatewayError(503, "upstream_unavailable", "all upstream candidates are busy, retry shortly", {"Retry-After": "5"})
        self.failure = last.failure.value
        raise GatewayError(502, "upstream_error", f"all upstream attempts failed ({last.failure.value})", {"Retry-After": "5"})

    # ── 流式 ───────────────────────────────────────────────────

    def _account(self, d: Delta) -> None:
        self.last_activity = time.monotonic()
        if d.usage:
            self.usage = self.usage.merge(d.usage)
        # 输出 token 估算：UTF-8 字节 / 4（英文约 4 字符/token，中文 3 字节约 0.75 token）
        if d.content:
            self.out_bytes += len(d.content.encode("utf-8", "ignore"))
        if d.reasoning:
            self.out_bytes += len(d.reasoning.encode("utf-8", "ignore"))
        for tc in d.tool_calls or []:
            fn = tc.get("function") or {}
            self.out_bytes += len((fn.get("arguments") or "").encode("utf-8", "ignore")) + len(fn.get("name") or "")
        if d.finish_reason:
            self.finish_reason = d.finish_reason
        self.outcome.feed(d)

    def _over_budget(self) -> bool:
        if self.budget is None or self.decision is None:
            return False
        c = self.decision
        out = max(self.out_bytes // 4, self.usage.output)
        spent = c.offer.list_usd(c.est_miss, c.est_cached, out) / self.engine.settings.policy.credit_usd
        return spent > self.budget

    async def _stream_once(self) -> AsyncIterator[Delta]:
        """把当前这次尝试的输出流出来。不 finalize、不做重试判断。"""
        idle = self.engine.settings.policy.stream_idle_timeout_s
        # 验证阶段缓冲过的片段：原样交出去但**不再计一次量**，否则工具参数会被拼两遍
        replay, self._replay = self._replay, []
        for d in replay:
            yield d
        pending, self._pending = self._pending, []
        for d in pending:
            self._account(d)
            yield d
        while self._agen is not None:
            try:
                d = await asyncio.wait_for(self._agen.__anext__(), timeout=idle)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                raise UpstreamError(FailureClass.TIMEOUT, f"stream idle for more than {idle:g}s") from None
            self._account(d)
            yield d
            if not d.finish_reason and self._over_budget():
                self.status, self.finish_reason = "quota_cut", "length"
                log.info("request %s cut: daily credit budget reached", self.request_id)
                yield Delta(finish_reason="length")
                break

    # ── 验证失败后的定向重试（Weitzman 保留值规则）──────────────────
    #
    # 前提是有了本地验证器（verify.py）：现在"这次回答是不是结构上不合格"可观测了，
    # 于是"要不要换个模型重做"变成一个最优停止问题。Weitzman(1979) 的潘多拉盒子给了
    # 闭式解：每个候选算一个保留值 σ，按 σ 降序开箱，手里的最好结果 ≥ 下一个 σ 时停。
    # 把奖励建模成伯努利（1−p 概率拿到 value，p 概率拿到 0），代入 E[max(X−σ,0)]=c 得
    #     σ = value − c / (1 − p)          c = 成本 + 等待成本，p = 验证器坏比率
    # 手里是坏答案（价值 0），所以"继续开箱"的条件恰好是 σ > 0，换谁就是 argmax σ。

    def _can_retry(self) -> bool:
        pol = self.engine.settings.policy
        if not pol.verify_retry or self.plan is None or self.f is None or self.decision is None:
            return False
        if self.attempts > pol.max_attempts:
            return False
        if not self.stream:
            return True  # 非流式本来就要收齐才返回，缓冲零代价
        # 流式：只有"面向机器消费"的回答才缓冲（客户端拿不到半个工具调用或半个 JSON，
        # 所以缓冲不影响体验）；给人读的散文一旦开始流就不能收回。
        return bool(self.f.has_tools or self.f.json_mode)

    async def _probe_stream_shape(self) -> bool:
        """声明了工具时，看实际输出先是工具调用还是正文。

        agent 通常每轮都附带 tools，即使模型最终只回答普通文字。只看请求字段会把
        这种正文完整缓冲到结束，客户端直到回答完成才看到第一个流式片段。正文或
        思考片段先出现时立即转发；之后即使出现工具调用，也不能再撤回已发的片段。
        """
        buf: list[Delta] = []
        gen = self._stream_once()
        try:
            async for d in gen:
                buf.append(d)
                if d.tool_calls:
                    return True
                if d.content or d.reasoning or d.extra:
                    return False
            return True  # 空回答仍可由验证器判断是否需要重做
        except UpstreamError as ex:
            self._pending_error = ex
            return False
        finally:
            self._replay = buf
            await gen.aclose()

    @staticmethod
    def _sigma(c: Candidate, p: float) -> float:
        return c.value - (c.cost + c.latency_cost) / (1.0 - min(max(p, 0.0), 0.95))

    def retry_target(self, failed: Candidate) -> Candidate | None:
        """按保留值挑重试对象；没有比"就这样交出去"更值得开的盒子就返回 None。

        门槛取失败候选自己的 σ，而不是 0 —— 任务价值表只为相对排序校准过，绝对值
        没有意义（chat/1 的 score 本来就是负的），拿 0 当门槛会让便宜任务永不重试。
        失败候选的坏概率按后验取 max(实测, 0.5)：它刚刚在这条 prompt 上失败过。
        """
        e = self.engine
        pol = e.settings.policy
        assert self.plan is not None
        best = None
        best_sigma = self._sigma(failed, max(failed.bad_rate, 0.5))
        for c in self.plan.ranked:
            if c.offer.model.key == failed.offer.model.key:
                continue  # 同一个模型换个渠道也修不了结构性错误
            if c.credits > pol.verify_retry_max_credits:
                continue
            ok, _ = e.health.available(c.offer.key)
            if not ok or e.channel_load.get(c.offer.channel.id, 0) >= c.offer.channel.max_concurrency:
                continue
            sigma = self._sigma(c, c.bad_rate)
            if sigma > best_sigma:
                best, best_sigma = c, sigma
        return best

    def _absorb(self, c: Candidate, u: Usage) -> None:
        """被丢弃的那次尝试由网关承担（与对冲落败同一处理），不计入用户额度。"""
        e, o = self.engine, c.offer
        real = o.real_usd(u.input_miss, u.input_cached, u.output)
        units = o.units(u.input_miss, u.input_cached, u.output)
        now = time.time()
        if units:
            e.capacity.commit(o, units, now)
            e.writer.add_channel_usage(o.budget_scope, now, units)
        if real:
            e.note_metered(self.day, o.channel.id, real)
            e.writer.add_metered(self.day, o.channel.id, real)
            e.metrics.inc("thalamus_upstream_usd_total", real, channel=o.channel.id, kind="verify_retry")
        self.hedge_cost_usd += real

    async def resolve(self) -> None:
        """在响应开始之前把这次尝试收下来验一遍，必要时换模型重做。

        必须放在响应开始之前：HTTP 响应头先于 body 发出，如果等到流出去一半才换模型，
        `X-Thalamus-Model` / `X-Thalamus-Attempts` 就会说谎。缓冲的代价只落在
        "面向机器消费"的回答上（见 _can_retry）。
        """
        if not self._can_retry():
            return
        pol = self.engine.settings.policy
        assert self.f is not None
        if self.stream and self.f.has_tools and not self.f.json_mode:
            if not await self._probe_stream_shape():
                return
        buf: list[Delta] = []
        try:
            async for d in self._stream_once():
                buf.append(d)
                if self.out_bytes > pol.verify_buffer_bytes and not d.finish_reason:
                    break  # 回答比预期长：放弃重试，恢复真正的流式
            else:
                if await self._maybe_redo(pol):
                    return  # 已换模型重连，交给 deltas() 正常流第二次
        except UpstreamError as ex:
            self._pending_error = ex  # 保持"先把已有输出交出去，再报错"的语义
        self._replay = buf

    async def _maybe_redo(self, pol: Any) -> bool:
        assert self.decision is not None and self.intent is not None and self.f is not None
        bad = [fl for fl in self.outcome.flags(self.finish_reason, self.status) if fl in pol.verify_retry_flags]
        if not bad:
            return False
        target = self.retry_target(self.decision)
        if target is None:
            return False
        failed, failed_usage = self.decision, self.usage
        log.info("verify retry: %s -> %s (%s)", failed.offer.key, target.offer.key, ",".join(bad))
        try:
            await self._connect_to(target)
        except (UpstreamError, GatewayError):
            return False  # 重试连不上：原样把第一次的结果交出去，好过整个请求失败

        e, task = self.engine, self.intent.task
        for fl in bad:  # 第一次的负面信号必须记下来，否则学习闭环断了
            e.learner.penalize(failed.offer.model.key, task, oc.LEARN_WEIGHT.get(fl, 0.5))
        e.outcomes.observe(failed.offer.model.key, task, True)
        e.metrics.inc("thalamus_verify_retry_total", model=failed.offer.model.key, flag=bad[0])
        self._end_attempt(failed)
        self._absorb(failed, failed_usage)
        self.verify_retried, self.verify_flags = True, bad
        # 丢弃第一次的计量与判定，只按重做的这次给用户计费
        self.usage, self.out_bytes, self.finish_reason = Usage(), 0, None
        self.outcome = oc.OutcomeTracker(
            self.f.json_mode, tools=self.body.get("tools"), check_code=task in ("code", "agent")
        )
        return True

    async def deltas(self) -> AsyncIterator[Delta]:
        try:
            async for d in self._stream_once():
                yield d
            if self._pending_error is not None:
                err, self._pending_error = self._pending_error, None
                raise err
        except UpstreamError as ex:
            self.status, self.failure = "error", ex.failure.value
            raise
        except BaseException:
            if self.status == "ok":
                self.status = "client_closed"
            raise
        finally:
            self.finalize()

    def headers(self) -> dict[str, str]:
        assert self.decision is not None and self.intent is not None
        h = {
            "X-Thalamus-Model": self.decision.offer.model.key,
            "X-Thalamus-Task": self.intent.label(),
            "X-Thalamus-Attempts": str(self.attempts),
            "X-Thalamus-Request-Id": self.request_id,
            "X-Thalamus-Quota-Mode": self.mode,
        }
        if self.t_dispatch is not None:
            # 网关自己花了多久才把请求发给上游（鉴权+读体+判定+配额+打分）。
            # 外部可以据此把"网关开销"和"网络+上游"分开，不需要 admin 权限。
            h["X-Thalamus-Route-Ms"] = f"{(self.t_dispatch - self.t0) * 1000:.1f}"
        if self.hedged:
            h["X-Thalamus-Hedged"] = "1"
        return h

    @property
    def model_name(self) -> str:
        return self.decision.offer.model.key if self.decision else "thalamus"

    def usage_openai(self) -> dict[str, Any]:
        u = self.final_usage if self.finalized else self.usage
        d: dict[str, Any] = {
            "prompt_tokens": u.input_total,
            "completion_tokens": u.output,
            "total_tokens": u.input_total + u.output,
            "prompt_tokens_details": {"cached_tokens": u.input_cached},
        }
        if u.reasoning:
            d["completion_tokens_details"] = {"reasoning_tokens": u.reasoning}
        return d

    def _chunk(self, delta: dict[str, Any], finish: str | None) -> str:
        obj = {
            "id": self.request_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))}\n\n"

    def _close_abnormal(self, gen: Any) -> None:
        """外层生成器被关闭/取消时立即结算，不依赖内层生成器被垃圾回收。"""
        if not self.finalized:
            if self.status == "ok":
                self.status = "client_closed"
            self.finalize()
        self.engine.spawn(_aclose(gen))

    async def sse(self, include_usage: bool) -> AsyncIterator[str]:
        first = True
        gen = self.deltas()
        try:
            async for d in gen:
                delta: dict[str, Any] = {}
                if d.content:
                    delta["content"] = d.content
                if d.reasoning:
                    delta["reasoning_content"] = d.reasoning
                if d.tool_calls:
                    delta["tool_calls"] = d.tool_calls
                if d.extra:
                    delta.update(d.extra)
                if not delta and not d.finish_reason:
                    continue
                if first:
                    delta = {"role": "assistant", **delta}
                    if "content" not in delta and "tool_calls" not in delta:
                        delta["content"] = ""
                    first = False
                yield self._chunk(delta, d.finish_reason)
            if include_usage:
                obj = {
                    "id": self.request_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model_name,
                    "choices": [],
                    "usage": self.usage_openai(),
                }
                yield f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"
        except UpstreamError as ex:
            err = {"error": {"message": f"upstream stream interrupted ({ex.failure.value})", "type": ex.failure.value}}
            yield f"data: {json.dumps(err)}\n\n"
        finally:
            self._close_abnormal(gen)
        yield "data: [DONE]\n\n"

    async def collect(self) -> dict[str, Any]:
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        extra: dict[str, Any] = {}
        gen = self.deltas()
        try:
            async for d in gen:
                content.append(d.content)
                reasoning.append(d.reasoning)
                if d.extra:
                    extra.update(d.extra)
                for tc in d.tool_calls or []:
                    i = int(tc.get("index", 0))
                    cur = calls.setdefault(i, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        cur["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if name and name != cur["function"]["name"]:
                        cur["function"]["name"] += name
                    cur["function"]["arguments"] += fn.get("arguments") or ""
        finally:
            self._close_abnormal(gen)
        text = "".join(content)
        message: dict[str, Any] = {"role": "assistant", "content": text if (text or not calls) else None}
        if any(reasoning):
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        message.update({k: v for k, v in extra.items() if k not in message})
        return {
            "id": self.request_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model_name,
            "choices": [
                {"index": 0, "message": message, "finish_reason": self.finish_reason or ("tool_calls" if calls else "stop")}
            ],
            "usage": self.usage_openai(),
        }

    # ── 结算（恰好一次）──────────────────────────────────────────

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        e = self.engine
        e._calls.discard(self)
        e.limiter.release(self.key.id)
        duration = time.perf_counter() - self.t0
        c = self.decision
        task = self.intent.task if self.intent else "none"
        if self.t_dispatch is not None:
            e.metrics.observe("thalamus_routing_overhead_seconds", self.t_dispatch - self.t0, OVERHEAD_BUCKETS)
        if self.attempts:
            e.metrics.inc("thalamus_attempts_total", self.attempts)

        if c is None:
            if self.rid:
                e.ledger.release(self.rid)
            e.metrics.inc("thalamus_requests_total", status=self.status, task=task, model="none")
            e.writer.log_request(self._record(None, Usage(), duration))
            return

        offer = c.offer
        self._end_attempt(c)
        if self._agen is not None:
            e.spawn(_aclose(self._agen))

        u = self.usage
        est_out = max(1, self.out_bytes // 4)
        if u.empty:
            self.estimated = True
            u = Usage(input_miss=c.est_miss, input_cached=c.est_cached, output=est_out)
        elif u.output < est_out:
            # 流被截断/中断时上游常常只给了开头的输入 usage，输出按实际转发量估算
            self.estimated = True
            u = Usage(u.input_miss or c.est_miss, u.input_cached, est_out, u.reasoning)
        self.final_usage = u
        pol = e.settings.policy
        self.real_usd = offer.real_usd(u.input_miss, u.input_cached, u.output)
        self.units = offer.units(u.input_miss, u.input_cached, u.output)
        self.credits = offer.list_usd(u.input_miss, u.input_cached, u.output) / pol.credit_usd

        # 结果信号
        self.flags = self.outcome.flags(self.finish_reason, self.status)
        if self.hedged:
            self.flags.append(oc.HEDGED)
        if self.verify_retried:
            self.flags.append(oc.VERIFY_RETRY)
        if self.status == "quota_cut":
            self.flags.append(oc.QUOTA_CUT)
        negative = [fl for fl in self.flags if fl in oc.LEARN_WEIGHT]

        gen_s = duration - (self.ttft or 0.0)
        tps = u.output / gen_s if u.output > 20 and gen_s > 0.2 else None
        assert self.intent is not None and self.f is not None
        model = offer.model.key
        if self.status == "ok":
            e.health.success(offer.key, self.attempt_ttft, tps)
            # 验证器的判定同时喂两处：learner 微调质量分，outcomes 给"答错的代价"定价
            e.outcomes.observe(model, task, any(fl in oc.BAD_FLAGS for fl in self.flags))
            for fl in negative:
                e.learner.penalize(model, task, oc.LEARN_WEIGHT[fl])
                if fl in (oc.TOOL_ARGS_INVALID, oc.TOOL_SCHEMA_INVALID) and task != "agent":
                    e.learner.penalize(model, "agent", oc.LEARN_WEIGHT[fl])
            if not negative:
                e.learner.reward(model, task)
        elif self.status in ("client_closed", "abandoned", "shutdown", "quota_cut"):
            if self.attempt_ttft is None:
                e.health.abort(offer.key)  # 首 token 前就断开：说明不了上游好坏
            else:
                e.health.success(offer.key, self.attempt_ttft, None)
        elif self.failure:
            try:
                e.health.failure(offer.key, FailureClass(self.failure), "mid-stream")
            except ValueError:
                pass
        if self.units:
            e.capacity.commit(offer, self.units)
        if self.real_usd:
            e.note_metered(self.day, offer.channel.id, self.real_usd)
        if self.status != "error":
            e.sessions.put(
                self.f.session_key,
                SessionState(
                    task=self.intent.task,
                    difficulty=self.intent.difficulty,
                    offer_key=offer.key,
                    model_key=model,
                    prompt_tokens=u.input_total,
                    last_user_hash=self.f.last_user_hash,
                    n_messages=self.f.n_messages,
                    ts=time.time(),
                    request_id=self.request_id,
                ),
            )

        m = e.metrics
        m.inc("thalamus_requests_total", status=self.status, task=task, model=model)
        m.inc("thalamus_credits_total", self.credits, model=model)
        if self.real_usd:
            m.inc("thalamus_upstream_usd_total", self.real_usd, channel=offer.channel.id, kind="served")
        if self.units:
            m.inc("thalamus_subscription_units_total", self.units, scope=offer.budget_scope)
        m.inc("thalamus_tokens_total", u.input_miss, kind="input_miss", model=model)
        m.inc("thalamus_tokens_total", u.input_cached, kind="input_cached", model=model)
        m.inc("thalamus_tokens_total", u.output, kind="output", model=model)
        for fl in self.flags:
            m.inc("thalamus_outcome_flags_total", flag=fl, model=model)
        if self.ttft is not None:
            m.observe("thalamus_ttft_seconds", self.ttft, LATENCY_BUCKETS, model=model)

        # 账本在内存里立即结算；其余全部进写缓冲，由后台批量落库
        now = time.time()
        if self.rid:
            e.ledger.settle(self.rid, self.credits, u.input_total, u.input_cached, u.output)
        if self.units:
            e.writer.add_channel_usage(offer.budget_scope, now, self.units)
        if self.real_usd:
            e.writer.add_metered(self.day, offer.channel.id, self.real_usd)
        e.writer.touch_key(self.key.id, now)
        e.writer.log_request(self._record(offer, u, duration))

    def _record(self, offer: Offer | None, u: Usage, duration: float) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "key_id": self.key.id,
            "ts": time.time(),
            "task": self.intent.task if self.intent else None,
            "difficulty": self.intent.difficulty if self.intent else None,
            "model": offer.model.key if offer else None,
            "channel": offer.channel.id if offer else None,
            "credits": round(self.credits, 4),
            "real_usd": round(self.real_usd + self.hedge_cost_usd, 7),
            "units": round(self.units, 4),
            "in_miss": u.input_miss,
            "in_cached": u.input_cached,
            "out_tokens": u.output,
            "ttft_ms": round(self.ttft * 1000, 1) if self.ttft else None,
            "duration_ms": round(duration * 1000, 1),
            "attempts": self.attempts,
            "status": self.status,
            "failure": self.failure or (",".join(self.failures)[:300] if self.failures else None),
            "reason": (self.plan.reason() if self.plan and self.plan.ranked else None),
            "stream": int(self.stream),
            "mode": self.mode,
            "flags": ",".join(self.flags),
        }

"""按 key 的每日额度、每分钟请求数与并发控制。"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import Settings
from .store import ApiKey, Store
from .types import GatewayError


def day_str(tz: str, now: float | None = None) -> str:
    return datetime.fromtimestamp(now or time.time(), ZoneInfo(tz)).strftime("%Y-%m-%d")


def seconds_to_reset(tz: str, now: float | None = None) -> int:
    t = datetime.fromtimestamp(now or time.time(), ZoneInfo(tz))
    tomorrow = (t + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - t).total_seconds()))


@dataclass
class QuotaState:
    used: float
    reserved: float
    limit: float  # <= 0 表示不限
    mode: str  # normal | soft | economy | reject

    @property
    def ratio(self) -> float:
        return (self.used + self.reserved) / self.limit if self.limit > 0 else 0.0


USAGE_FIELDS = ("used", "requests", "in_tokens", "cached_tokens", "out_tokens")


class Ledger:
    """预留 → 结算两段式。额度单位 credits（按标价折算）。

    账本在内存里，数据库只是持久化（persist.WriteBehind 批量写回）。
    asyncio 单线程：reserve 里"检查 + 扣减"之间没有 await，天然原子，并发请求不可能一起穿过上限。
    首次访问某个 key 当天的记录时从库里加载一次（单飞，避免并发重复加载）。
    """

    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings
        self._rows: dict[tuple[str, str], dict[str, float]] = {}
        self._loading: dict[tuple[str, str], asyncio.Future] = {}
        self._res: dict[str, tuple[str, str, float, float]] = {}  # rid -> (key_id, day, amount, created)
        self._dirty: set[tuple[str, str]] = set()

    def today(self) -> str:
        return day_str(self.settings.policy.timezone)

    async def _row(self, key_id: str, day: str) -> dict[str, float]:
        k = (key_id, day)
        row = self._rows.get(k)
        if row is not None:
            return row
        fut = self._loading.get(k)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            self._loading[k] = fut
            try:
                u = await self.store.get_usage(key_id, day)
                row = self._rows.get(k)  # 加载期间可能已被 reset 创建
                if row is None:
                    row = {f: float(u.get(f) or 0) for f in USAGE_FIELDS}
                    row["reserved"] = 0.0  # 进程重启后，库里残留的预留都已失效
                    self._rows[k] = row
                fut.set_result(row)
            except BaseException as e:
                fut.set_exception(e)
                raise
            finally:
                self._loading.pop(k, None)
            return row
        return await asyncio.shield(fut)

    async def usage(self, key_id: str, day: str | None = None) -> dict[str, float]:
        return dict(await self._row(key_id, day or self.today()))

    async def state(self, key: ApiKey, day: str | None = None) -> QuotaState:
        u = await self._row(key.id, day or self.today())
        limit = key.daily_credits
        st = QuotaState(u["used"], u["reserved"], limit, "normal")
        if limit <= 0:
            st.mode = "reject"  # 0 = 冻结，而不是不限额
            return st
        k = self.settings.policy.keys
        r = st.ratio
        if r < k.soft_ratio:
            st.mode = "normal"
        elif r < 1.0:
            st.mode = "soft"
        elif key.exhaust_mode == "economy" and r < 1.0 + k.grace_ratio:
            st.mode = "economy"
        else:
            st.mode = "reject"
        return st

    def ceiling(self, key: ApiKey, mode: str) -> float:
        if key.daily_credits <= 0:
            return 0.0
        if mode == "economy":
            return key.daily_credits * (1.0 + self.settings.policy.keys.grace_ratio)
        return key.daily_credits

    async def reserve(self, key: ApiKey, credits: float, mode: str, day: str) -> str | None:
        row = await self._row(key.id, day)
        # ↓ 从这里到 return 之间没有 await：检查与扣减是原子的
        amount = max(credits, 0.01)
        if row["used"] + row["reserved"] + amount > self.ceiling(key, mode):
            return None
        row["reserved"] += amount
        rid = f"r_{uuid.uuid4().hex[:16]}"
        self._res[rid] = (key.id, day, amount, time.monotonic())
        return rid

    def settle(self, rid: str, credits: float, in_tokens: int, cached: int, out: int) -> None:
        r = self._res.pop(rid, None)
        if r is None:
            return  # 重复结算无效
        key_id, day, amount, _ = r
        row = self._rows.get((key_id, day))
        if row is None:
            return
        row["reserved"] = max(0.0, row["reserved"] - amount)
        row["used"] += credits
        row["requests"] += 1
        row["in_tokens"] += in_tokens
        row["cached_tokens"] += cached
        row["out_tokens"] += out
        self._dirty.add((key_id, day))

    def release(self, rid: str) -> None:
        r = self._res.pop(rid, None)
        if r is None:
            return
        row = self._rows.get((r[0], r[1]))
        if row is not None:
            row["reserved"] = max(0.0, row["reserved"] - r[2])

    def sweep(self, older_than_s: float = 3600) -> int:
        """回收异常残留的预留；清掉已写回库、且不是今天的内存记录。"""
        now = time.monotonic()
        stale = [rid for rid, (_, _, _, t) in self._res.items() if now - t > older_than_s]
        for rid in stale:
            self.release(rid)
        today = self.today()
        busy = {(k, d) for k, d, _, _ in self._res.values()}
        for k in [k for k in self._rows if k[1] != today and k not in self._dirty and k not in busy]:
            self._rows.pop(k, None)
        return len(stale)

    def reset_used(self, key_id: str, day: str) -> None:
        row = self._rows.setdefault((key_id, day), {**{f: 0.0 for f in USAGE_FIELDS}, "reserved": 0.0})
        row["used"] = 0.0
        self._dirty.add((key_id, day))

    def forget(self, key_id: str) -> None:
        for k in [k for k in self._rows if k[0] == key_id]:
            self._rows.pop(k, None)
            self._dirty.discard(k)

    async def usage_for_day(self, day: str) -> dict[str, dict[str, float]]:
        """管理台用：库里的记录叠加内存里的最新值。"""
        out = await self.store.usage_for_day(day)
        for (key_id, d), row in self._rows.items():
            if d == day:
                out[key_id] = {**out.get(key_id, {}), **row}
        return out

    # ── 给 WriteBehind 用 ─────────────────────────────────────

    def take_dirty(self) -> list[tuple]:
        rows = []
        for key_id, day in self._dirty:
            r = self._rows.get((key_id, day))
            if r is not None:
                rows.append((key_id, day, r["used"], int(r["requests"]), int(r["in_tokens"]),
                             int(r["cached_tokens"]), int(r["out_tokens"])))
        self._dirty.clear()
        return rows

    def restore_dirty(self, rows: list[tuple]) -> None:
        for row in rows:
            self._dirty.add((row[0], row[1]))

    def exhausted_error(self, key: ApiKey) -> GatewayError:
        reset = seconds_to_reset(self.settings.policy.timezone)
        return GatewayError(
            429,
            "quota_exceeded",
            f"daily credit quota exhausted for this key; resets in {reset}s",
            {"Retry-After": str(reset), "X-Thalamus-Quota-Reset": str(reset)},
        )


class Limiter:
    """每 key 的 RPM（滑动 60s）与并发。上限每次现读，改配置立即生效且不会错放许可。"""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._inflight: dict[str, int] = {}
        self._cond: dict[str, asyncio.Condition] = {}

    def check_rpm(self, key: ApiKey) -> None:
        now = time.monotonic()
        q = self._hits.setdefault(key.id, deque())
        while q and q[0] <= now - 60:
            q.popleft()
        if len(q) >= max(key.rpm, 1):  # 0 或负数按 1 处理，永远不会变成不限
            retry = max(1, int(60 - (now - q[0])) + 1)
            raise GatewayError(
                429, "rate_limited", f"rate limit {key.rpm}/min exceeded", {"Retry-After": str(retry)}
            )
        q.append(now)

    async def acquire(self, key: ApiKey, wait_s: float = 10.0) -> None:
        limit = max(key.max_concurrency, 1)
        cond = self._cond.setdefault(key.id, asyncio.Condition())
        async with cond:
            try:
                await asyncio.wait_for(
                    cond.wait_for(lambda: self._inflight.get(key.id, 0) < limit), timeout=wait_s
                )
            except asyncio.TimeoutError as e:
                raise GatewayError(
                    429, "concurrency_limited", f"too many concurrent requests (max {key.max_concurrency})",
                    {"Retry-After": "2"},
                ) from e
            self._inflight[key.id] = self._inflight.get(key.id, 0) + 1

    def release(self, key_id: str) -> None:
        self._inflight[key_id] = max(0, self._inflight.get(key_id, 0) - 1)
        cond = self._cond.get(key_id)
        if cond is not None:
            asyncio.get_running_loop().create_task(self._notify(cond))

    @staticmethod
    async def _notify(cond: asyncio.Condition) -> None:
        async with cond:
            cond.notify_all()

    def inflight(self, key_id: str) -> int:
        return self._inflight.get(key_id, 0)

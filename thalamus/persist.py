"""后台批量写库。

压测发现：所有数据库操作挤在同一个 aiosqlite 线程里串行执行，请求路径上的读（当日用量、预留）
要排在其他请求的结算写后面，并发 20 时仅"发出上游请求之前"就要 20ms。

现在：
  · 额度账本在内存里（quota.Ledger），请求路径不再读写数据库
  · 请求日志、用量、订阅窗口、按量花费、last_used、事后标记全部进缓冲区，
    每 interval 秒用独立连接在一个事务里批量提交
代价：进程崩溃（非正常退出）最多丢失最近 interval 秒的记录；正常关闭会先刷盘。
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:  # pragma: no cover
    from .quota import Ledger

log = logging.getLogger("thalamus.persist")

USAGE_UPSERT = (
    "INSERT INTO usage_daily (key_id, day, used, reserved, requests, in_tokens, cached_tokens, out_tokens) "
    "VALUES (?,?,?,0,?,?,?,?) ON CONFLICT(key_id, day) DO UPDATE SET used=excluded.used, reserved=0, "
    "requests=excluded.requests, in_tokens=excluded.in_tokens, cached_tokens=excluded.cached_tokens, "
    "out_tokens=excluded.out_tokens"
)
CHANNEL_UPSERT = (
    "INSERT INTO channel_usage (scope, minute, units) VALUES (?,?,?) "
    "ON CONFLICT(scope, minute) DO UPDATE SET units = units + excluded.units"
)
METERED_UPSERT = (
    "INSERT INTO metered_daily (day, channel, usd) VALUES (?,?,?) "
    "ON CONFLICT(day, channel) DO UPDATE SET usd = usd + excluded.usd"
)
FLAG_UPDATE = (
    "UPDATE requests SET flags = CASE WHEN flags IS NULL OR flags = '' THEN ? "
    "WHEN (',' || flags || ',') LIKE ('%,' || ? || ',%') THEN flags ELSE flags || ',' || ? END WHERE id = ?"
)


class WriteBehind:
    def __init__(self, db_path: str, ledger: "Ledger", interval: float = 0.25, max_logs: int = 100_000):
        self.db_path = db_path
        self.ledger = ledger
        self.interval = interval
        self.max_logs = max_logs
        self._db: aiosqlite.Connection | None = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._reset_buffers()
        self.dropped = 0
        self.flushes = 0

    def _reset_buffers(self) -> None:
        self._logs: list[dict[str, Any]] = []
        self._channel: dict[tuple[str, int], float] = defaultdict(float)
        self._metered: dict[tuple[str, str], float] = defaultdict(float)
        self._touch: dict[str, float] = {}
        self._flags: list[tuple[str, str]] = []

    # ── 生命周期 ───────────────────────────────────────────────

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self.db_path, isolation_level=None)
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.flush()
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("write-behind flush failed")

    # ── 入队（同步，O(1)）─────────────────────────────────────────

    def log_request(self, rec: dict[str, Any]) -> None:
        if len(self._logs) >= self.max_logs:
            self.dropped += 1
            return
        self._logs.append(rec)

    def add_channel_usage(self, scope: str, ts: float, units: float) -> None:
        self._channel[(scope, int(ts // 60))] += units

    def add_metered(self, day: str, channel: str, usd: float) -> None:
        self._metered[(day, channel)] += usd

    def touch_key(self, key_id: str, ts: float) -> None:
        self._touch[key_id] = ts

    def add_flag(self, request_id: str, flag: str) -> None:
        self._flags.append((request_id, flag))

    @property
    def pending(self) -> int:
        return len(self._logs) + len(self._channel) + len(self._metered) + len(self._touch) + len(self._flags)

    # ── 刷盘 ───────────────────────────────────────────────────

    async def flush(self) -> None:
        if self._db is None:
            return
        async with self._lock:
            usage_rows = self.ledger.take_dirty()
            if not usage_rows and not self.pending:
                return
            logs, channel, metered = self._logs, self._channel, self._metered
            touch, flags = self._touch, self._flags
            self._reset_buffers()
            db = self._db
            try:
                await db.execute("BEGIN")
                if logs:
                    cols = list(logs[0].keys())
                    await db.executemany(
                        f"INSERT OR REPLACE INTO requests ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [tuple(r.get(c) for c in cols) for r in logs],
                    )
                if flags:
                    await db.executemany(FLAG_UPDATE, [(f, f, f, rid) for rid, f in flags])
                if usage_rows:
                    await db.executemany(USAGE_UPSERT, usage_rows)
                if channel:
                    await db.executemany(CHANNEL_UPSERT, [(s, m, u) for (s, m), u in channel.items()])
                if metered:
                    await db.executemany(METERED_UPSERT, [(d, c, u) for (d, c), u in metered.items()])
                if touch:
                    await db.executemany("UPDATE api_keys SET last_used_at=? WHERE id=?", [(ts, k) for k, ts in touch.items()])
                await db.execute("COMMIT")
                self.flushes += 1
            except BaseException:
                try:
                    await db.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                # 放回缓冲区，下次重试
                self.ledger.restore_dirty(usage_rows)
                self._logs[:0] = logs[: self.max_logs]
                for k, v in channel.items():
                    self._channel[k] += v
                for k, v in metered.items():
                    self._metered[k] += v
                for k, v in touch.items():
                    self._touch.setdefault(k, v)
                self._flags[:0] = flags
                raise

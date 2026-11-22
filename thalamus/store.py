"""SQLite 持久层（WAL）。

不存任何 prompt / 回复原文，只存计量与路由元数据。
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS api_keys (
  id                TEXT PRIMARY KEY,
  name              TEXT NOT NULL,
  key_hash          TEXT UNIQUE NOT NULL,
  prefix            TEXT NOT NULL,
  daily_credits     REAL NOT NULL,
  rpm               INTEGER NOT NULL,
  max_concurrency   INTEGER NOT NULL,
  max_output_tokens INTEGER NOT NULL DEFAULT 0,
  plan              TEXT NOT NULL DEFAULT 'standard',
  exhaust_mode      TEXT NOT NULL DEFAULT 'economy',
  allowed_models    TEXT,
  allowed_regions   TEXT,
  enabled           INTEGER NOT NULL DEFAULT 1,
  expires_at        REAL,
  note              TEXT NOT NULL DEFAULT '',
  created_at        REAL NOT NULL,
  last_used_at      REAL
);

CREATE TABLE IF NOT EXISTS usage_daily (
  key_id        TEXT NOT NULL,
  day           TEXT NOT NULL,
  used          REAL NOT NULL DEFAULT 0,
  reserved      REAL NOT NULL DEFAULT 0,
  requests      INTEGER NOT NULL DEFAULT 0,
  in_tokens     INTEGER NOT NULL DEFAULT 0,
  cached_tokens INTEGER NOT NULL DEFAULT 0,
  out_tokens    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (key_id, day)
);

CREATE TABLE IF NOT EXISTS reservations (
  id         TEXT PRIMARY KEY,
  key_id     TEXT NOT NULL,
  day        TEXT NOT NULL,
  amount     REAL NOT NULL,
  state      TEXT NOT NULL DEFAULT 'open',
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_res_open ON reservations(state, created_at);

CREATE TABLE IF NOT EXISTS requests (
  id          TEXT PRIMARY KEY,
  key_id      TEXT,
  ts          REAL NOT NULL,
  task        TEXT,
  difficulty  INTEGER,
  model       TEXT,
  channel     TEXT,
  credits     REAL DEFAULT 0,
  real_usd    REAL DEFAULT 0,
  units       REAL DEFAULT 0,
  in_miss     INTEGER DEFAULT 0,
  in_cached   INTEGER DEFAULT 0,
  out_tokens  INTEGER DEFAULT 0,
  ttft_ms     REAL,
  duration_ms REAL,
  attempts    INTEGER DEFAULT 1,
  status      TEXT,
  failure     TEXT,
  reason      TEXT,
  stream      INTEGER DEFAULT 1,
  mode        TEXT,
  flags       TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_req_ts ON requests(ts DESC);
CREATE INDEX IF NOT EXISTS idx_req_key_ts ON requests(key_id, ts DESC);

CREATE TABLE IF NOT EXISTS channel_usage (
  scope  TEXT NOT NULL,
  minute INTEGER NOT NULL,
  units  REAL NOT NULL,
  PRIMARY KEY (scope, minute)
);

CREATE TABLE IF NOT EXISTS metered_daily (
  day     TEXT NOT NULL,
  channel TEXT NOT NULL,
  usd     REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (day, channel)
);

CREATE TABLE IF NOT EXISTS learned_quality (
  model      TEXT NOT NULL,
  task       TEXT NOT NULL,
  delta      REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (model, task)
);
"""

KEY_FIELDS = (
    "name", "daily_credits", "rpm", "max_concurrency", "max_output_tokens", "plan",
    "exhaust_mode", "allowed_models", "allowed_regions", "enabled", "expires_at", "note",
)
JSON_FIELDS = ("allowed_models", "allowed_regions")


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class ApiKey:
    id: str
    name: str
    prefix: str
    daily_credits: float
    rpm: int
    max_concurrency: int
    max_output_tokens: int
    plan: str
    exhaust_mode: str
    allowed_models: list[str] | None
    allowed_regions: list[str] | None  # None = 不限；否则只允许这些渠道 region
    enabled: bool
    expires_at: float | None
    note: str
    created_at: float
    last_used_at: float | None
    key_hash: str = field(default="", repr=False)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("key_hash", None)
        return d


class Store:
    def __init__(self, path: str):
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path, isolation_level=None)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._migrate()

    # 旧库升级：只做"加列"这类向后兼容的变更
    MIGRATIONS = {
        "requests": {"flags": "TEXT DEFAULT ''"},
        "api_keys": {"allowed_regions": "TEXT"},
    }

    async def _migrate(self) -> None:
        for table, cols in self.MIGRATIONS.items():
            existing = {r["name"] for r in await self._all(f"PRAGMA table_info({table})")}
            for col, ddl in cols.items():
                if col not in existing:
                    await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("store not connected")
        return self._db

    async def _one(self, sql: str, args: tuple | list = ()) -> aiosqlite.Row | None:
        async with self.db.execute(sql, args) as c:
            return await c.fetchone()

    async def _all(self, sql: str, args: tuple | list = ()) -> list[aiosqlite.Row]:
        async with self.db.execute(sql, args) as c:
            return list(await c.fetchall())

    # ── keys ────────────────────────────────────────────────────

    @staticmethod
    def _key(r: aiosqlite.Row) -> ApiKey:
        allowed = json.loads(r["allowed_models"]) if r["allowed_models"] else None
        regions = json.loads(r["allowed_regions"]) if r["allowed_regions"] else None
        return ApiKey(
            id=r["id"],
            name=r["name"],
            prefix=r["prefix"],
            daily_credits=r["daily_credits"],
            rpm=r["rpm"],
            max_concurrency=r["max_concurrency"],
            max_output_tokens=r["max_output_tokens"],
            plan=r["plan"],
            exhaust_mode=r["exhaust_mode"],
            allowed_models=allowed,
            allowed_regions=regions,
            enabled=bool(r["enabled"]),
            expires_at=r["expires_at"],
            note=r["note"] or "",
            created_at=r["created_at"],
            last_used_at=r["last_used_at"],
            key_hash=r["key_hash"],
        )

    @staticmethod
    def _norm(fields: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in fields.items() if k in KEY_FIELDS}
        for f_ in JSON_FIELDS:
            if f_ in out:
                v = out[f_]
                out[f_] = json.dumps(list(v)) if v else None
        if "enabled" in out:
            out["enabled"] = int(bool(out["enabled"]))
        return out

    async def create_key(self, name: str, defaults: dict[str, Any], **over: Any) -> tuple[ApiKey, str]:
        raw = "sk-thl-" + secrets.token_urlsafe(32)
        row: dict[str, Any] = {**defaults, **{k: v for k, v in over.items() if v is not None}, "name": name}
        row = self._norm(row)
        row.update(
            id=f"k_{uuid.uuid4().hex[:12]}",
            key_hash=hash_key(raw),
            prefix=raw[:12],
            created_at=time.time(),
        )
        row.setdefault("enabled", 1)
        row.setdefault("note", "")
        cols = ",".join(row)
        await self.db.execute(f"INSERT INTO api_keys ({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        key = await self.get_key(row["id"])
        assert key is not None
        return key, raw

    async def get_key(self, kid: str) -> ApiKey | None:
        r = await self._one("SELECT * FROM api_keys WHERE id=?", (kid,))
        return self._key(r) if r else None

    async def get_key_by_raw(self, raw: str) -> ApiKey | None:
        r = await self._one("SELECT * FROM api_keys WHERE key_hash=?", (hash_key(raw),))
        return self._key(r) if r else None

    async def list_keys(self) -> list[ApiKey]:
        return [self._key(r) for r in await self._all("SELECT * FROM api_keys ORDER BY created_at DESC")]

    async def update_key(self, kid: str, fields: dict[str, Any]) -> ApiKey | None:
        sets = self._norm(fields)
        if sets:
            clause = ",".join(f"{k}=?" for k in sets)
            await self.db.execute(f"UPDATE api_keys SET {clause} WHERE id=?", [*sets.values(), kid])
        return await self.get_key(kid)

    async def rotate_key(self, kid: str) -> str | None:
        if await self.get_key(kid) is None:
            return None
        raw = "sk-thl-" + secrets.token_urlsafe(32)
        await self.db.execute("UPDATE api_keys SET key_hash=?, prefix=? WHERE id=?", (hash_key(raw), raw[:12], kid))
        return raw

    async def delete_key(self, kid: str) -> bool:
        cur = await self.db.execute("DELETE FROM api_keys WHERE id=?", (kid,))
        await self.db.execute("DELETE FROM usage_daily WHERE key_id=?", (kid,))
        return cur.rowcount > 0

    async def touch_key(self, kid: str, ts: float) -> None:
        await self.db.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (ts, kid))

    # ── 每日额度 ───────────────────────────────────────────────

    async def get_usage(self, kid: str, day: str) -> dict[str, Any]:
        r = await self._one("SELECT * FROM usage_daily WHERE key_id=? AND day=?", (kid, day))
        if r is None:
            return {"used": 0.0, "reserved": 0.0, "requests": 0, "in_tokens": 0, "cached_tokens": 0, "out_tokens": 0}
        return dict(r)

    async def usage_for_day(self, day: str) -> dict[str, dict[str, Any]]:
        return {r["key_id"]: dict(r) for r in await self._all("SELECT * FROM usage_daily WHERE day=?", (day,))}

    async def usage_history(self, kid: str, limit: int = 14) -> list[dict[str, Any]]:
        rows = await self._all("SELECT * FROM usage_daily WHERE key_id=? ORDER BY day DESC LIMIT ?", (kid, limit))
        return [dict(r) for r in rows]

    async def try_reserve(self, kid: str, day: str, amount: float, ceiling: float) -> str | None:
        """原子预留：一条带条件的 UPDATE，并发请求不可能一起穿过上限。"""
        await self.db.execute("INSERT OR IGNORE INTO usage_daily (key_id, day) VALUES (?, ?)", (kid, day))
        cur = await self.db.execute(
            "UPDATE usage_daily SET reserved = reserved + ? "
            "WHERE key_id=? AND day=? AND (used + reserved + ?) <= ?",
            (amount, kid, day, amount, ceiling),
        )
        if cur.rowcount == 0:
            return None
        rid = f"r_{uuid.uuid4().hex[:16]}"
        await self.db.execute(
            "INSERT INTO reservations (id, key_id, day, amount, created_at) VALUES (?,?,?,?,?)",
            (rid, kid, day, amount, time.time()),
        )
        return rid

    async def settle(self, rid: str, credits: float, in_tokens: int, cached: int, out: int) -> None:
        r = await self._one("SELECT key_id, day, amount, state FROM reservations WHERE id=?", (rid,))
        if r is None or r["state"] != "open":
            return
        await self.db.execute(
            "UPDATE usage_daily SET reserved = MAX(reserved - ?, 0), used = used + ?, requests = requests + 1, "
            "in_tokens = in_tokens + ?, cached_tokens = cached_tokens + ?, out_tokens = out_tokens + ? "
            "WHERE key_id=? AND day=?",
            (r["amount"], credits, in_tokens, cached, out, r["key_id"], r["day"]),
        )
        await self.db.execute("UPDATE reservations SET state='settled' WHERE id=?", (rid,))

    async def release(self, rid: str) -> None:
        r = await self._one("SELECT key_id, day, amount, state FROM reservations WHERE id=?", (rid,))
        if r is None or r["state"] != "open":
            return
        await self.db.execute(
            "UPDATE usage_daily SET reserved = MAX(reserved - ?, 0) WHERE key_id=? AND day=?",
            (r["amount"], r["key_id"], r["day"]),
        )
        await self.db.execute("UPDATE reservations SET state='released' WHERE id=?", (rid,))

    async def sweep_reservations(self, older_than_s: float = 3600) -> int:
        rows = await self._all(
            "SELECT id FROM reservations WHERE state='open' AND created_at < ?", (time.time() - older_than_s,)
        )
        for r in rows:
            await self.release(r["id"])
        await self.db.execute(
            "DELETE FROM reservations WHERE state!='open' AND created_at < ?", (time.time() - 86400,)
        )
        return len(rows)

    # ── 请求日志 ───────────────────────────────────────────────

    async def log_request(self, rec: dict[str, Any]) -> None:
        cols = ",".join(rec)
        await self.db.execute(
            f"INSERT OR REPLACE INTO requests ({cols}) VALUES ({','.join('?' * len(rec))})", list(rec.values())
        )

    async def recent_requests(self, limit: int = 100, key_id: str | None = None) -> list[dict[str, Any]]:
        if key_id:
            rows = await self._all("SELECT * FROM requests WHERE key_id=? ORDER BY ts DESC LIMIT ?", (key_id, limit))
        else:
            rows = await self._all("SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    async def stats_since(self, since: float) -> dict[str, Any]:
        g = await self._one(
            "SELECT COUNT(*) n, COALESCE(SUM(credits),0) credits, COALESCE(SUM(real_usd),0) real_usd, "
            "COALESCE(AVG(ttft_ms),0) ttft_ms, SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END) errors, "
            "SUM(CASE WHEN attempts>1 THEN 1 ELSE 0 END) fallbacks, "
            "COALESCE(SUM(in_miss),0) in_miss, COALESCE(SUM(in_cached),0) in_cached, COALESCE(SUM(out_tokens),0) out_tokens "
            "FROM requests WHERE ts >= ?",
            (since,),
        )
        by_offer = await self._all(
            "SELECT model, channel, COUNT(*) n, COALESCE(SUM(credits),0) credits, COALESCE(SUM(real_usd),0) real_usd, "
            "COALESCE(SUM(units),0) units, COALESCE(AVG(ttft_ms),0) ttft_ms "
            "FROM requests WHERE ts >= ? AND status='ok' GROUP BY model, channel ORDER BY n DESC",
            (since,),
        )
        by_task = await self._all(
            "SELECT task, difficulty, COUNT(*) n FROM requests WHERE ts >= ? GROUP BY task, difficulty ORDER BY n DESC",
            (since,),
        )
        return {
            "global": dict(g) if g else {},
            "by_offer": [dict(r) for r in by_offer],
            "by_task": [dict(r) for r in by_task],
        }

    async def add_flag(self, request_id: str, flag: str) -> None:
        """事后给某条请求补标记（例如下一轮发现用户在纠正它）。"""
        await self.db.execute(
            "UPDATE requests SET flags = CASE WHEN flags IS NULL OR flags = '' THEN ? "
            "WHEN (',' || flags || ',') LIKE ('%,' || ? || ',%') THEN flags ELSE flags || ',' || ? END WHERE id = ?",
            (flag, flag, flag, request_id),
        )

    async def outcome_stats(self, since: float) -> list[dict[str, Any]]:
        from .outcome import ALL_FLAGS

        cols = ", ".join(
            f"SUM(CASE WHEN (',' || COALESCE(flags,'') || ',') LIKE '%,{f},%' THEN 1 ELSE 0 END) AS {f}" for f in ALL_FLAGS
        )
        rows = await self._all(
            f"SELECT model, task, COUNT(*) n, {cols} FROM requests WHERE ts >= ? AND model IS NOT NULL "
            "GROUP BY model, task ORDER BY n DESC",
            (since,),
        )
        return [dict(r) for r in rows]

    async def latency_stats(self, since: float, min_n: int = 5) -> list[dict[str, Any]]:
        """每个 offer 实测的首 token 与出字速度，用于重启后的冷启动。

        必须按 (model, channel) 分组而不是只按 model：同一个模型在不同渠道上的
        网络距离可能差一个数量级（荷兰 → 北京 vs 荷兰 → 美国），而 upstreams.yaml
        里的 ttft_s/tps 先验是跟模型走的、不区分渠道。只有实测能分辨这件事。
        """
        rows = await self._all(
            "SELECT model, channel, COUNT(*) n, AVG(ttft_ms) ttft_ms, "
            # 出字速度只能从"首 token 之后"的那段算，否则等待时间会被算进吞吐里
            "SUM(out_tokens) out_tokens, SUM(duration_ms - ttft_ms) gen_ms "
            "FROM requests WHERE ts >= ? AND status='ok' AND model IS NOT NULL "
            "AND ttft_ms > 0 AND duration_ms > ttft_ms AND out_tokens > 0 "
            "GROUP BY model, channel HAVING COUNT(*) >= ?",
            (since, min_n),
        )
        return [dict(r) for r in rows]

    async def prune_requests(self, older_than_s: float) -> None:
        await self.db.execute("DELETE FROM requests WHERE ts < ?", (time.time() - older_than_s,))

    # ── 订阅窗口 / 按量花费 / 学习 ─────────────────────────────

    async def add_channel_usage(self, scope: str, ts: float, units: float) -> None:
        await self.db.execute(
            "INSERT INTO channel_usage (scope, minute, units) VALUES (?,?,?) "
            "ON CONFLICT(scope, minute) DO UPDATE SET units = units + excluded.units",
            (scope, int(ts // 60), units),
        )

    async def load_channel_usage(self, since_ts: float) -> list[tuple[str, float, float]]:
        rows = await self._all(
            "SELECT scope, minute, units FROM channel_usage WHERE minute >= ? ORDER BY minute", (int(since_ts // 60),)
        )
        return [(r["scope"], r["minute"] * 60.0, r["units"]) for r in rows]

    async def prune_channel_usage(self, before_ts: float) -> None:
        await self.db.execute("DELETE FROM channel_usage WHERE minute < ?", (int(before_ts // 60),))

    async def add_metered(self, day: str, channel: str, usd: float) -> None:
        await self.db.execute(
            "INSERT INTO metered_daily (day, channel, usd) VALUES (?,?,?) "
            "ON CONFLICT(day, channel) DO UPDATE SET usd = usd + excluded.usd",
            (day, channel, usd),
        )

    async def metered_for_day(self, day: str) -> dict[str, float]:
        return {r["channel"]: r["usd"] for r in await self._all("SELECT channel, usd FROM metered_daily WHERE day=?", (day,))}

    async def load_learned(self) -> list[tuple[str, str, float]]:
        return [(r["model"], r["task"], r["delta"]) for r in await self._all("SELECT model, task, delta FROM learned_quality")]

    async def save_learned(self, rows: list[tuple[str, str, float]]) -> None:
        now = time.time()
        for m, t, d in rows:
            await self.db.execute(
                "INSERT INTO learned_quality (model, task, delta, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(model, task) DO UPDATE SET delta=excluded.delta, updated_at=excluded.updated_at",
                (m, t, d, now),
            )

    async def backup(self, dest: str) -> None:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        await self.db.execute("VACUUM INTO ?", (dest,))

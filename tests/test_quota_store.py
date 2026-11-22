import asyncio

import pytest
from helpers import load_test_settings
from thalamus.env import load_env
from thalamus.quota import Ledger
from thalamus.store import Store

DEFAULTS = {"daily_credits": 100, "rpm": 60, "max_concurrency": 4, "max_output_tokens": 0, "plan": "standard", "exhaust_mode": "economy"}


def test_latency_seed_survives_restart_and_separates_channels(tmp_path):
    """重启不该把实测延迟丢掉：upstreams.yaml 的 ttft_s/tps 先验跟模型走、不区分渠道，
    分辨不出"荷兰 → 北京"和"荷兰 → 美国"差的那两百毫秒，而等待成本与死线全靠这两个数。"""
    from thalamus.health import HealthBook

    async def run():
        st = Store(str(tmp_path / "lat.db"))
        await st.connect()
        now = 1_800_000_000.0
        for i in range(8):
            for ch, ttft, tps in (("deepseek", 1150.0, 40.0), ("opencode_go", 310.0, 78.0)):
                await st.log_request({
                    "id": f"r-{ch}-{i}", "key_id": "k", "ts": now - 600, "task": "chat", "difficulty": 1,
                    "model": "deepseek-v4.1-flash", "channel": ch, "ttft_ms": ttft,
                    "duration_ms": ttft + 400 / tps * 1000, "out_tokens": 400, "status": "ok",
                })
        # 样本不够的、失败的、非流式没有 ttft 的，都不该进来
        await st.log_request({"id": "thin", "key_id": "k", "ts": now - 600, "model": "kimi-k2.7-code",
                              "channel": "opencode_go", "ttft_ms": 5000.0, "duration_ms": 9000.0,
                              "out_tokens": 100, "status": "ok"})
        await st.log_request({"id": "bad", "key_id": "k", "ts": now - 600, "model": "deepseek-v4.1-flash",
                              "channel": "deepseek", "ttft_ms": 60000.0, "duration_ms": 60001.0,
                              "out_tokens": 1, "status": "error"})
        rows = await st.latency_stats(now - 3 * 86400)
        await st.close()
        return rows

    rows = asyncio.run(run())
    assert {r["channel"] for r in rows} == {"deepseek", "opencode_go"}, rows
    assert all(r["n"] == 8 for r in rows), rows  # 失败那条没被算进来，样本不足那组被 HAVING 挡掉

    s = load_test_settings(tmp_path)
    hb = HealthBook(s.policy)
    cn, us = s.offer("deepseek-v4.1-flash@deepseek"), s.offer("deepseek-v4.1-flash@opencode_go")
    assert hb.ttft(cn) == hb.ttft(us), "seed 之前两个渠道只能共用模型先验"
    hb.seed(rows)
    assert hb.ttft(cn) == pytest.approx(1.15, abs=0.01)
    assert hb.ttft(us) == pytest.approx(0.31, abs=0.01)
    assert hb.tps(cn) == pytest.approx(40, rel=0.02) and hb.tps(us) == pytest.approx(78, rel=0.02)
    # 没有实测的 offer 继续用先验，不会被别人的数字污染
    k = s.offer("kimi-k2.7-code@opencode_go")
    assert hb.ttft(k) == k.model.ttft_s


def test_concurrent_reservations_never_exceed_ceiling(tmp_path):
    async def run():
        st = Store(str(tmp_path / "q.db"))
        await st.connect()
        key, _ = await st.create_key("a", DEFAULTS)
        results = await asyncio.gather(*(st.try_reserve(key.id, "2026-01-01", 10, 150) for _ in range(20)))
        ok = [r for r in results if r]
        assert len(ok) == 15
        u = await st.get_usage(key.id, "2026-01-01")
        assert u["reserved"] == 150
        await st.settle(ok[0], 3.0, 100, 50, 20)
        await st.settle(ok[0], 3.0, 100, 50, 20)  # 重复结算无效
        await st.release(ok[1])
        u = await st.get_usage(key.id, "2026-01-01")
        assert u["used"] == 3.0 and u["reserved"] == 130 and u["requests"] == 1
        await st.close()

    asyncio.run(run())


def test_ledger_modes(tmp_path):
    async def run():
        s = load_test_settings(tmp_path)
        st = Store(s.db_path)
        await st.connect()
        led = Ledger(st, s)
        key, raw = await st.create_key("a", DEFAULTS)
        assert (await st.get_key_by_raw(raw)).id == key.id
        day = led.today()
        assert (await led.state(key, day)).mode == "normal"
        rid = await led.reserve(key, 85, "normal", day)
        led.settle(rid, 85, 0, 0, 0)
        assert (await led.state(key, day)).mode == "soft"
        rid = await led.reserve(key, 20, "normal", day)
        assert rid is None  # 超过 100%
        rid = await led.reserve(key, 20, "economy", day)
        assert rid is not None  # 经济档有 15% 宽限
        led.settle(rid, 20, 0, 0, 0)
        assert (await led.state(key, day)).mode == "economy"
        rid = await led.reserve(key, 20, "economy", day)
        assert rid is None
        key.exhaust_mode = "reject"
        assert (await led.state(key, day)).mode == "reject"
        await st.close()

    asyncio.run(run())


def test_memory_ledger_concurrency_and_persistence(tmp_path):
    """内存账本：并发预留恰好不超额；批量写回后重启进程（新 Ledger）能读回用量。"""
    from thalamus.persist import WriteBehind

    async def run():
        s = load_test_settings(tmp_path)
        st = Store(s.db_path)
        await st.connect()
        key, _ = await st.create_key("a", {**DEFAULTS, "daily_credits": 150})
        led = Ledger(st, s)
        wb = WriteBehind(s.db_path, led, interval=3600)
        await wb.start()
        day = led.today()
        rids = await asyncio.gather(*(led.reserve(key, 10, "normal", day) for _ in range(20)))
        ok = [r for r in rids if r]
        assert len(ok) == 15
        for r in ok[:5]:
            led.settle(r, 3.0, 100, 50, 20)
            led.settle(r, 3.0, 100, 50, 20)  # 重复结算无效
        for r in ok[5:]:
            led.release(r)
        u = await led.usage(key.id, day)
        assert u["used"] == 15.0 and u["reserved"] == 0 and u["requests"] == 5
        wb.log_request({"id": "req1", "key_id": key.id, "ts": 1.0, "status": "ok", "flags": ""})
        wb.add_flag("req1", "user_correction")
        await wb.flush()
        await wb.stop()

        fresh = Ledger(st, s)  # 模拟重启
        u2 = await fresh.usage(key.id, day)
        assert u2["used"] == 15.0 and u2["requests"] == 5 and u2["out_tokens"] == 100
        row = await st._one("SELECT flags FROM requests WHERE id='req1'")
        assert row["flags"] == "user_correction"
        await st.close()

    asyncio.run(run())


def test_env_loader(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('A_X=1   # comment\nB_X="quoted # not comment"\nexport C_X=hello\n# D_X=no\nE_X=has#hash\n', encoding="utf-8")
    for k in ("A_X", "B_X", "C_X", "D_X", "E_X"):
        monkeypatch.delenv(k, raising=False)
    load_env(p)
    import os

    assert os.environ["A_X"] == "1"
    assert os.environ["B_X"] == "quoted # not comment"
    assert os.environ["C_X"] == "hello"
    assert "D_X" not in os.environ
    assert os.environ["E_X"] == "has#hash"

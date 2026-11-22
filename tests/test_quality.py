"""结果信号、隐式反馈、对冲、指标、数据库迁移。"""
import asyncio
import sqlite3
import time

from helpers import ADMIN, MockUpstream, body, gateway, load_test_settings, new_key
from thalamus import outcome as oc
from thalamus.intent import is_correction
from thalamus.store import Store
from thalamus.types import Delta


def run(coro):
    asyncio.run(coro)


def auth(k, **extra):
    return {"Authorization": f"Bearer {k}", **extra}


# ── 纯函数 ─────────────────────────────────────────────────────


def test_outcome_tracker_flags():
    t = oc.OutcomeTracker(json_expected=False)
    t.feed(Delta(tool_calls=[{"index": 0, "function": {"name": "f", "arguments": "{\"a\": "}}]))
    t.feed(Delta(tool_calls=[{"index": 0, "function": {"arguments": "1"}}]))
    assert oc.TOOL_ARGS_INVALID in t.flags("tool_calls", "ok")

    t = oc.OutcomeTracker(json_expected=False)
    t.feed(Delta(tool_calls=[{"index": 0, "function": {"name": "f", "arguments": ""}}]))  # 无参工具
    assert t.flags("tool_calls", "ok") == []

    t = oc.OutcomeTracker(json_expected=True)
    t.feed(Delta(content='```json\n{"ok": true}\n```'))
    assert t.flags("stop", "ok") == []
    t = oc.OutcomeTracker(json_expected=True)
    t.feed(Delta(content="Sure! here is the JSON: {ok}"))
    assert t.flags("stop", "ok") == [oc.JSON_INVALID]

    t = oc.OutcomeTracker(json_expected=True)
    t.feed(Delta(reasoning="thinking only"))
    assert t.flags("stop", "ok") == [oc.EMPTY]
    t = oc.OutcomeTracker(json_expected=True)
    t.feed(Delta(content='{"a": '))
    assert t.flags("length", "ok") == [oc.TRUNCATED]  # 被截断的不据此判定 JSON


def test_correction_detection_multilingual():
    for s in ("不对，还是报错", "还是不行啊", "That's wrong, try again", "it still doesn't work",
              "Nee, dat klopt niet", "Het werkt niet, nog steeds dezelfde fout"):
        assert is_correction(s), s
    for s in ("好的谢谢", "对了，顺便问一下", "What is wrong with this code? ```py\nx=1```", "werkt dit ook op Windows?"):
        # "What is wrong with..." 是新问题的常见开头；这里接受它被识别为纠正（只在同会话紧接上一轮时才生效）
        if s.startswith("What is wrong"):
            continue
        assert not is_correction(s), s


def test_store_migration_adds_flags_column(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, key_id TEXT, ts REAL NOT NULL, task TEXT)")
    con.execute("INSERT INTO requests (id, ts) VALUES ('r1', 1)")
    con.commit()
    con.close()

    async def go():
        st = Store(str(path))
        await st.connect()
        cols = {r["name"] for r in await st._all("PRAGMA table_info(requests)")}
        assert "flags" in cols
        await st.add_flag("r1", "user_correction")
        await st.add_flag("r1", "user_correction")
        await st.add_flag("r1", "regenerated")
        row = await st._one("SELECT flags FROM requests WHERE id='r1'")
        assert row["flags"] == "user_correction,regenerated"
        await st.close()

    run(go())


# ── 端到端 ─────────────────────────────────────────────────────


def test_invalid_tool_args_are_flagged_and_learned(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(kind="badtools")
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("巴黎天气", model="deepseek-v4.1-flash", tools=tools), headers=auth(k))
            assert r.status_code == 200
            await app.state.engine.drain()
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert "tool_args_invalid" in logs[0]["flags"]
            deltas = {(d["model"], d["task"]): d["delta"] for d in app.state.engine.learner.snapshot()}
            assert deltas[("deepseek-v4.1-flash", "agent")] < 0
            ov = (await client.get("/admin/api/overview", headers=ADMIN)).json()
            assert ov["outcomes"][0]["tool_args_invalid"] == 1

    run(go())


def test_json_mode_violation_flagged(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            k = await new_key(client)
            r = await client.post(
                "/v1/chat/completions",
                json=body("classify sentiment", model="deepseek-v4.1-flash", response_format={"type": "json_object"}),
                headers=auth(k),
            )
            assert r.status_code == 200
            await app.state.engine.drain()
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert logs[0]["flags"] == "json_invalid"

    run(go())


def test_correction_escalates_and_flags_previous_request(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            k = await new_key(client)
            H = auth(k, **{"X-Session-Id": "conv-1"})
            first = {"model": "thalamus/auto", "messages": [{"role": "user", "content": "python 怎么读文件"}]}
            r1 = await client.post("/v1/chat/completions", json=first, headers=H)
            assert r1.headers["x-thalamus-task"] == "code/1"
            first_id, first_model = r1.headers["x-thalamus-request-id"], r1.headers["x-thalamus-model"]
            await app.state.engine.drain()

            second = {"model": "thalamus/auto", "messages": [
                *first["messages"], {"role": "assistant", "content": "用 open()"},
                {"role": "user", "content": "不对，还是报错"}]}
            r2 = await client.post("/v1/chat/completions", json=second, headers=H)
            assert r2.headers["x-thalamus-task"] == "code/2"  # 沿用任务并升一级难度
            await app.state.engine.drain()
            logs = {x["id"]: x for x in (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]}
            assert "user_correction" in logs[first_id]["flags"]
            deltas = {(d["model"], d["task"]): d["delta"] for d in app.state.engine.learner.snapshot()}
            assert deltas[(first_model, "code")] < 0

    run(go())


def test_regenerate_flags_previous_request(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            k = await new_key(client)
            H = auth(k, **{"X-Session-Id": "conv-2"})
            b = body("写一首关于海的诗")
            r1 = await client.post("/v1/chat/completions", json=b, headers=H)
            await app.state.engine.drain()
            r2 = await client.post("/v1/chat/completions", json=b, headers=H)
            assert r2.headers["x-thalamus-model"] != r1.headers["x-thalamus-model"]  # 换一个模型给不同答案
            await app.state.engine.drain()
            logs = {x["id"]: x for x in (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]}
            assert "regenerated" in logs[r1.headers["x-thalamus-request-id"]]["flags"]

    run(go())


def test_hedge_wins_when_first_upstream_is_slow(tmp_path):
    async def go():
        s = load_test_settings(tmp_path)
        s.policy.hedge_min_delay_s = 0.2
        s.policy.hedge_ttft_factor = 0.0
        mock = MockUpstream()
        mock.on(kind="text", delay=2.0, times=1)  # 第一个发出的请求很慢
        async with gateway(tmp_path, mock, s) as (app, client, _):
            k = await new_key(client)
            t0 = time.perf_counter()
            r = await client.post("/v1/chat/completions", json=body("你好", model="deepseek-v4.1-flash"), headers=auth(k))
            elapsed = time.perf_counter() - t0
            assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Hello world"
            assert r.headers.get("x-thalamus-hedged") == "1" and r.headers["x-thalamus-attempts"] == "2"
            assert elapsed < 1.5, elapsed
            e = app.state.engine
            await e.drain()
            assert sum(e.channel_load.values()) == 0 and all(v == 0 for v in e.capacity._inflight.values())
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert "hedged" in logs[0]["flags"]

    run(go())


def test_no_hedge_for_code_tasks(tmp_path):
    async def go():
        s = load_test_settings(tmp_path)
        s.policy.hedge_min_delay_s = 0.1
        s.policy.hedge_ttft_factor = 0.0
        mock = MockUpstream()
        mock.on(kind="text", delay=0.5, times=1)
        async with gateway(tmp_path, mock, s) as (app, client, _):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("python 怎么读文件", model="deepseek-v4.1-flash"), headers=auth(k))
            assert r.status_code == 200 and "x-thalamus-hedged" not in r.headers and r.headers["x-thalamus-attempts"] == "1"

    run(go())


def test_metrics_endpoint(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            assert (await client.get("/metrics")).status_code == 401
            assert (await client.get("/metrics", headers={"Authorization": "Bearer nope"})).status_code == 401
            k = await new_key(client)
            await client.post("/v1/chat/completions", json=body("你好", model="deepseek-v4.1-flash"), headers=auth(k))
            await app.state.engine.drain()
            r = await client.get("/metrics", headers=ADMIN)
            assert r.status_code == 200
            text = r.text
            assert 'thalamus_requests_total{model="deepseek-v4.1-flash",status="ok",task="chat"} 1' in text
            assert "thalamus_ttft_seconds_bucket" in text and "thalamus_routing_overhead_seconds_count 1" in text
            assert "thalamus_subscription_window_ratio" in text

    run(go())

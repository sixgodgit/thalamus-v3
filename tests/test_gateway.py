"""端到端：真实配置 + 假上游 + ASGI 客户端。"""
import asyncio

from helpers import ADMIN, MockUpstream, body, gateway, load_test_settings, new_key, parse_sse


def auth(k):
    return {"Authorization": f"Bearer {k}"}


def run(coro):
    asyncio.run(coro)


GLM_THINKING_400 = (
    '{"error":{"message":"The `reasoning_content` in the thinking mode must be passed back to the API. '
    '(request_id: ade94daf-b174-43d2-a898-d778c401d94b)","type":"invalid_request_error",'
    '"param":null,"code":"invalid_request_error"}}'
)

# 第二轮对话：客户端把历史发回来，但 assistant 消息里**没有** reasoning_content。
# 大多数 webui 都是这样 —— 它不是 OpenAI 规范里的字段，没人存。
SECOND_TURN = {
    "model": "thalamus/auto", "stream": False,
    "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        {"role": "user", "content": "再说一遍"},
    ],
}


def test_thinking_models_excluded_when_client_drops_reasoning_content(tmp_path):
    """线上故障（2026-10-12）：webui 第二轮起反复 400
      "The reasoning_content in the thinking mode must be passed back to the API"

    请求本身没问题，是这个模型吃不下它 —— 和"不支持图片"同一类能力不匹配。
    所以要在路由时就排掉要求回传的模型，而不是把 400 转给客户端。
    """

    async def go():
        mock = MockUpstream()
        # 要求回传的模型一律 400；真走到它就说明路由没挡住
        mock.on(model="glm-5.3", kind="error", status=400, body=GLM_THINKING_400)
        mock.on(model="glm-5.3-flash", kind="error", status=400, body=GLM_THINKING_400)
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=SECOND_TURN, headers=auth(k))
            assert r.status_code == 200, r.text
            served = r.headers["x-thalamus-model"]
            assert not app.state.settings.models[served].needs_reasoning_echo, served
            assert all(c["model"] not in ("glm-5.3", "glm-5.3-flash") for c in mock.calls), mock.calls

            # 客户端确实回传了 → 不再排除，思考模型重新可用
            echoed = {**SECOND_TURN, "messages": [
                dict(m, reasoning_content="（上一轮的思考）") if m["role"] == "assistant" else m
                for m in SECOND_TURN["messages"]]}
            plan = await client.post("/admin/api/route-preview", json=echoed, headers=ADMIN)
            assert plan.status_code == 200, plan.text
            ranked = {c["model"] for c in plan.json()["ranked"]}
            assert "glm-5.3" in ranked, plan.json()["rejected"]

            # 第一轮（历史里还没有 assistant）也不该被排除
            first = await client.post("/admin/api/route-preview", headers=ADMIN,
                                      json={"messages": [{"role": "user", "content": "你好"}]})
            assert "glm-5.3" in {c["model"] for c in first.json()["ranked"]}

    run(go())


def test_model_incompatible_400_switches_model_instead_of_failing(tmp_path):
    """兜底：万一判断失准、400 还是回来了，也要换个模型重做，而不是把 400 交出去。
    并且同一个模型在别的渠道上还是同一个模型，不该再白试一遍。"""

    async def go():
        mock = MockUpstream()
        # 点名 glm-5.3 绕过上面那条硬过滤，直接制造"400 已经发生了"的局面
        mock.on(model="glm-5.3", kind="error", status=400, body=GLM_THINKING_400)
        s = load_test_settings(tmp_path)
        s.models["glm-5.3"].needs_reasoning_echo = False  # 故意让硬过滤失效
        # 让质量说话，这样 glm-5.3(chat 8.3) 排在 deepseek-flash(7.6) 前面，400 才会真的发生
        s.policy.tasks["chat"].usd_per_sec = 0.0
        s.policy.tasks["chat"].usd_per_point = [0.01, 0.01, 0.01]
        async with gateway(tmp_path, mock, settings=s) as (app, client, _):
            k = await new_key(client, allowed_models=["glm-5.3", "deepseek-v4.1-flash"])
            r = await client.post("/v1/chat/completions", json=SECOND_TURN, headers=auth(k))
            assert r.status_code == 200, r.text
            assert r.headers["x-thalamus-model"] != "glm-5.3"
            # glm-5.3 有两个渠道（glm_plan / opencode_go），只该被试一次
            assert sum(1 for c in mock.calls if c["model"] == "glm-5.3") == 1, mock.calls
            # 能力不匹配不算上游有毛病：不进错误率、不熔断
            h = app.state.engine.health.snapshot()
            for key in ("glm-5.3@glm_plan", "glm-5.3@opencode_go"):
                assert h.get(key, {}).get("error_rate", 0) == 0, (key, h.get(key))
                assert h.get(key, {}).get("state", "closed") == "closed", (key, h.get(key))

    run(go())


def test_startup_failure_closes_the_db_instead_of_hanging_exit(tmp_path, monkeypatch):
    """aiosqlite 的连接跑在非守护线程上，不关掉解释器就退不出去。
    所以启动中途失败必须先关连接再抛 —— 否则一个配置问题会表现成
    "systemctl 卡住 90 秒后被 SIGKILL"，真正的报错根本看不到。
    （这条是 2026-10-12 用一个临时脚本踩出来的：它少关了一次连接，进程就再也没退出。）
    """
    import threading

    from thalamus.app import create_app
    from thalamus.engine import Engine

    async def boom(self):
        raise RuntimeError("假装 data/ 没有写权限")

    monkeypatch.setattr(Engine, "start", boom)
    s = load_test_settings(tmp_path)
    s.db_path = str(tmp_path / "startup.db")
    app = create_app(s, warm=False)

    before = {t.name for t in threading.enumerate()}
    try:
        asyncio.run(app.state.startup())
        raise AssertionError("应该抛 RuntimeError")
    except RuntimeError as e:
        assert "写权限" in str(e)
    leaked = [t for t in threading.enumerate()
              if t.name not in before and t.is_alive() and not t.daemon]
    assert not leaked, f"连接线程没被关掉，进程将无法退出：{[t.name for t in leaked]}"


def test_shutdown_closes_the_db_even_if_an_earlier_step_raises(tmp_path, monkeypatch):
    """关闭时前一步抛异常不能连带让数据库连接关不掉 —— 同样会卡住退出、被 SIGKILL，
    而正常关闭本来是要先刷盘的（刷盘失败只丢 250ms，卡住退出丢的是整个缓冲）。

    这里挑连接池那一步来坏：它没有后台任务，engine.stop 仍然正常刷盘收尾，
    所以测的是"后面的步骤照样会走到"，而不是顺手造出一堆孤儿任务。
    """

    async def go():
        async with gateway(tmp_path) as (app, client, _):
            k = await new_key(client)
            assert (await client.post("/v1/chat/completions", json=body("hi"), headers=auth(k))).status_code == 200

            async def boom():
                raise RuntimeError("连接池关闭时炸了")

            monkeypatch.setattr(app.state.pool, "aclose", boom)
        # gateway() 退出时会调 shutdown()；pool 那步抛了，但 store 仍然必须被关掉
        assert app.state.store._db is None, "pool.aclose 失败后数据库连接没被关掉"

    run(go())


def test_nonstream_and_stream_basic(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("你好", model="deepseek-v4.1-flash"), headers=auth(k))
            assert r.status_code == 200, r.text
            j = r.json()
            assert j["choices"][0]["message"]["content"] == "Hello world"
            assert r.headers["x-thalamus-model"] == "deepseek-v4.1-flash"
            assert j["usage"]["prompt_tokens_details"]["cached_tokens"] == 60

            r = await client.post(
                "/v1/chat/completions",
                json=body("你好", model="deepseek-v4.1-flash", stream=True, stream_options={"include_usage": True}),
                headers=auth(k),
            )
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            ev = parse_sse(r.text)
            assert ev[-1] == "[DONE]"
            text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1] if isinstance(e, dict) and e.get("choices"))
            assert text == "Hello world"
            assert ev[0]["choices"][0]["delta"]["role"] == "assistant"
            assert any(isinstance(e, dict) and e.get("usage") for e in ev)

            e = app.state.engine
            await e.drain()
            keys = (await client.get("/admin/api/keys", headers=ADMIN)).json()["keys"]
            assert keys[0]["requests_today"] == 2 and keys[0]["used_today"] > 0 and keys[0]["reserved_today"] == 0
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert {x["status"] for x in logs} == {"ok"}

    run(go())


def test_fallback_before_first_token(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(kind="error", status=500, times=1)
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("你好", model="deepseek-v4.1-flash"), headers=auth(k))
            assert r.status_code == 200, r.text
            assert r.headers["x-thalamus-attempts"] == "2"
            assert len({c["host"] for c in mock.calls}) == 2

    run(go())


def test_first_token_timeout_fallback(tmp_path):
    async def go():
        s = load_test_settings(tmp_path)
        for t in s.policy.tasks.values():
            t.first_token_timeout_s = 0.3
        mock = MockUpstream()
        mock.on(kind="hang", times=1)
        async with gateway(tmp_path, mock, s) as (app, client, _):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("你好", model="deepseek-v4.1-flash"), headers=auth(k))
            assert r.status_code == 200 and r.headers["x-thalamus-attempts"] == "2"
            snap = app.state.engine.health.snapshot()
            assert any("first_token_timeout" in h["last_error"] for h in snap.values())

    run(go())


def test_tool_calls_openai_and_anthropic(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(kind="tools")
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client)
            for model in ("deepseek-v4.1-flash", "minimax-m3"):
                r = await client.post("/v1/chat/completions", json=body("巴黎天气", model=model, tools=tools), headers=auth(k))
                assert r.status_code == 200, r.text
                msg = r.json()["choices"][0]
                assert msg["finish_reason"] == "tool_calls"
                tc = msg["message"]["tool_calls"][0]
                assert tc["function"] == {"name": "get_weather", "arguments": '{"city":"Paris"}'} and tc["id"]
            anth = [c for c in mock.calls if c["path"].endswith("/messages")][0]
            assert anth["payload"]["tools"][0]["input_schema"] == {"type": "object"}
            assert anth["headers"]["x-opencode-session"]

    run(go())


def test_anthropic_text_think_tags_removed(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("hi", model="minimax-m3"), headers=auth(k))
            m = r.json()["choices"][0]["message"]
            assert m["content"] == "Hello world" and m["reasoning_content"] == "plan"

    run(go())


def test_vision_request_keeps_image(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            k = await new_key(client)
            content = [{"type": "text", "text": "图里是什么"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
            r = await client.post("/v1/chat/completions", json=body(content), headers=auth(k))
            assert r.status_code == 200
            assert r.headers["x-thalamus-model"] == "deepseek-v4.1-flash"
            assert r.headers["x-thalamus-task"].startswith("vision")
            assert mock.calls[-1]["payload"]["messages"][0]["content"] == content

    run(go())


def test_daily_quota_exhaustion(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(out_tokens=2000)
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client, daily_credits=3.0, exhaust_mode="reject")
            codes = []
            for _ in range(10):
                r = await client.post("/v1/chat/completions", json=body("你好", model="deepseek-v4.1-flash"), headers=auth(k))
                codes.append(r.status_code)
                await app.state.engine.drain()
                if r.status_code == 429:
                    assert r.json()["error"]["type"] == "quota_exceeded" and "retry-after" in r.headers
                    break
            assert codes[0] == 200 and codes[-1] == 429

    run(go())


def test_rpm_limit(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, _):
            k = await new_key(client, rpm=2)
            codes = [
                (await client.post("/v1/chat/completions", json=body("hi"), headers=auth(k))).status_code for _ in range(3)
            ]
            assert codes == [200, 200, 429]

    run(go())


def test_client_disconnect_still_settles(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(kind="stall")
        async with gateway(tmp_path, mock) as (app, client, _):
            e = app.state.engine
            k = await new_key(client)
            key = await e.authenticate(k)
            call = await e.open(body("你好", model="deepseek-v4.1-flash", stream=True), key, stream=True)
            gen = call.sse(False)
            first = await gen.__anext__()
            assert "Hello" in first
            await gen.aclose()
            await e.drain()
            assert call.finalized and call.status == "client_closed"
            u = await e.store.get_usage(key.id, e.ledger.today())
            assert u["reserved"] == 0 and u["used"] > 0 and u["requests"] == 1
            assert e.limiter.inflight(key.id) == 0 and sum(e.channel_load.values()) == 0

    run(go())


def test_subscription_quota_error_marks_capacity(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(channel="opencode_go", kind="error", status=429, body="usage limit reached for 5 hour window")
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("写个快排", model="kimi-k2.7-code"), headers=auth(k))
            assert r.status_code == 502
            cap = app.state.engine.capacity.snapshot()
            assert cap["opencode_go"]["exhausted_until"] > 0
            pv = await client.post("/admin/api/route-preview", json=body("写个快排", model="glm-5.3"), headers=ADMIN)
            rej = {c["offer"]: c["reject"] for c in pv.json()["rejected"]}
            assert "额度" in rej["glm-5.3@opencode_go"]
            assert "glm-5.3@glm_plan" in {c["offer"] for c in pv.json()["ranked"]}

    run(go())


def test_breaker_opens_after_repeated_failures(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(channel="deepseek", model="deepseek-v4-pro", kind="error", status=500)
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client, rpm=100)
            codes = []
            for _ in range(4):
                codes.append((await client.post("/v1/chat/completions", json=body("hi", model="deepseek-v4-pro"), headers=auth(k))).status_code)
            assert codes[:3] == [502, 502, 502] and codes[3] == 503
            assert len(mock.calls) == 3

    run(go())


def test_midstream_failure_reports_error_event(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(kind="midfail")
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("hi", model="deepseek-v4.1-flash", stream=True), headers=auth(k))
            ev = parse_sse(r.text)
            assert any(isinstance(x, dict) and "error" in x for x in ev) and ev[-1] == "[DONE]"
            await app.state.engine.drain()
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert logs[0]["status"] == "error"

    run(go())


def test_admin_and_key_lifecycle(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, _):
            assert (await client.get("/admin/api/keys")).status_code == 401
            assert (await client.post("/admin/api/login", json={"token": "wrong"})).status_code == 401
            r = await client.post("/admin/api/login", json={"token": "admin-test-token"})
            assert r.status_code == 200
            sid = r.headers["set-cookie"].split(";")[0].split("=", 1)[1]
            assert "httponly" in r.headers["set-cookie"].lower()
            C = {"Cookie": f"thl_admin={sid}"}

            r = await client.post("/admin/api/keys", json={"name": "张三", "daily_credits": 800, "plan": "pro", "expires_days": 30}, headers=C)
            assert r.status_code == 201
            kid, raw = r.json()["key"]["id"], r.json()["api_key"]
            assert raw.startswith("sk-thl-") and r.json()["key"]["expires_at"]
            models = (await client.get("/v1/models", headers=auth(raw))).json()["data"]
            assert "kimi-k3" in {m["id"] for m in models}
            # 别名也必须报出上下文与输出上限，否则客户端只能用自己的默认值（常见 131072 = 131.1k）
            by_id = {m["id"]: m for m in models}
            biggest = max(m["context_length"] for m in models if not m["id"].startswith("thalamus/"))
            assert biggest == 1_000_000
            for name in ("thalamus/auto", "thalamus/fast", "thalamus/pro", "thalamus/code", "thalamus/long"):
                a = by_id[name]
                assert a["context_length"] == a["context_window"] == biggest, (name, a)
                assert a["max_output_tokens"] > 0, a
            assert by_id["thalamus/vision"]["context_length"] <= biggest  # 只统计能读图的模型

            r = await client.patch(f"/admin/api/keys/{kid}", json={"plan": "standard", "rpm": 5}, headers=C)
            assert r.json()["key"]["plan"] == "standard"
            models = (await client.get("/v1/models", headers=auth(raw))).json()["data"]
            assert "kimi-k3" not in {m["id"] for m in models}

            assert (await client.patch(f"/admin/api/keys/{kid}", json={"plan": "gold"}, headers=C)).status_code == 400
            assert (await client.patch(f"/admin/api/keys/{kid}", json={"enabled": False}, headers=C)).status_code == 200
            assert (await client.get("/v1/models", headers=auth(raw))).status_code == 403
            await client.patch(f"/admin/api/keys/{kid}", json={"enabled": True}, headers=C)

            new_raw = (await client.post(f"/admin/api/keys/{kid}/rotate", headers=C)).json()["api_key"]
            assert (await client.get("/v1/models", headers=auth(raw))).status_code == 401
            assert (await client.get("/v1/models", headers=auth(new_raw))).status_code == 200

            for path in ("/admin/api/overview", "/admin/api/channels", "/admin/api/learning", "/admin/api/config", f"/admin/api/keys/{kid}/usage"):
                assert (await client.get(path, headers=C)).status_code == 200, path
            pv = await client.post("/admin/api/route-preview", json={"messages": [{"role": "user", "content": "写个 python 爬虫"}], "key_id": kid}, headers=C)
            assert pv.json()["intent"]["task"] == "code" and pv.json()["ranked"]
            rl = await client.post("/admin/api/reload", headers=C)
            assert rl.status_code == 200 and rl.json()["reloaded"]
            assert (await client.post("/admin/api/logout", headers=C)).status_code == 200
            assert (await client.get("/admin/api/keys", headers=C)).status_code == 401
            assert (await client.delete(f"/admin/api/keys/{kid}", headers=ADMIN)).status_code == 200
            assert (await client.get("/v1/models", headers=auth(new_raw))).status_code == 401
            assert (await client.get("/admin")).status_code == 200

    run(go())

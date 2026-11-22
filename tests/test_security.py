"""防白嫖：没有有效 key 绝不触达上游；有 key 也不能绕过额度。

每个测试最后都检查假上游的调用记录 —— 这是"有没有花我们的钱"的唯一事实来源。
"""
import asyncio
import os
import re

import pytest
from starlette.routing import Route

from helpers import ADMIN, MockUpstream, body, gateway, load_test_settings, new_key, parse_sse

BODY = {"model": "deepseek-v4.1-flash", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
FORGED = [
    {},
    {"Authorization": ""},
    {"Authorization": "Bearer "},
    {"Authorization": "Bearer null"},
    {"Authorization": "Bearer undefined"},
    {"Authorization": "Bearer sk-thl-not-a-real-key"},
    {"Authorization": "Basic YWRtaW46YWRtaW4="},
    {"Authorization": "bearer sk-thl-"},
    {"X-API-Key": ""},
    {"X-API-Key": "sk-thl-guess"},
    {"api-key": "null"},
]
PUBLIC_OK = {("/health", "GET"), ("/admin", "GET"), ("/admin/", "GET"), ("/admin/api/logout", "POST")}


def run(coro):
    asyncio.run(coro)


def test_no_route_reaches_upstream_without_valid_key(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock) as (app, client, _):
            await new_key(client)  # 库里存在真实 key，确保不是"空库放行"
            from thalamus.api_admin import router as admin_router
            from thalamus.api_v1 import router as v1_router

            sources = [("", app.routes), ("/v1", v1_router.routes), ("/admin", admin_router.routes)]
            routes = [
                (prefix + r.path, m)
                for prefix, rs in sources
                for r in rs
                if isinstance(r, Route)
                for m in (r.methods or {"GET"})
                if m != "HEAD"
            ]
            assert len(routes) >= 20, routes
            assert any(p == "/v1/chat/completions" for p, _ in routes)
            for path, method in routes:
                real = path.replace("{kid}", "k_doesnotexist")
                creds = FORGED + ([{"Authorization": "Bearer admin-test-token"}] if real.startswith("/v1") else [])
                for h in creds:
                    for variant in {real, real + "/", real.upper(), "/" + real.lstrip("/")}:
                        r = await client.request(method, variant, json=BODY, headers=h)
                        if (real, method) in PUBLIC_OK and variant == real:
                            continue
                        if real.startswith("/v1") or real.startswith("/admin/api"):
                            assert r.status_code >= 300 or real == "/admin/api/logout", (method, variant, h, r.status_code)
            # 也试试 OpenAI 其他常见端点与路径穿越
            for p in ("/v1/completions", "/v1/responses", "/v1/embeddings", "/v1/messages", "/v1/../v1/chat/completions",
                      "/chat/completions", "/v1/chat/completions?api_key=sk-thl-x", "/admin/../v1/chat/completions"):
                r = await client.post(p, json=BODY)
                assert r.status_code >= 300, (p, r.status_code)
            assert mock.calls == [], mock.calls[:3]

    run(go())


def test_revoked_keys_never_reach_upstream(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock) as (app, client, _):
            r = await client.post("/admin/api/keys", json={"name": "a"}, headers=ADMIN)
            kid, raw = r.json()["key"]["id"], r.json()["api_key"]
            H = {"Authorization": f"Bearer {raw}"}

            # 先用一次，让 key 进入进程内缓存，再验证吊销类操作立即生效
            assert (await client.post("/v1/chat/completions", json=body("hi"), headers=H)).status_code == 200
            n = len(mock.calls)

            await client.patch(f"/admin/api/keys/{kid}", json={"enabled": False}, headers=ADMIN)
            assert (await client.post("/v1/chat/completions", json=body("hi"), headers=H)).status_code == 403
            await client.patch(f"/admin/api/keys/{kid}", json={"enabled": True, "expires_days": -1}, headers=ADMIN)
            assert (await client.post("/v1/chat/completions", json=body("hi"), headers=H)).status_code == 403
            await client.patch(f"/admin/api/keys/{kid}", json={"expires_days": None, "daily_credits": 0}, headers=ADMIN)
            r = await client.post("/v1/chat/completions", json=body("hi"), headers=H)
            assert r.status_code == 429 and r.json()["error"]["type"] == "quota_exceeded"
            await client.patch(f"/admin/api/keys/{kid}", json={"daily_credits": 100}, headers=ADMIN)
            new_raw = (await client.post(f"/admin/api/keys/{kid}/rotate", headers=ADMIN)).json()["api_key"]
            assert (await client.post("/v1/chat/completions", json=body("hi"), headers=H)).status_code == 401
            await client.delete(f"/admin/api/keys/{kid}", headers=ADMIN)
            r = await client.post("/v1/chat/completions", json=body("hi"), headers={"Authorization": f"Bearer {new_raw}"})
            assert r.status_code == 401
            assert len(mock.calls) == n

            for bad in ({"rpm": 0}, {"max_concurrency": 0}, {"daily_credits": -1}):
                assert (await client.post("/admin/api/keys", json={"name": "x", **bad}, headers=ADMIN)).status_code == 400

    run(go())


def test_huge_output_is_cut_at_daily_budget(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(kind="long")
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client, daily_credits=5, max_output_tokens=0, exhaust_mode="reject")
            r = await client.post(
                "/v1/chat/completions",
                json=body("写一篇长文", model="deepseek-v4.1-flash", stream=True, max_tokens=384000),
                headers={"Authorization": f"Bearer {k}"},
            )
            ev = [e for e in parse_sse(r.text) if isinstance(e, dict) and e.get("choices")]
            assert ev[-1]["choices"][0]["finish_reason"] == "length"
            await app.state.engine.drain()
            keys = (await client.get("/admin/api/keys", headers=ADMIN)).json()["keys"]
            used = keys[0]["used_today"]
            assert 4.0 < used <= 5.3, used  # 截在额度附近，而不是把 2000 段全部生成完
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert logs[0]["status"] == "quota_cut"
            # 额度用完后不再触达上游
            n = len(mock.calls)
            r = await client.post("/v1/chat/completions", json=body("hi"), headers={"Authorization": f"Bearer {k}"})
            assert r.status_code == 429 and len(mock.calls) == n

    run(go())


def test_disconnect_before_first_token_is_charged(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.on(kind="hang")
        async with gateway(tmp_path, mock) as (app, client, _):
            e = app.state.engine
            key = await e.authenticate(await new_key(client))
            task = asyncio.create_task(e.open(body("你好" * 2000, model="deepseek-v4.1-flash", stream=True), key, stream=True))
            for _ in range(200):
                if mock.calls:
                    break
                await asyncio.sleep(0.01)
            assert mock.calls, "request never reached the mock upstream"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await e.drain()
            u = await e.store.get_usage(key.id, e.ledger.today())
            assert u["used"] > 0 and u["reserved"] == 0 and u["requests"] == 1
            assert e.limiter.inflight(key.id) == 0 and sum(e.channel_load.values()) == 0

    run(go())


def test_oversized_body_is_rejected_before_parsing(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock) as (app, client, _):
            app.state.engine.settings.policy.max_request_bytes = 20_000
            k = await new_key(client)
            big = body("x" * 50_000, model="deepseek-v4.1-flash")
            r = await client.post("/v1/chat/completions", json=big, headers={"Authorization": f"Bearer {k}"})
            assert r.status_code == 413 and r.json()["error"]["type"] == "payload_too_large"
            assert mock.calls == []
            ok = body("hi", model="deepseek-v4.1-flash")
            assert (await client.post("/v1/chat/completions", json=ok, headers={"Authorization": f"Bearer {k}"})).status_code == 200

    run(go())


def test_invalid_keys_are_throttled_and_negatively_cached(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock) as (app, client, _):
            e = app.state.engine
            e.settings.policy.auth_fail_per_min = 5
            good = await new_key(client)
            queries = 0
            orig = e.store.get_key_by_raw

            async def counting(raw):
                nonlocal queries
                queries += 1
                return await orig(raw)

            e.store.get_key_by_raw = counting
            e.invalidate_keys()

            # 同一个无效 key 反复试：只应查库一次（负缓存）
            for _ in range(4):
                r = await client.post("/v1/chat/completions", json=body("hi"), headers={"Authorization": "Bearer sk-thl-bogus"})
                assert r.status_code == 401
            assert queries == 1, queries

            # 随机无效 key 刷接口：超过阈值后直接 429，不再查库
            codes = []
            for i in range(8):
                r = await client.post("/v1/chat/completions", json=body("hi"), headers={"Authorization": f"Bearer sk-thl-r{i}"})
                codes.append(r.status_code)
            assert 429 in codes and codes[-1] == 429
            assert queries <= 5, queries
            # 伪造 X-Forwarded-For 不能绕过按来源限流（只信 uvicorn 校验过的 client）
            e._auth_fails.clear()
            codes = []
            for i in range(10):
                r = await client.post(
                    "/v1/chat/completions", json=body("hi"),
                    headers={"Authorization": f"Bearer sk-thl-x{i}", "X-Forwarded-For": f"9.9.9.{i}"},
                )
                codes.append(r.status_code)
            assert 429 in codes, codes

            # 有效 key 在同一 IP 上也会被这个阈值挡住（保守取舍），恢复计数后应正常
            e._auth_fails.clear()
            assert (await client.post("/v1/chat/completions", json=body("hi", model="deepseek-v4.1-flash"),
                                      headers={"Authorization": f"Bearer {good}"})).status_code == 200

    run(go())


def test_region_whitelist_is_a_hard_boundary(tmp_path):
    """限定区域的 key，在换家 / 点名模型 / 全部候选失败时都不能越界。"""

    async def go():
        mock = MockUpstream()
        mock.on(channel="deepseek", kind="error")
        mock.on(channel="glm_plan", kind="error")
        async with gateway(tmp_path, mock) as (app, client, _):
            cn = await new_key(client, name="cn-only", allowed_regions=["cn"])
            HCN = {"Authorization": f"Bearer {cn}"}

            # 两个 cn 渠道全挂，也绝不能落到 us 渠道去兜底
            r = await client.post("/v1/chat/completions", json=body("写个快排"), headers=HCN)
            assert r.status_code >= 400, r.text
            assert mock.calls and {c["host"] for c in mock.calls} == {"api.deepseek.com", "open.bigmodel.cn"}

            # 只在 us 渠道上架的模型，对 cn key 不可见也不可点名
            ids = {m["id"] for m in (await client.get("/v1/models", headers=HCN)).json()["data"]}
            assert "glm-5.3-flash" in ids and "kimi-k2.7-code" not in ids and "minimax-m3" not in ids
            n = len(mock.calls)
            r = await client.post("/v1/chat/completions", json=body("hi", model="kimi-k2.7-code"), headers=HCN)
            assert r.status_code >= 400 and len(mock.calls) == n

            # 反向：us key 点名只有 cn 有的模型 → 无候选，且一次上游都不发
            us = await new_key(client, name="us-only", allowed_regions=["us"])
            HUS = {"Authorization": f"Bearer {us}"}
            r = await client.post("/v1/chat/completions", json=body("hi", model="deepseek-v4-pro"), headers=HUS)
            assert r.status_code >= 400 and len(mock.calls) == n
            r = await client.post("/v1/chat/completions", json=body("写个快排"), headers=HUS)
            assert r.status_code == 200 and mock.calls[-1]["host"] == "opencode.ai"

            # 未知区域 / 没有可用渠道的区域，管理接口直接拒绝
            for bad in (["eu"], ["cn", "eu"], []):
                r = await client.post("/admin/api/keys", json={"name": "x", "allowed_regions": bad or ["nope"]}, headers=ADMIN)
                assert r.status_code == 400, (bad, r.status_code)

            # 取消限制后立即恢复全渠道
            kid = next(k["id"] for k in (await client.get("/admin/api/keys", headers=ADMIN)).json()["keys"] if k["name"] == "us-only")
            r = await client.patch(f"/admin/api/keys/{kid}", json={"allowed_regions": None}, headers=ADMIN)
            assert r.json()["key"]["allowed_regions"] is None
            ids = {m["id"] for m in (await client.get("/v1/models", headers=HUS)).json()["data"]}
            assert {"kimi-k2.7-code", "glm-5.3-flash"} <= ids

    run(go())


def test_min_context_is_a_hard_boundary(tmp_path):
    """"只对接 1M 模型"必须是硬约束：点名也不行，/v1/models 里也看不到。"""

    async def go():
        mock = MockUpstream()
        s = load_test_settings(tmp_path)
        s.policy.min_context_tokens = 1_000_000
        async with gateway(tmp_path, mock, settings=s) as (app, client, _):
            k = await new_key(client)
            H = {"Authorization": f"Bearer {k}"}
            short = sorted({o.model.key for o in s.usable_offers() if o.model.context < 1_000_000})
            assert short, "配置里已经没有低于 1M 的模型，这个用例失去意义，请换一个构造"
            for m in short:
                r = await client.post("/v1/chat/completions", json=body("hi", model=m), headers=H)
                assert r.status_code >= 400, (m, r.status_code)
            assert mock.calls == [], mock.calls[:2]
            ids = {m["id"] for m in (await client.get("/v1/models", headers=H)).json()["data"]}
            assert "deepseek-v4.1-flash" in ids and not ids & set(short)
            r = await client.post("/v1/chat/completions", json=body("写个快排"), headers=H)
            assert r.status_code == 200 and mock.calls[-1]["model"] in ("deepseek-flash", "deepseek-v4.1-flash")

    run(go())


def test_first_token_budget_scales_with_input_length(tmp_path):
    """长上下文的预填充很慢：首 token 预算必须随输入放大，否则长请求会被误判为上游超时。"""

    async def go():
        mock = MockUpstream()
        mock.on(delay=0.35)
        s = load_test_settings(tmp_path)
        s.policy.tasks["chat"].first_token_timeout_s = 0.05
        s.policy.prefill_tps = 2_000  # 2000 token/s：约 1000 token 的输入 → 多给 0.5s
        async with gateway(tmp_path, mock, settings=s) as (app, client, _):
            k = await new_key(client, max_concurrency=1)
            H = {"Authorization": f"Bearer {k}"}
            short = await client.post("/v1/chat/completions", json=body("hi", model="deepseek-v4.1-flash"), headers=H)
            assert short.status_code >= 400, short.text  # 0.05s 预算 → 首 token 超时
            long = await client.post(
                "/v1/chat/completions",
                json=body("很长的上下文 " * 800, model="deepseek-v4.1-flash"),
                headers=H,
            )
            assert long.status_code == 200, long.text

    run(go())


def test_unknown_model_name_is_rejected_not_silently_rerouted(tmp_path):
    """配置写错（例如视觉模型写成不存在的名字）必须报错，不能 200 换个模型糊过去。"""

    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock) as (app, client, _):
            k = await new_key(client)
            H = {"Authorization": f"Bearer {k}"}
            for bad in ("mimo-v2.5", "gpt-5.6-luna", "thalamus/bogus", "deepseek-v4.1-flas"):
                r = await client.post("/v1/chat/completions", json=body("hi", model=bad), headers=H)
                assert r.status_code == 404, (bad, r.status_code)
                assert r.json()["error"]["type"] == "model_not_found"
                assert "GET /v1/models" in r.json()["error"]["message"]
            assert mock.calls == []

            # 所有合法写法都不受影响
            for ok in ("thalamus/auto", "thalamus/fast", "thalamus/code-3", "thalamus/vision",
                       "deepseek-v4.1-flash", "THALAMUS/AUTO"):
                r = await client.post("/v1/chat/completions", json=body("hi", model=ok), headers=H)
                assert r.status_code == 200, (ok, r.text)

            app.state.engine.settings.policy.unknown_model = "route"
            r = await client.post("/v1/chat/completions", json=body("hi", model="gpt-4o"), headers=H)
            assert r.status_code == 200  # drop-in 兼容模式

    run(go())


def test_response_format_is_filtered_not_sent_to_models_without_it(tmp_path):
    """上游会用 400 拒绝不支持的 response_format，与其每次撞一次，不如在路由阶段排除。"""

    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock) as (app, client, _):
            e = app.state.engine
            for m in e.settings.models.values():
                m.json_mode = False
            k = await new_key(client)
            H = {"Authorization": f"Bearer {k}"}
            r = await client.post(
                "/v1/chat/completions",
                json=body("给我 JSON", response_format={"type": "json_object"}),
                headers=H,
            )
            assert r.status_code == 400 and r.json()["error"]["type"] == "unsupported_input"
            assert "response_format" in r.json()["error"]["message"]
            assert mock.calls == []  # 一次也没去撞上游

            # 只要有一个模型支持就该走它
            e.settings.models["deepseek-v4-pro"].json_mode = True
            r = await client.post(
                "/v1/chat/completions",
                json=body("给我 JSON", response_format={"type": "json_object"}),
                headers=H,
            )
            assert r.status_code == 200 and r.headers["x-thalamus-model"] == "deepseek-v4-pro"
            assert mock.calls[-1]["payload"]["response_format"] == {"type": "json_object"}  # 原样透传

    run(go())


def test_admin_console_can_be_locked_to_source_ips(tmp_path):
    """审计发现控制台公网可达。反代白名单之外，应用层也要有一道。"""

    async def go():
        import httpx

        from thalamus.app import create_app

        mock = MockUpstream()
        s = load_test_settings(tmp_path)
        app = create_app(s, transport=httpx.MockTransport(mock.handler), warm=False)
        await app.state.startup()
        os.environ["THALAMUS_ADMIN_ALLOW"] = "10.0.0.0/8, 203.0.113.7"
        try:
            for ip, allowed in (("10.1.2.3", True), ("203.0.113.7", True), ("8.8.8.8", False), ("203.0.113.8", False)):
                c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=(ip, 1234)),
                                      base_url="http://test", timeout=10)
                try:
                    for path in ("/admin", "/admin/api/keys", "/admin/api/login"):
                        m = c.post if path.endswith("login") else c.get
                        r = await m(path, headers=ADMIN, **({"json": {"token": "x"}} if path.endswith("login") else {}))
                        if allowed:
                            assert r.status_code != 404, (ip, path, r.status_code)
                        else:
                            assert r.status_code == 404 and r.json()["error"]["type"] == "not_found", (ip, path)
                    # 数据面不受影响：白名单只管 /admin
                    assert (await c.get("/health")).status_code == 200
                finally:
                    await c.aclose()
        finally:
            os.environ.pop("THALAMUS_ADMIN_ALLOW", None)
            await app.state.shutdown()

    run(go())


def test_client_session_header_is_never_forwarded_raw(tmp_path):
    async def go():
        async with gateway(tmp_path) as (app, client, mock):
            k = await new_key(client)
            r = await client.post(
                "/v1/chat/completions",
                json=body("写个快排", model="kimi-k2.7-code"),
                headers={"Authorization": f"Bearer {k}", "X-Session-Id": "evil; value=1 <script>"},
            )
            assert r.status_code == 200
            sent = mock.calls[-1]["headers"]["x-opencode-session"]
            assert re.fullmatch(r"[0-9a-f]{24}", sent), sent

    run(go())

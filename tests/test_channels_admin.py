"""控制台自助加渠道：探测模型 → 勾选启用 → 热生效 → 真发一条测试请求。

每个用例都用独立的 config 目录副本，绝不写进仓库里的 config/。
"""
import asyncio
import shutil

import yaml

from helpers import ADMIN, ROOT, MockUpstream, body, gateway, new_key
from thalamus.config import load_settings

NEW = {
    "id": "qwen",
    "base_url": "https://dashscope.example.com/compatible-mode/v1",
    "dialect": "openai_chat",
    "region": "cn",
    "api_key": "sk-upstream-secret-value",
    "models": [
        {
            "key": "qwen3-max", "upstream": "qwen3-max", "display": "Qwen3 Max",
            "context": 1_000_000, "max_output": 32768, "tools": True, "quality": 8.0,
            "price": {"input": 0.05, "cached": 0.005, "output": 0.2},
        }
    ],
}


def run(coro):
    asyncio.run(coro)


def isolated(tmp_path, **policy):
    """独立的 config 目录 + 独立的 db/secrets 路径。"""
    cdir = tmp_path / "config"
    cdir.mkdir(parents=True, exist_ok=True)
    for name in ("upstreams.yaml", "policy.yaml"):
        shutil.copy(ROOT / "config" / name, cdir / name)
    s = load_settings(cdir)
    s.db_path = str(tmp_path / "t.db")
    s.admin_token = "admin-test-token"
    s.policy.min_context_tokens = 0
    for k, v in policy.items():
        setattr(s.policy, k, v)
    return s


def test_probe_lists_models_and_writes_nothing(tmp_path):
    async def go():
        mock = MockUpstream()
        mock.models = [
            {"id": "qwen3-flash"},
            {"id": "qwen3-max", "context_length": 1_000_000, "display_name": "Qwen3 Max"},
            "qwen3-lite",  # 有些实现直接给字符串
        ]
        async with gateway(tmp_path, mock, settings=isolated(tmp_path)) as (app, client, _):
            before = len((await client.get("/admin/api/channels", headers=ADMIN)).json()["channels"])
            r = await client.post(
                "/admin/api/channels/probe",
                json={"base_url": NEW["base_url"], "api_key": "sk-x", "dialect": "openai_chat"},
                headers=ADMIN,
            )
            d = r.json()
            assert r.status_code == 200 and d["ok"] and d["count"] == 3, d
            ids = [m["id"] for m in d["models"]]
            assert ids == ["qwen3-flash", "qwen3-lite", "qwen3-max"]
            got = {m["id"]: m for m in d["models"]}
            assert got["qwen3-max"]["context"] == 1_000_000  # 能认出来的上下文长度预填
            assert got["qwen3-flash"]["context"] is None  # 认不出来就留空，让人填
            assert mock.gets and "/models" in mock.gets[-1]
            # 探测不写任何配置
            after = (await client.get("/admin/api/channels", headers=ADMIN)).json()["channels"]
            assert len(after) == before
            assert not (tmp_path / "config" / "channels.d").exists()

    run(go())


def test_probe_reports_upstream_error_verbatim(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock, settings=isolated(tmp_path)) as (app, client, _):
            r = await client.post(
                "/admin/api/channels/probe",
                json={"base_url": "https://nope.example.com", "api_key": "sk-x"},
                headers=ADMIN,
            )
            assert r.status_code == 200  # 探测失败不是服务器错误
            assert r.json()["ok"] is True  # 假上游对任何 GET 都回 200
            r = await client.post("/admin/api/channels/probe", json={"base_url": "ftp://x", "api_key": "k"}, headers=ADMIN)
            assert r.status_code == 400
            r = await client.post("/admin/api/channels/probe", json={"base_url": NEW["base_url"]}, headers=ADMIN)
            assert r.status_code == 400  # 没给 key

    run(go())


def test_create_channel_takes_effect_without_restart(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock, settings=isolated(tmp_path)) as (app, client, _):
            r = await client.post("/admin/api/channels", json=NEW, headers=ADMIN)
            d = r.json()
            assert r.status_code == 201, d
            assert d["configured"] is True and d["offers"] == ["qwen3-max@qwen"]

            # 配置落在独立文件里，主配置没被动过
            f = tmp_path / "config" / "channels.d" / "qwen.yaml"
            doc = yaml.safe_load(f.read_text(encoding="utf-8"))
            assert doc["channels"]["qwen"]["api_key_env"] == "THALAMUS_KEY_QWEN"
            assert doc["models"]["qwen3-max"]["context"] == 1_000_000
            assert "qwen" not in (ROOT / "config" / "upstreams.yaml").read_text(encoding="utf-8")

            # 密钥进了运行时密钥库（0600 的 json），没有进环境变量，也不回显原值
            assert (tmp_path / "secrets.json").is_file()
            masked = (await client.get("/admin/api/secrets", headers=ADMIN)).json()["secrets"]
            mine = next(s for s in masked if s["name"] == "THALAMUS_KEY_QWEN")
            assert mine["source"] == "runtime" and NEW["api_key"] not in mine["value"]

            # 立刻可路由：点名新模型就能打到新地址，不用重启
            k = await new_key(client)
            h = {"Authorization": f"Bearer {k}"}
            r = await client.post("/v1/chat/completions", json=body("hi", model="qwen3-max"), headers=h)
            assert r.status_code == 200, r.text
            assert mock.calls[-1]["host"] == "dashscope.example.com"
            assert mock.calls[-1]["payload"]["model"] == "qwen3-max"
            assert r.headers["x-thalamus-model"] == "qwen3-max"
            ids = {m["id"] for m in (await client.get("/v1/models", headers=h)).json()["data"]}
            assert "qwen3-max" in ids

            # 也进了自动路由的候选集（订阅渠道闲置时更便宜，所以不一定胜出，但必须参与打分）
            pv = (await client.post("/admin/api/route-preview",
                                    json={"messages": [{"role": "user", "content": "写个快排"}]},
                                    headers=ADMIN)).json()
            assert "qwen3-max@qwen" in [c["offer"] for c in pv["ranked"]], pv["ranked"]

    run(go())


def test_created_channel_can_be_disabled_and_deleted(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock, settings=isolated(tmp_path)) as (app, client, _):
            assert (await client.post("/admin/api/channels", json=NEW, headers=ADMIN)).status_code == 201
            k = await new_key(client)
            h = {"Authorization": f"Bearer {k}"}

            r = await client.patch("/admin/api/channels/qwen", json={"enabled": False}, headers=ADMIN)
            assert r.status_code == 200 and r.json()["enabled"] is False
            n = len(mock.calls)
            r = await client.post("/v1/chat/completions", json=body("hi", model="qwen3-max"), headers=h)
            assert r.status_code >= 400 and len(mock.calls) == n  # 停用后点名也用不到

            assert (await client.patch("/admin/api/channels/qwen", json={"enabled": True}, headers=ADMIN)).status_code == 200
            assert (await client.post("/v1/chat/completions", json=body("hi", model="qwen3-max"), headers=h)).status_code == 200

            # 换 key 立即生效
            r = await client.patch("/admin/api/channels/qwen", json={"api_key": "sk-rotated"}, headers=ADMIN)
            assert r.status_code == 200 and "api_key" in r.json()["changed"]
            await client.post("/v1/chat/completions", json=body("hi", model="qwen3-max"), headers=h)
            assert mock.calls[-1]["headers"]["authorization"] == "Bearer sk-rotated"

            assert (await client.delete("/admin/api/channels/qwen", headers=ADMIN)).status_code == 200
            assert not (tmp_path / "config" / "channels.d" / "qwen.yaml").exists()
            ids = {m["id"] for m in (await client.get("/v1/models", headers=h)).json()["data"]}
            assert "qwen3-max" not in ids
            # 删掉后点名它 → 404（默认 unknown_model=reject），一次上游都不发
            n = len(mock.calls)
            r = await client.post("/v1/chat/completions", json=body("hi", model="qwen3-max"), headers=h)
            assert r.status_code == 404 and r.json()["error"]["type"] == "model_not_found", r.text
            assert len(mock.calls) == n

            # 需要 drop-in 兼容时可以切回"退回自动路由"，但绝不会打到已删除的地址
            app.state.engine.settings.policy.unknown_model = "route"
            r = await client.post("/v1/chat/completions", json=body("hi", model="qwen3-max"), headers=h)
            assert r.status_code == 200 and mock.calls[-1]["host"] != "dashscope.example.com"
            assert r.headers["x-thalamus-model"] != "qwen3-max"

    run(go())


def test_yaml_defined_channels_are_not_console_editable(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock, settings=isolated(tmp_path)) as (app, client, _):
            chans = {c["id"]: c for c in (await client.get("/admin/api/channels", headers=ADMIN)).json()["channels"]}
            assert chans["deepseek"]["managed"] is False and chans["deepseek"]["key_source"] == "env"
            assert (await client.delete("/admin/api/channels/deepseek", headers=ADMIN)).status_code == 400
            assert (await client.patch("/admin/api/channels/deepseek", json={"enabled": False}, headers=ADMIN)).status_code == 400
            # 但换 key 是允许的（运行时密钥覆盖环境变量）
            r = await client.patch("/admin/api/channels/deepseek", json={"api_key": "sk-new-ds"}, headers=ADMIN)
            assert r.status_code == 200
            k = await new_key(client)
            # 用只在 deepseek 渠道上架的模型，确保打到的就是它
            await client.post("/v1/chat/completions", json=body("hi", model="deepseek-v4-pro"),
                              headers={"Authorization": f"Bearer {k}"})
            assert mock.calls[-1]["headers"]["authorization"] == "Bearer sk-new-ds"

    run(go())


def test_bad_input_is_rejected_and_nothing_is_written(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock, settings=isolated(tmp_path)) as (app, client, _):
            cd = tmp_path / "config" / "channels.d"
            bad = [
                {**NEW, "id": "../evil"},                       # 路径穿越
                {**NEW, "id": "1qwen"},                         # 不能数字开头
                {**NEW, "id": "a"},                             # 太短
                {**NEW, "base_url": "javascript:alert(1)"},
                {**NEW, "models": []},                          # 一个模型都没勾
                {**NEW, "api_key": ""},
                {**NEW, "kind": "subscription"},                # 订阅渠道不走控制台
                {**NEW, "models": [{"key": "x", "price": {"input": 1}}]},          # 缺 output 价
                {**NEW, "models": [{"key": "x", "price": {"input": -1, "output": 1}, "context": 1000}]},
                {**NEW, "models": [{"key": "a b", "price": {"input": 1, "output": 1}}]},  # key 带空格
            ]
            for payload in bad:
                r = await client.post("/admin/api/channels", json=payload, headers=ADMIN)
                assert r.status_code == 400, (payload.get("id"), r.status_code, r.text[:120])
            assert not cd.exists() or not list(cd.glob("*.yaml"))
            assert "qwen" not in {c["id"] for c in (await client.get("/admin/api/channels", headers=ADMIN)).json()["channels"]}

            assert (await client.post("/admin/api/channels", json=NEW, headers=ADMIN)).status_code == 201
            assert (await client.post("/admin/api/channels", json=NEW, headers=ADMIN)).status_code == 409  # 重名

    run(go())


def test_test_endpoint_surfaces_real_upstream_behaviour(tmp_path):
    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock, settings=isolated(tmp_path)) as (app, client, _):
            assert (await client.post("/admin/api/channels", json=NEW, headers=ADMIN)).status_code == 201
            r = await client.post("/admin/api/channels/qwen/test", json={"model": "qwen3-max"}, headers=ADMIN)
            d = r.json()
            assert d["ok"] and d["text"] and d["upstream_model"] == "qwen3-max", d
            assert d["usage"]["input_miss"] == 40 and d["usage"]["output"] == 20  # usage 字段被正确解析
            assert d["usage_missing"] is False and d["finish_reason"] == "stop"
            # 测试请求不计入用户额度，也不产生请求日志
            assert (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"] == []

            mock.on(model="qwen3-max", kind="error", status=429, body="Rate limit reached for requests")
            d = (await client.post("/admin/api/channels/qwen/test", json={"model": "qwen3-max"}, headers=ADMIN)).json()
            assert d["ok"] is False and d["failure"] == "rate_limited" and "Rate limit" in d["message"]

            r = await client.post("/admin/api/channels/qwen/test", json={"model": "nope"}, headers=ADMIN)
            assert r.status_code == 404

    run(go())


def test_new_channel_respects_context_gate_and_reports_it(tmp_path):
    async def go():
        mock = MockUpstream()
        s = isolated(tmp_path, min_context_tokens=1_000_000)
        async with gateway(tmp_path, mock, settings=s) as (app, client, _):
            small = {**NEW, "id": "local", "base_url": "http://127.0.0.1:11434/v1",
                     "models": [{**NEW["models"][0], "key": "llama-8b", "context": 131072}]}
            r = await client.post("/admin/api/channels", json=small, headers=ADMIN)
            assert r.status_code == 201, r.text
            # 能加进来，但被 1M 门槛挡在路由之外，并且启动校验会说清楚
            assert any("排除了" in w and "llama-8b" in w for w in r.json()["warnings"]), r.json()["warnings"]
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json=body("hi", model="llama-8b"),
                                  headers={"Authorization": f"Bearer {k}"})
            assert r.status_code >= 400 and mock.calls == []

    run(go())

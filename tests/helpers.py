"""测试工具：可编程的假上游（OpenAI 流式 + Anthropic 流式）与网关启动器。"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

ROOT = Path(__file__).resolve().parent.parent
ADMIN = {"Authorization": "Bearer admin-test-token"}

HOSTS = {"deepseek": "api.deepseek.com", "glm_plan": "open.bigmodel.cn", "opencode_go": "opencode.ai"}


@dataclass
class Rule:
    kind: str = "text"  # text | tools | error | hang | stall | midfail
    text: str = "Hello world"
    status: int = 500
    body: str = "upstream boom"
    delay: float = 0.0
    times: int | None = None
    out_tokens: int = 20


class AStream(httpx.AsyncByteStream):
    def __init__(self, agen: AsyncIterator[str]):
        self.agen = agen

    async def __aiter__(self):
        async for s in self.agen:
            yield s.encode()

    async def aclose(self) -> None:
        await self.agen.aclose()


class MockUpstream:
    def __init__(self) -> None:
        self.rules: list[tuple[str | None, str | None, Rule]] = []
        self.calls: list[dict[str, Any]] = []  # 只记 POST：它是"有没有花我们的钱"的事实来源
        self.gets: list[str] = []  # GET /models（预热与管理台探测）
        self.models: list[Any] = []  # GET /models 的 data 数组

    def on(self, channel: str | None = None, model: str | None = None, **kw: Any) -> Rule:
        r = Rule(**kw)
        self.rules.insert(0, (channel, model, r))
        return r

    def _match(self, host: str, model: str) -> Rule:
        for ch, m, r in self.rules:
            if ch and HOSTS[ch] != host:
                continue
            if m and m != model:
                continue
            if r.times is not None:
                if r.times <= 0:
                    continue
                r.times -= 1
            return r
        return Rule()

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.gets.append(str(request.url))
            return httpx.Response(200, json={"data": self.models})
        payload = json.loads(request.content)
        host, model = request.url.host, payload.get("model")
        self.calls.append(
            {"host": host, "path": request.url.path, "model": model, "payload": payload, "headers": dict(request.headers)}
        )
        r = self._match(host, model)
        if r.kind == "error":
            return httpx.Response(r.status, text=r.body)
        gen = self._anthropic(r) if request.url.path.endswith("/messages") else self._openai(r, model)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=AStream(gen))

    @staticmethod
    async def _openai(r: Rule, model: str) -> AsyncIterator[str]:
        def ev(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        def ch(delta: dict, finish: str | None = None) -> str:
            return ev({"id": "x", "object": "chat.completion.chunk", "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        if r.kind == "hang":
            await asyncio.sleep(3600)
        if r.delay:
            await asyncio.sleep(r.delay)
        if r.kind == "long":
            yield ch({"role": "assistant", "content": ""})
            for _ in range(2000):
                yield ch({"content": "字" * 50})
            yield ch({}, "stop")
        elif r.kind in ("tools", "badtools"):
            yield ch({"role": "assistant", "content": None, "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]})
            yield ch({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":" if r.kind == "tools" else "{city: "}}]})
            yield ch({"tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}" if r.kind == "tools" else "Paris"}}]})
            yield ch({}, "tool_calls")
        else:
            half = len(r.text) // 2
            yield ch({"role": "assistant", "content": r.text[:half]})
            if r.kind == "stall":
                await asyncio.sleep(3600)
            if r.kind == "midfail":
                raise httpx.ReadError("connection reset")
            yield ch({"content": r.text[half:]})
            yield ch({}, "stop")
        yield ev({"id": "x", "choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": r.out_tokens, "prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40}})
        yield "data: [DONE]\n\n"

    @staticmethod
    async def _anthropic(r: Rule) -> AsyncIterator[str]:
        def ev(t: str, obj: dict) -> str:
            return f"event: {t}\ndata: {json.dumps({'type': t, **obj})}\n\n"

        if r.kind == "hang":
            await asyncio.sleep(3600)
        yield ev("message_start", {"message": {"id": "m", "usage": {"input_tokens": 50, "cache_read_input_tokens": 30, "output_tokens": 1}}})
        if r.kind == "tools":
            yield ev("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}})
            yield ev("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}})
            yield ev("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"Paris\"}"}})
            yield ev("content_block_stop", {"index": 0})
            yield ev("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 15}})
        else:
            yield ev("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
            yield ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "<thi"}})
            yield ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "nk>plan</think>" + r.text}})
            yield ev("content_block_stop", {"index": 0})
            yield ev("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12}})
        yield ev("message_stop", {})


def load_test_settings(tmp_path: Path):
    from thalamus.config import load_settings

    s = load_settings(ROOT / "config")
    s.db_path = str(tmp_path / "test.db")
    s.admin_token = "admin-test-token"
    # 线上策略是"只对接 1M 上下文的模型"，但路由逻辑的测试需要按量 / 积分 / 百分比三种
    # 渠道和 anthropic 方言同时在场，所以默认放开这道门槛；门槛本身由
    # test_router.test_min_context_gate_* 和 test_security.test_min_context_is_a_hard_boundary 覆盖。
    s.policy.min_context_tokens = 0
    # 门槛的"按任务豁免"是一项配置选择，不是代码属性：测试里一律从空开始，
    # 哪个用例要它就自己打开（见 test_router.test_min_context_exemption_*）。
    # 否则把门槛设成 1M 的用例会因为线上 policy.yaml 豁免了 chat 而意外放进
    # 200k 的 minimax（唯一的 anthropic offer），假上游剧本就不适用了。
    s.policy.min_context_exempt_tasks = frozenset()
    # 同上：TOTP 默认关，避免继承宿主机的运行期配置
    s.totp_secret, s.totp_only = "", False
    return s


@asynccontextmanager
async def gateway(tmp_path: Path, mock: MockUpstream | None = None, settings=None):
    from thalamus.app import create_app

    mock = mock or MockUpstream()
    s = settings or load_test_settings(tmp_path)
    app = create_app(s, transport=httpx.MockTransport(mock.handler), warm=False)
    await app.state.startup()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30)
    try:
        yield app, client, mock
    finally:
        await client.aclose()
        await app.state.shutdown()


async def new_key(client: httpx.AsyncClient, **fields: Any) -> str:
    r = await client.post("/admin/api/keys", json={"name": "tester", **fields}, headers=ADMIN)
    assert r.status_code == 201, r.text
    return r.json()["api_key"]


def parse_sse(text: str) -> list[Any]:
    out: list[Any] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def body(content: Any, **kw: Any) -> dict[str, Any]:
    return {"model": kw.pop("model", "thalamus/auto"), "messages": [{"role": "user", "content": content}], **kw}

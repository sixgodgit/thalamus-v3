"""本地验证器：工具参数 schema、代码块语法，以及"答错的代价"如何影响排序。

误报比漏报严重得多（会污染学习信号），所以这里一半的用例是"不该报的别报"。
"""
import json
import math

from helpers import ROOT
from thalamus.config import load_settings
from thalamus.health import OutcomeBook
from thalamus.outcome import (
    BAD_FLAGS,
    CODE_INVALID,
    EMPTY,
    JSON_INVALID,
    TOOL_ARGS_INVALID,
    TOOL_SCHEMA_INVALID,
    OutcomeTracker,
)
from thalamus.types import Delta, Usage
from thalamus.verify import check_code_blocks, check_tool_args

WEATHER = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "unit": {"type": "string", "enum": ["c", "f"]},
                "verbose": {"type": "boolean"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]


def test_tool_args_schema_violations():
    ok = check_tool_args("get_weather", {"city": "Paris", "days": 3, "unit": "c"}, WEATHER)
    assert ok == "", ok
    assert "unknown function" in check_tool_args("get_wether", {"city": "x"}, WEATHER)
    assert "missing required" in check_tool_args("get_weather", {"days": 1}, WEATHER)
    assert "unexpected" in check_tool_args("get_weather", {"city": "x", "colour": "red"}, WEATHER)
    assert "should be" in check_tool_args("get_weather", {"city": "x", "days": "three"}, WEATHER)
    assert "should be" in check_tool_args("get_weather", {"city": 42}, WEATHER)
    assert "not in enum" in check_tool_args("get_weather", {"city": "x", "unit": "kelvin"}, WEATHER)
    # bool 不能当 integer（Python 里 bool 是 int 的子类，容易漏）
    assert "should be" in check_tool_args("get_weather", {"city": "x", "days": True}, WEATHER)
    # 3.0 当 integer 可以，3.5 不行
    assert check_tool_args("get_weather", {"city": "x", "days": 3.0}, WEATHER) == ""
    assert "should be" in check_tool_args("get_weather", {"city": "x", "days": 3.5}, WEATHER)


def test_tool_args_stays_silent_when_it_cannot_judge():
    """判不了就别判：没声明 tools、schema 用了 anyOf/$ref、参数不是对象。"""
    assert check_tool_args("get_weather", {"city": "x"}, None) == ""
    assert check_tool_args("get_weather", {"city": "x"}, []) == ""
    fancy = [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "number"}]}}}}}]
    assert check_tool_args("f", {"x": True}, fancy) == ""
    ref = [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"x": {"$ref": "#/defs/T"}}}}}]
    assert check_tool_args("f", {"x": {"any": "thing"}}, ref) == ""
    # 没写 additionalProperties: False 就不管多余字段
    loose = [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"a": {"type": "string"}}}}}]
    assert check_tool_args("f", {"a": "1", "b": 2}, loose) == ""
    assert check_tool_args("f", "not-an-object", loose) == ""


def test_code_block_syntax_detection():
    assert check_code_blocks("没有代码") == ""
    assert check_code_blocks("```python\nx = 1\ny = 2\n```") == ""
    bad = check_code_blocks("```python\ndef f(:\n    return 1\n```")
    assert "python block" in bad, bad
    assert "json block" in check_code_blocks('```json\n{"a": 1,}\n```')
    assert "yaml block" in check_code_blocks("```yaml\na: 1\n b: [\n```")
    assert check_code_blocks('```json\n{"a": 1}\n```') == ""


def test_code_block_false_positives_are_avoided():
    """片段式输出是正常的，不能算模型答错。"""
    cases = [
        "```python\ndef f(x):\n    ...\n```",                      # 省略号
        "```python\n# ...\nresult = compute()\n```",
        "```python\nclass A:\n    # 省略其余方法\n    pass\n```",
        "```\ndef f(:\n```",                                        # 没标语言
        "```python\n>>> f(1)\n2\n```",                               # REPL 回显
        "```python\n$ pip install x\n```",                           # shell
        "```diff\n- old\n+ new(\n```",                               # diff
        "```python\n- item\n+ item(\n```",                           # diff 形态但标了 python
        "```python\nx = (\n```",                                     # 未闭合 = 被截断的片段
        "```python\ns = 'unterminated\nt = 2\n```",
        "```text\n{不是 json}\n```",                                 # 非检查语言
        "```python\n\n```",                                          # 空块
    ]
    for src in cases:
        assert check_code_blocks(src) == "", src


def _run(tracker: OutcomeTracker, *deltas: Delta, finish: str = "stop") -> list[str]:
    for d in deltas:
        tracker.feed(d)
    return tracker.flags(finish, "ok")


def test_tracker_reports_schema_violation_separately_from_bad_json():
    call = [{"index": 0, "id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": ""}}]
    t = OutcomeTracker(False, tools=WEATHER)
    flags = _run(t, Delta(tool_calls=call), Delta(tool_calls=[
        {"index": 0, "function": {"arguments": '{"days": 2}'}}]), finish="tool_calls")
    assert flags == [TOOL_SCHEMA_INVALID] and "missing required" in t.detail

    t2 = OutcomeTracker(False, tools=WEATHER)
    flags = _run(t2, Delta(tool_calls=call), Delta(tool_calls=[
        {"index": 0, "function": {"arguments": '{"city": '}}]), finish="tool_calls")
    assert flags == [TOOL_ARGS_INVALID]

    t3 = OutcomeTracker(False, tools=WEATHER)
    flags = _run(t3, Delta(tool_calls=call), Delta(tool_calls=[
        {"index": 0, "function": {"arguments": '{"city": "Paris"}'}}]), finish="tool_calls")
    assert flags == []


def test_tracker_code_check_only_when_enabled_and_output_complete():
    broken = Delta(content="给你：\n```python\ndef f(:\n    pass\n```")
    assert _run(OutcomeTracker(False, check_code=True), broken) == [CODE_INVALID]
    assert _run(OutcomeTracker(False, check_code=False), broken) == []
    # 被截断的输出天然不完整，不据此判定质量
    assert CODE_INVALID not in _run(OutcomeTracker(False, check_code=True), broken, finish="length")
    # 其他信号仍然独立工作
    assert _run(OutcomeTracker(True, check_code=True), Delta(content="不是 json")) == [JSON_INVALID]
    assert _run(OutcomeTracker(False, check_code=True), Delta(content="   ")) == [EMPTY]


def test_outcome_book_shrinks_to_prior_then_diverges():
    b = OutcomeBook(prior_rate=0.03, prior_n=20.0)
    assert b.rate("m", "code") == 0.03  # 没数据时就是先验，不扰动路由
    for _ in range(3):
        b.observe("m", "code", True)
    assert b.rate("m", "code") < 0.2  # 三次失败还不足以判死
    for _ in range(200):
        b.observe("m", "code", True)
    assert b.rate("m", "code") > 0.8  # 持续失败才会被定价
    for _ in range(500):
        b.observe("m", "code", False)
    assert b.rate("m", "code") < 0.1  # 衰减让旧表现不会一直压着
    assert b.rate("other", "code") == 0.03


def test_outcome_book_seeds_from_stored_stats():
    b = OutcomeBook()
    rows = [{"model": "m", "task": "code", "n": 100, **{f: 0 for f in BAD_FLAGS}}]
    rows[0][TOOL_SCHEMA_INVALID] = 40
    b.seed(rows)
    assert 0.3 < b.rate("m", "code") < 0.4
    assert any(r["model"] == "m" for r in b.snapshot())


def test_end_to_end_flags_and_learning(tmp_path):
    """整条链路：假上游给出违反 schema 的工具参数 → 标记落进请求日志、模型被扣分、
    坏比率被记住并出现在管理台。"""
    import asyncio

    from helpers import ADMIN, MockUpstream, gateway, new_key

    async def go():
        mock = MockUpstream()
        mock.on(kind="tools")  # helpers 的 tools 剧本只给 {"city": "Paris"}
        async with gateway(tmp_path, mock, settings=prod_like(tmp_path)) as (app, client, _):
            e = app.state.engine
            k = await new_key(client)
            H = {"Authorization": f"Bearer {k}"}
            # 声明 days 必填 → 上游只给 city，属于 schema 违规
            tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city", "days"]}}}]
            r = await client.post("/v1/chat/completions", json={
                "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "巴黎天气"}],
                "tools": tools}, headers=H)
            assert r.status_code == 200, r.text
            await e.drain()
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert TOOL_SCHEMA_INVALID in logs[0]["flags"], logs[0]["flags"]
            assert e.outcomes.rate("deepseek-v4.1-flash", logs[0]["task"]) > 0.03
            assert any(d["delta"] < 0 for d in e.learner.snapshot())
            # 同样的上游输出，但客户端没声明 required → 不该报
            tools[0]["function"]["parameters"]["required"] = ["city"]
            r = await client.post("/v1/chat/completions", json={
                "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "巴黎天气"}],
                "tools": tools}, headers=H)
            await e.drain()
            logs = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"]
            assert TOOL_SCHEMA_INVALID not in (logs[0]["flags"] or "")

    asyncio.run(go())


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def prod_like(tmp_path):
    """与线上一致的 1M 门槛 —— 顺带把 200k 的 minimax（唯一的 anthropic offer）排除，
    这样候选全是 openai_chat，假上游的剧本才都适用。"""
    from helpers import load_test_settings

    s = load_test_settings(tmp_path)
    s.policy.min_context_tokens = 1_000_000
    return s


def test_verify_retry_swaps_model_and_keeps_the_good_answer(tmp_path):
    import asyncio

    from helpers import ADMIN, MockUpstream, gateway, new_key

    async def go():
        mock = MockUpstream()
        mock.on(kind="tools")               # 兜底：合法工具调用
        mock.on(kind="badtools", times=1)   # 第一次给不合法的参数
        async with gateway(tmp_path, mock, settings=prod_like(tmp_path)) as (app, client, _):
            e = app.state.engine
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "巴黎天气怎么样"}], "tools": TOOLS,
            }, headers={"Authorization": f"Bearer {k}"})
            assert r.status_code == 200, r.text
            body = r.json()

            assert len(mock.calls) == 2, mock.calls
            first, second = mock.calls[0]["model"], mock.calls[1]["model"]
            assert first != second, (first, second)
            assert r.headers["x-thalamus-attempts"] == "2"
            # 客户端拿到的是重做之后的好答案
            args = body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            assert json.loads(args) == {"city": "Paris"}

            await e.drain()
            log = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"][0]
            assert "verify_retry" in log["flags"] and TOOL_ARGS_INVALID not in log["flags"], log["flags"]
            # 第一次的负面信号被记在那个模型头上（学习闭环不能断）
            bad_model = next(m for m in e.settings.models if m != log["model"]
                             and any(o.upstream == first and o.model.key == m for o in e.settings.offers))
            assert e.outcomes.rate(bad_model, log["task"]) > 0.03
            assert any(d["model"] == bad_model and d["delta"] < 0 for d in e.learner.snapshot())

    asyncio.run(go())


def test_no_retry_when_no_candidate_is_worth_opening(tmp_path):
    """Weitzman 的停止规则：没有正保留值的候选就不重试，把坏答案原样交出去。"""
    import asyncio

    from helpers import ADMIN, MockUpstream, gateway, new_key

    async def go():
        mock = MockUpstream()
        mock.on(kind="badtools")
        async with gateway(tmp_path, mock, settings=prod_like(tmp_path)) as (app, client, _):
            e = app.state.engine
            e.settings.policy.verify_retry_max_credits = 0.0  # 谁都不够便宜
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "巴黎天气"}], "tools": TOOLS,
            }, headers={"Authorization": f"Bearer {k}"})
            assert r.status_code == 200 and len(mock.calls) == 1
            assert r.headers["x-thalamus-attempts"] == "1"
            await e.drain()
            log = (await client.get("/admin/api/requests", headers=ADMIN)).json()["requests"][0]
            assert TOOL_ARGS_INVALID in log["flags"] and "verify_retry" not in log["flags"]

    asyncio.run(go())


def test_reservation_value_picks_by_sigma_not_by_score(tmp_path):
    """σ = 价值 − (成本+等待)/(1−坏比率)：坏比率高的候选即使分数高也不该被选去重试。"""
    import asyncio

    from helpers import MockUpstream, gateway, new_key

    async def go():
        mock = MockUpstream()
        async with gateway(tmp_path, mock, settings=prod_like(tmp_path)) as (app, client, _):
            e = app.state.engine
            key = await e.authenticate(await new_key(client))
            call = await e.open({"messages": [{"role": "user", "content": "巴黎天气"}], "tools": TOOLS},
                                key, stream=False)
            await call.collect()
            ranked = call.plan.ranked
            assert len(ranked) >= 3
            chosen = call.decision.offer.model.key

            failed = call.decision
            others = [c for c in ranked if c.offer.model.key != chosen]
            for c in ranked:
                c.bad_rate = 0.0
            top = call.retry_target(failed)
            assert top is not None and top.offer.model.key != chosen
            # 把它的坏比率推到极高 → 保留值掉下去 → 换成别人
            top.bad_rate = 0.95
            nxt = call.retry_target(failed)
            assert nxt is None or nxt.offer.key != top.offer.key
            # 所有候选都很坏 → 一个都不比"就这样交出去"更值得开
            for c in others:
                c.bad_rate = 0.95
            assert call.retry_target(failed) is None

    asyncio.run(go())


def test_streamed_prose_is_never_buffered_for_retry(tmp_path):
    """给人读的散文一旦开始流就不能收回：不缓冲、不重试。"""
    import asyncio

    from helpers import MockUpstream, body, gateway, new_key, parse_sse

    async def go():
        mock = MockUpstream()
        mock.on(kind="badtools")
        async with gateway(tmp_path, mock, settings=prod_like(tmp_path)) as (app, client, _):
            k = await new_key(client)
            H = {"Authorization": f"Bearer {k}"}
            r = await client.post("/v1/chat/completions", json=body("讲个笑话", stream=True), headers=H)
            assert r.status_code == 200 and len(mock.calls) == 1
            assert r.headers["x-thalamus-attempts"] == "1"

            # 流式 + 工具调用则可以缓冲（客户端拿不到半个工具调用）
            mock.calls.clear()
            mock.on(kind="tools")
            mock.on(kind="badtools", times=1)
            r = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "巴黎天气"}], "tools": TOOLS, "stream": True}, headers=H)
            assert r.status_code == 200 and len(mock.calls) == 2
            events = [x for x in parse_sse(r.text) if isinstance(x, dict) and x.get("choices")]
            assert events, r.text[:200]

    asyncio.run(go())


def test_long_answer_gives_up_retry_instead_of_stalling(tmp_path):
    """声明了工具却回了长篇散文：超过缓冲上限就放弃重试、恢复真正的流式。"""
    import asyncio

    from helpers import MockUpstream, gateway, new_key, parse_sse

    async def go():
        mock = MockUpstream()
        mock.on(kind="long")
        async with gateway(tmp_path, mock, settings=prod_like(tmp_path)) as (app, client, _):
            app.state.engine.settings.policy.verify_buffer_bytes = 2048
            k = await new_key(client)
            r = await client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "写长文"}], "tools": TOOLS, "stream": True,
            }, headers={"Authorization": f"Bearer {k}"})
            assert r.status_code == 200 and len(mock.calls) == 1
            chunks = [x for x in parse_sse(r.text) if isinstance(x, dict) and x.get("choices")]
            assert len(chunks) > 50  # 内容完整流了出来，没有被缓冲吞掉

    asyncio.run(go())


def test_bad_rate_is_priced_into_routing():
    """这是整件事的目的：老出错的模型会被"重做一次的代价"推到后面。"""
    from thalamus.capacity import CapacityTracker
    from thalamus.features import extract
    from thalamus.health import HealthBook, QualityLearner
    from thalamus.intent import classify
    from thalamus.router import KeyView, Router

    s = load_settings(ROOT / "config")
    s.policy.min_context_tokens = 0
    book = OutcomeBook()
    r = Router(s, CapacityTracker(s), HealthBook(s.policy), QualityLearner(s.policy), {}, book)
    body = {"messages": [{"role": "user", "content": "这个 python 函数为什么会死锁"}]}
    f = extract(body, "k")
    it = classify(f, s.policy.long_context_tokens)
    kv = KeyView("k", "standard", None, 0)

    first = r.plan(f, it, kv).ranked[0].offer.model.key
    base = {c.offer.key: c.risk_cost for c in r.plan(f, it, kv).ranked}

    for _ in range(300):  # 让胜出者在这个任务上持续失败
        book.observe(first, it.task, True)
    after = r.plan(f, it, kv)
    hurt = next(c for c in after.ranked if c.offer.model.key == first)
    assert hurt.bad_rate > 0.8 and hurt.risk_cost > base[hurt.offer.key] * 5
    assert after.ranked[0].offer.model.key != first, [c.offer.key for c in after.ranked[:3]]
    assert not math.isnan(after.ranked[0].score)

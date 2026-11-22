from helpers import ROOT
from thalamus.adapters.anthropic import AnthropicAdapter, to_anthropic_messages, to_anthropic_tool_choice
from thalamus.adapters.base import CallContext, ThinkSplitter, classify_http, resolve_max_tokens
from thalamus.adapters.openai_chat import OpenAIChatAdapter, parse_usage
from thalamus.config import load_settings
from thalamus.types import FailureClass


def offer(key):
    return load_settings(ROOT / "config").offer(key)


def test_think_splitter_across_chunks():
    t = ThinkSplitter()
    out = [t.feed("Hi <thi"), t.feed("nk>secret</th"), t.feed("ink> there"), t.flush()]
    assert "".join(c for c, _ in out) == "Hi  there"
    assert "".join(r for _, r in out) == "secret"


def test_classify_http():
    assert classify_http(429, "Rate limit reached for requests").failure is FailureClass.RATE_LIMITED
    assert classify_http(429, "usage limit reached for 5 hour window").failure is FailureClass.QUOTA_EXHAUSTED
    assert classify_http(429, '{"error":{"code":"1308","message":"已达到 5 小时的使用上限"}}').failure is FailureClass.QUOTA_EXHAUSTED
    assert classify_http(400, "This model's maximum context length is 128000").failure is FailureClass.CONTEXT_OVERFLOW
    assert classify_http(401, "bad key").failure is FailureClass.AUTH_FAILED
    assert classify_http(402, "Insufficient Balance").failure is FailureClass.QUOTA_EXHAUSTED
    assert classify_http(404, "model not found").failure is FailureClass.PROVIDER_DOWN
    e = classify_http(400, "invalid tool schema")
    assert e.failure is FailureClass.CLIENT_ERROR and not e.retryable
    assert classify_http(503, "overloaded").retryable


def test_parse_usage_shapes():
    ds = parse_usage({"prompt_tokens": 100, "completion_tokens": 9, "prompt_cache_hit_tokens": 70, "prompt_cache_miss_tokens": 30})
    assert (ds.input_miss, ds.input_cached, ds.output) == (30, 70, 9)
    zp = parse_usage({"prompt_tokens": 100, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 40}})
    assert (zp.input_miss, zp.input_cached) == (60, 40)
    assert parse_usage(None) is None


def test_resolve_max_tokens():
    glm = offer("glm-5.3@glm_plan")
    assert resolve_max_tokens({"max_tokens": 100}, glm, 0) == 2048  # reasoning 模型抬到下限
    assert resolve_max_tokens({"max_tokens": 100}, glm, 500) == 100  # key 上限低于下限时尊重 key
    ds = offer("deepseek-v4.1-flash@deepseek")
    assert resolve_max_tokens({}, ds, 0) is None
    assert resolve_max_tokens({"max_tokens": 9000}, ds, 4000) == 4000
    # 长上下文：输入吃掉窗口后，剩余空间是 max_tokens 的硬上限
    assert resolve_max_tokens({"max_tokens": 200_000}, ds, 0, room=5_000) == 5_000
    assert resolve_max_tokens({"max_tokens": 3_000}, ds, 0, room=5_000) == 3_000
    assert resolve_max_tokens({}, ds, 0, room=5_000) is None  # 客户端没给就不替它设
    assert resolve_max_tokens({}, ds, 0, default=8192, room=5_000) == 5_000


def test_openai_payload_passthrough():
    ad = OpenAIChatAdapter(offer("deepseek-v4.1-flash@deepseek"), None)
    body = {
        "model": "thalamus/auto", "stream": False, "max_tokens": 50, "session_id": "s", "n": 5,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}],
        "tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "auto", "response_format": {"type": "json_object"},
    }
    from thalamus.adapters.base import CallContext

    p = ad.build(body, CallContext("s", "r"))
    assert p["model"] == "deepseek-flash" and p["stream"] is True and p["stream_options"]["include_usage"]
    assert p["messages"] == body["messages"] and p["tools"] == body["tools"] and p["response_format"] == body["response_format"]
    assert "session_id" not in p and "n" not in p and p["max_tokens"] == 50


def test_extra_body_overrides_client_params():
    """实测发现 kimi-k2.7-code 只接受 temperature=1，别的值直接 400 且不可重试。
    网关原样透传客户端参数，所以必须让 offer 的 extra_body 压过客户端的值。"""
    o = offer("kimi-k2.7-code@opencode_go")
    assert o.extra_body.get("temperature") == 1, "upstreams.yaml 里这条覆盖不能丢"
    p = OpenAIChatAdapter(o, None).build(
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2, "top_p": 0.9},
        CallContext("s", "r"),
    )
    assert p["temperature"] == 1 and p["top_p"] == 0.9  # 只覆盖声明过的那个参数


def test_openai_think_tags_and_reasoning_field():
    o = offer("minimax-m3@opencode_go")
    o.dialect = "openai_chat"
    o.think_tags = True
    ad = OpenAIChatAdapter(o, None)
    ds = ad.parse("", {"choices": [{"delta": {"content": "<think>a</think>B"}}]})
    assert ds[0].content == "B" and ds[0].reasoning == "a"
    ds = ad.parse("", {"choices": [{"delta": {"reasoning_content": "hmm"}}]})
    assert ds[0].reasoning == "hmm" and ds[0].meaningful


def test_anthropic_request_translation():
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": [{"type": "text", "text": "看图"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]},
        {"role": "assistant", "content": "calling", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{\"x\":1}"}},
            {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "user", "content": "继续"},
    ]
    system, out = to_anthropic_messages(msgs)
    assert system == [{"type": "text", "text": "be nice"}]
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"][1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}
    tu = [b for b in out[1]["content"] if b["type"] == "tool_use"]
    assert tu[0]["input"] == {"x": 1} and tu[1]["input"] == {"_raw": "not json"}
    kinds = [b["type"] for b in out[2]["content"]]
    assert kinds == ["tool_result", "tool_result", "text"]
    assert to_anthropic_tool_choice("required", None) == {"type": "any"}
    assert to_anthropic_tool_choice({"type": "function", "function": {"name": "a"}}, False) == {
        "type": "tool", "name": "a", "disable_parallel_tool_use": True}


def test_anthropic_stream_tool_calls():
    ad = AnthropicAdapter(offer("minimax-m3@opencode_go"), None)
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "f"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "1}"}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
    ]
    deltas = [d for e in events for d in ad.parse(e["type"], e)]
    tcs = [tc for d in deltas for tc in (d.tool_calls or [])]
    assert tcs[0] == {"index": 0, "id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}
    assert "".join(tc["function"].get("arguments", "") for tc in tcs) == '{"a":1}'
    assert deltas[-1].finish_reason == "tool_calls" and deltas[-1].usage.output == 7
    assert deltas[0].usage.input_cached == 5

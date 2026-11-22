"""意图分类黄金集。改分类规则必须跑这个。"""
import pytest

from thalamus.features import extract
from thalamus.intent import classify, inherit, parse_alias

ARTICLE = (
    "欧洲央行周四宣布维持利率不变，这是连续第三次按兵不动。央行表示，通胀率已回落至接近目标水平，"
    "但服务业价格仍有粘性。多位官员在会后发言中暗示，年内仍可能有一次降息。市场对此反应平淡，"
    "欧元兑美元小幅走低。分析人士普遍预计，下一次政策调整将在十二月的会议上讨论。"
) * 3

GOLDEN = [
    ("你好", "chat"),
    ("今天天气怎么样", "chat"),
    ("帮我写一首关于秋天的诗", "chat"),
    ("我最近压力很大，睡不好，怎么办", "chat"),
    ("给我讲个笑话吧", "chat"),
    ("Python 里怎么把列表去重并保持顺序？", "code"),
    ("```python\ndef f(x):\n    return x+1\n```\n这段代码有什么问题", "code"),
    (
        "Traceback (most recent call last):\n  File \"app.py\", line 3, in <module>\n    import foo\n"
        "ModuleNotFoundError: No module named 'foo'",
        "code",
    ),
    ("帮我写一个 React 组件，显示用户列表并支持分页", "code"),
    ("nginx 配置反向代理之后 SSE 一直被缓冲，怎么解决", "code"),
    ("Why does my Rust code panic here? fn main() { let v = vec![1]; println!(\"{}\", v[5]); }", "code"),
    ("docker 容器里 pip install 很慢，怎么换国内源", "code"),
    ("分析一下特斯拉和比亚迪在欧洲市场的竞争策略，对比优缺点", "reason"),
    ("为什么美联储加息会导致新兴市场货币贬值？请深入分析传导机制", "reason"),
    ("Compare the pros and cons of microservices vs monolith for a 5-person startup, and recommend a strategy", "reason"),
    ("证明：对任意正整数 n，n^3 - n 能被 6 整除", "reason"),
    ("我该接受 A 公司的 offer 还是留在现公司？请从薪资、成长、风险三个方面权衡利弊", "reason"),
    ("把下面这段话翻译成英文：我们下周二开会讨论预算。", "utility"),
    ("总结这篇文章的要点：" + ARTICLE, "utility"),
    ("Extract all email addresses from this text: contact john@a.com or mary@b.org", "utility"),
    ("帮我润色一下这段自我介绍：我叫小王，喜欢跑步和摄影。", "utility"),
    # 荷兰语（目标市场）
    ("Hoi, hoe gaat het?", "chat"),
    ("Kun je een verjaardagsgedicht voor mijn moeder schrijven?", "chat"),
    ("Waarom werkt mijn Python script niet? Ik krijg een foutmelding bij het importeren van pandas", "code"),
    ("Schrijf een functie in JavaScript die een lijst sorteert", "code"),
    ("Vergelijk de voor- en nadelen van huren en kopen in Amsterdam en geef een strategie", "reason"),
    ("Analyseer het risico van een renteverhoging voor mijn hypotheek, stap voor stap", "reason"),
    ("Vertaal deze zin naar het Engels: we vergaderen dinsdag over het budget.", "utility"),
]


def _intent(content, **body_kw):
    body = {"messages": [{"role": "user", "content": content}], **body_kw}
    return classify(extract(body, "k"), long_threshold=120_000)


@pytest.mark.parametrize("text,expected", GOLDEN)
def test_golden(text, expected):
    got = _intent(text)
    assert got.task == expected, f"{text[:40]!r}: got {got.task} {got.signals}"


def test_structure_vision():
    it = _intent([{"type": "text", "text": "图里写了什么"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}])
    assert it.task == "vision" and it.source == "structure"


def test_structure_agent_loop():
    body = {
        "messages": [
            {"role": "user", "content": "查一下巴黎天气"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "w", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
        "tools": [{"type": "function", "function": {"name": "w", "parameters": {}}}],
    }
    assert classify(extract(body, "k"), 120_000).task == "agent"


def test_tools_without_loop_is_not_agent():
    body = {"messages": [{"role": "user", "content": "你好呀"}], "tools": [{"type": "function", "function": {"name": "w"}}]}
    assert classify(extract(body, "k"), 120_000).task == "chat"


def test_previous_tool_call_does_not_turn_a_new_user_request_into_agent():
    messages = [
        {"role": "user", "content": "查一下天气"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "w", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "assistant", "content": "今天晴天。"},
    ]
    tools = [{"type": "function", "function": {"name": f"tool_{i}"}} for i in range(16)]
    for question, expected in (("你好", "chat"), ("Python 里怎么读文件？", "code")):
        f = extract({"messages": messages + [{"role": "user", "content": question}], "tools": tools}, "k")
        assert not f.in_tool_loop and f.loop_depth == 0
        assert classify(f, 120_000).task == expected


def test_agent_difficulty_uses_current_loop_depth_not_conversation_age():
    messages = []
    for _ in range(8):
        messages.extend([{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好"}])
    messages.extend([
        {"role": "user", "content": "查一下天气"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "w", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ])
    f = extract({"messages": messages, "tools": [{"type": "function", "function": {"name": "w"}}]}, "k")
    assert f.in_tool_loop and f.loop_depth == 1 and f.n_user_turns == 9
    assert classify(f, 120_000).label() == "agent/1"


def test_historical_image_keeps_vision_capability_without_forcing_vision_task():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "这张照片是什么？"},
                                      {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]},
        {"role": "assistant", "content": "这是一只猫。"},
    ]
    simple = extract({"messages": messages + [{"role": "user", "content": "你好"}]}, "k")
    assert simple.n_images == 1 and simple.last_user_images == 0
    assert classify(simple, 120_000).task == "chat"
    followup = extract({"messages": messages + [{"role": "user", "content": "图里是什么颜色？"}]}, "k")
    assert classify(followup, 120_000).task == "vision"


def test_later_turn_still_recognizes_a_new_translation_request():
    messages = [
        {"role": "user", "content": "早上好"}, {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "今天怎么样"}, {"role": "assistant", "content": "很好"},
        {"role": "user", "content": "翻译：我们明天见。"},
    ]
    assert classify(extract({"messages": messages}, "k"), 120_000).task == "utility"


@pytest.mark.parametrize("prompt", [
    "使用 Nyx 保险箱中的信用卡去预订停车场",
    "帮我用信用卡预订停车位",
    "请替我下单购买一张火车票",
    "Book a parking space using the card in my Nyx vault",
    "Can you reserve a parking spot for tonight?",
    "Use my card to pay for the reservation",
])
def test_first_turn_booking_and_payment_use_agent_when_tools_are_available(prompt):
    tools = [{"type": "function", "function": {"name": name}} for name in
             ("nyx_vault_read", "parking_search", "parking_book", "payment_confirm")]
    body = {"model": "thalamus/auto", "messages": [{"role": "user", "content": prompt}], "tools": tools}
    f = extract(body, "k")
    assert not f.in_tool_loop
    it = classify(f, 120_000)
    assert it.label() == "agent/1" and "action_intent" in it.signals


@pytest.mark.parametrize("prompt", [
    "如何用信用卡预订停车位？",
    "请问怎么预订停车场？",
    "请问预订停车场的规则",
    "请分析停车场预订方案",
    "请修复预订页面的代码",
    "How do I book a parking space with my card?",
    "Please explain how to reserve a parking spot",
    "Please explain booking and payment options",
    "Please fix the booking page code",
])
def test_booking_advice_is_not_a_tool_action(prompt):
    body = {"messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "function", "function": {"name": "parking_book"}}]}
    assert classify(extract(body, "k"), 120_000).task != "agent"


def test_booking_request_without_tools_does_not_claim_agent_capability():
    assert _intent("帮我用信用卡预订停车位").task != "agent"


def test_structure_long():
    it = _intent("请阅读：" + "字" * 200_000)
    assert it.task == "long"


def test_alias_and_effort():
    t, d, boost = parse_alias("thalamus/reason-3", None)
    assert (t, d) == ("reason", 3)
    assert parse_alias("thalamus/fast", None)[2] > 1
    assert parse_alias("gpt-4o", None) == (None, None, 1.0)
    assert _intent("写一个快排", reasoning_effort="high").difficulty == 3


def test_difficulty_scales():
    easy = _intent("python 怎么读文件")
    hard = _intent("帮我从零实现一个分布式任务队列，整个项目包括调度、重试、持久化，用 python 写，要求完整实现")
    assert easy.task == hard.task == "code"
    assert hard.difficulty > easy.difficulty


def _agent_body(user, *, n_tools, ctx_chars, depth=1):
    msgs = [{"role": "system", "content": "x" * ctx_chars}, {"role": "user", "content": user}]
    for i in range(depth):
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "3 failed"},
        ]
    return {
        "messages": msgs,
        "tools": [{"type": "function", "function": {"name": f"t{i}", "parameters": {"type": "object"}}}
                  for i in range(n_tools)],
    }


def test_difficulty_ignores_request_size():
    """2026-10-06：同一句"跑一下测试"，空上下文走 deepseek-flash，带 58K 上下文走
    kimi-k2.7-code —— 难度被"请求有多大"抬到 3，而难度 1→3 的每分价值跳 7.5 倍。
    请求大小已经按 token 实打实计价了，不能再让它抬难度。"""
    small = classify(extract(_agent_body("跑一下测试", n_tools=3, ctx_chars=0), "k"), 120_000)
    # 工具数量是客户端配置（编码 agent 每次都声明十几个），上下文长度是成本事实
    for n_tools, ctx in ((16, 0), (3, 220_000), (16, 220_000)):
        it = classify(extract(_agent_body("跑一下测试", n_tools=n_tools, ctx_chars=ctx), "k"), 120_000)
        assert it.task == "agent"
        assert it.difficulty == small.difficulty, (n_tools, ctx, it.label())


def test_difficulty_follows_loop_depth_and_scope():
    """真正该升档的两种证据：绕了很多步还没搞定、以及用户要求的就是大改。"""
    base = classify(extract(_agent_body("跑一下测试", n_tools=16, ctx_chars=220_000), "k"), 120_000)
    deep = classify(extract(_agent_body("跑一下测试", n_tools=16, ctx_chars=220_000, depth=16), "k"), 120_000)
    big = classify(extract(_agent_body("重构整个项目的鉴权模块", n_tools=16, ctx_chars=0, depth=6), "k"), 120_000)
    assert base.difficulty == 1 and deep.difficulty == 3 and big.difficulty == 3


@pytest.mark.parametrize(
    "text",
    [
        "欧洲央行维持利率不变" * 30_000,  # 无空格长中文：曾触发正则平方级回溯
        "The quick brown fox jumps. " * 40_000,
        "a" * 1_000_000,
        ("def f():\n    return 1\n" * 2000) + "ValueError: x",
    ],
    ids=["cjk-300k", "english-1m", "no-space-1m", "code-40k"],
)
def test_classify_is_fast_on_huge_inputs(text):
    import time

    body = {"messages": [{"role": "user", "content": text}]}
    f = extract(body, "k")
    classify(f, 120_000)
    t0 = time.perf_counter()
    for _ in range(5):
        classify(f, 120_000)
    per_call_ms = (time.perf_counter() - t0) / 5 * 1000
    assert per_call_ms < 8, per_call_ms


def test_stacktrace_and_filenames_still_detected():
    assert _intent("修改app.py 之后报 KeyError: 'user'，怎么回事").task == "code"
    assert _intent("  at com.example.Foo.bar(Foo.java:12)\n  at com.example.Main.main(Main.java:3)").task == "code"


def test_followup_inherits_previous_task():
    it = _intent("继续")
    assert it.confidence < 0.35
    inherited = inherit(it, "code", 2)
    assert inherited.task == "code" and inherited.difficulty == 2 and inherited.source == "session"
    assert inherit(_intent("帮我写一首诗"), "code", 2).task == "chat"

"""请求特征提取。

只做字符串/结构检查，不调用任何模型，常规请求 < 2ms。
不保存任何原文：会话 key 与问题指纹都是哈希。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

IMAGE_TOKENS = 1000
ANALYZE_TAIL = 20_000  # 只分析最后一条用户消息的末尾这么多字符


def estimate_tokens(text: str) -> int:
    """粗估 token。中日韩字符约 1.4 字/token，其他约 3.8 字符/token。

    用 UTF-8 字节数与字符数之差估算宽字符数量，避免逐字符 Python 循环。
    """
    if not text:
        return 0
    n = len(text)
    wide = (len(text.encode("utf-8", "ignore")) - n) // 2
    wide = max(0, min(wide, n))
    return int(wide / 1.4 + (n - wide) / 3.8) + 1


def content_text(content: Any) -> tuple[str, int]:
    """返回 (文本, 图片数)。兼容字符串与多模态数组。"""
    if content is None:
        return "", 0
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        parts: list[str] = []
        images = 0
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text"):
                parts.append(str(p.get("text", "")))
            elif t in ("image_url", "input_image", "image"):
                images += 1
        return "\n".join(parts), images
    return str(content), 0


def _h(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:n]


@dataclass
class Features:
    last_user: str
    prev_user: str
    est_tokens: int
    n_images: int
    last_user_images: int  # 最近一条用户消息里的图片；历史图片仍计入 n_images 供能力过滤
    has_tools: bool
    n_tools: int
    in_tool_loop: bool
    loop_depth: int  # 最后一条用户消息之后已经走了几步工具 —— 本轮的"深度"
    # 这个客户端会不会把 reasoning_content 回传？
    #   None  = 历史里还没有 assistant 轮次（第一轮，无从判断，也不会出问题）
    #   True  = 最后一条 assistant 消息带着它
    #   False = 带不动 —— 要求回传的思考模型会直接 400，必须在路由时排掉
    reasoning_echo: bool | None
    n_user_turns: int
    n_messages: int
    json_mode: bool
    reasoning_effort: str | None
    max_tokens: int | None
    session_key: str
    last_user_hash: str


def extract(body: dict[str, Any], key_id: str, explicit_session: str | None = None) -> Features:
    msgs = body.get("messages") or []
    tokens = 0
    images = 0
    last_user_images = 0
    users: list[str] = []
    system_head = ""
    last_user_at = -1
    last_assistant_echo: bool | None = None
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text, imgs = content_text(m.get("content"))
        images += imgs
        tokens += estimate_tokens(text) + 4
        if m.get("tool_calls"):
            tokens += estimate_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
        if role in ("system", "developer"):
            if len(system_head) < 2000:
                system_head += text[: 2000 - len(system_head)]
        elif role == "user":
            users.append(text)
            last_user_at = i
            last_user_images = imgs
        elif role == "assistant":
            # 只看最近一条：客户端要么整体回传、要么整体不回传，而上游检查的是最近那轮
            echo = m.get("reasoning_content") or m.get("reasoning")
            last_assistant_echo = bool(echo) and isinstance(echo, (str, list))

    tools = body.get("tools") or []
    if tools:
        tokens += estimate_tokens(json.dumps(tools, ensure_ascii=False))
    tokens += images * IMAGE_TOKENS

    # 工具调用只说明它所在的那一轮在循环。新 user 消息开启新一轮，不能因为
    # 前一轮的 tool/tool_calls 还留在最近六条历史里，就把简单问题判成 agent。
    current_turn = (m for m in msgs[last_user_at + 1:] if isinstance(m, dict))
    in_loop = any(m.get("role") == "tool" or m.get("tool_calls") for m in current_turn)
    # 本轮已经走了几步。比"声明了几个工具"诚实得多：工具数量是客户端的属性
    # （编码 agent 每次请求都声明十几个），而走了十几步说明这个任务真的不好办。
    depth = sum(1 for m in msgs[last_user_at + 1:] if isinstance(m, dict) and m.get("role") == "tool")

    rf = body.get("response_format") or {}
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")

    first_user = users[0] if users else ""
    # 会话 key 一律哈希并绑定到 key：客户端传的原始值不会进入上游请求头（防注入/触发熔断），
    # 不同 key 之间也不会串会话。
    if explicit_session:
        session = _h(f"{key_id}|explicit|{explicit_session}", 24)
    else:
        session = _h(f"{key_id}|{system_head}|{first_user[:2000]}", 24)
    last_user = users[-1] if users else ""
    return Features(
        last_user=last_user[-ANALYZE_TAIL:],
        prev_user=(users[-2] if len(users) > 1 else "")[-4000:],
        est_tokens=tokens,
        n_images=images,
        last_user_images=last_user_images,
        has_tools=bool(tools),
        n_tools=len(tools),
        in_tool_loop=in_loop,
        loop_depth=depth,
        reasoning_echo=last_assistant_echo,
        n_user_turns=len(users),
        n_messages=len(msgs),
        json_mode=isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"),
        reasoning_effort=str(effort).lower() if effort else None,
        max_tokens=int(max_tokens) if isinstance(max_tokens, (int, float)) and max_tokens > 0 else None,
        session_key=session,
        last_user_hash=_h(last_user),
    )

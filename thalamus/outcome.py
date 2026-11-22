"""回答质量的客观信号。

只在内存里检查、只留下标记，不保存任何内容：
  tool_args_invalid  工具调用参数不是合法 JSON（agent 最常见的静默失败）
  json_invalid       请求要求 JSON 输出，但返回的不是合法 JSON
  empty              正常结束却没有正文也没有工具调用（常见于 reasoning 预算不足）
  truncated          因长度上限被截断
以下标记由网关在其他环节打上，也汇总在这里：
  hedged             本次发生了对冲请求
  quota_cut          因每日额度被网关截断
  user_correction    下一轮用户在纠正这次回答（事后回写到这条记录）
  regenerated        下一轮用户重新生成了这个问题（事后回写到这条记录）
"""
from __future__ import annotations

import json
from typing import Any

from .types import Delta
from .verify import check_code_blocks, check_tool_args

TOOL_ARGS_INVALID = "tool_args_invalid"
TOOL_SCHEMA_INVALID = "tool_schema_invalid"
JSON_INVALID = "json_invalid"
CODE_INVALID = "code_invalid"
EMPTY = "empty"
TRUNCATED = "truncated"
HEDGED = "hedged"
VERIFY_RETRY = "verify_retry"
QUOTA_CUT = "quota_cut"
USER_CORRECTION = "user_correction"
REGENERATED = "regenerated"

ALL_FLAGS = (
    TOOL_ARGS_INVALID, TOOL_SCHEMA_INVALID, JSON_INVALID, CODE_INVALID, EMPTY,
    TRUNCATED, HEDGED, VERIFY_RETRY, QUOTA_CUT, USER_CORRECTION, REGENERATED,
)

# 学习模块扣分权重（相对 learning.penalty）。code_invalid 权重最低：它是三个检查里
# 误报可能性最大的一个（片段式代码很常见），所以只让它轻微影响排序。
LEARN_WEIGHT = {
    TOOL_ARGS_INVALID: 1.0,
    TOOL_SCHEMA_INVALID: 0.9,
    USER_CORRECTION: 0.7,
    JSON_INVALID: 0.6,
    EMPTY: 0.6,
    CODE_INVALID: 0.4,
}

# 这些标记算"这次回答是坏的"，用于给"答错的代价"定价（见 health.OutcomeBook）
BAD_FLAGS = frozenset({TOOL_ARGS_INVALID, TOOL_SCHEMA_INVALID, JSON_INVALID, CODE_INVALID, EMPTY, USER_CORRECTION})

MAX_CAPTURE = 256_000


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


class OutcomeTracker:
    def __init__(self, json_expected: bool, tools: Any = None, check_code: bool = False):
        self.json_expected = json_expected
        self.tools = tools if isinstance(tools, list) else None
        self.check_code = check_code
        self.capture_text = json_expected or check_code
        self.has_content = False
        self.detail = ""  # 违规细节，只进日志与管理台，不回传客户端
        self._text: list[str] = []
        self._text_len = 0
        self._args: dict[int, list[str]] = {}
        self._names: dict[int, str] = {}
        self._args_len = 0
        self.overflow = False

    def feed(self, d: Delta) -> None:
        if d.content:
            if d.content.strip():
                self.has_content = True
            if self.capture_text:
                if self._text_len + len(d.content) > MAX_CAPTURE:
                    self.overflow = True
                else:
                    self._text.append(d.content)
                    self._text_len += len(d.content)
        for tc in d.tool_calls or []:
            try:
                idx = int(tc.get("index", 0))
            except (TypeError, ValueError):
                idx = 0
            fn = tc.get("function") or {}
            if isinstance(fn.get("name"), str) and fn["name"]:
                self._names[idx] = fn["name"]  # 名字在第一帧给，参数分多帧给
            buf = self._args.setdefault(idx, [])
            args = fn.get("arguments") or ""
            if isinstance(args, str) and args:
                if self._args_len + len(args) > MAX_CAPTURE:
                    self.overflow = True
                else:
                    buf.append(args)
                    self._args_len += len(args)

    def flags(self, finish_reason: str | None, status: str) -> list[str]:
        out: list[str] = []
        if finish_reason == "length":
            out.append(TRUNCATED)
        if status != "ok" or finish_reason == "length" or self.overflow:
            return out  # 被截断/中断的输出天然不完整，不据此判定质量
        if not self.has_content and not self._args:
            out.append(EMPTY)

        bad_args = False
        for idx, parts in self._args.items():
            raw = "".join(parts).strip()
            if not raw:
                continue
            obj = _load(raw)
            if obj is _BAD:
                out.append(TOOL_ARGS_INVALID)
                bad_args = True
                break
            why = check_tool_args(self._names.get(idx, ""), obj, self.tools)
            if why:
                out.append(TOOL_SCHEMA_INVALID)
                self.detail = why
                bad_args = True
                break

        text = "".join(self._text) if self.capture_text else ""
        if self.json_expected and self.has_content and not self._args:
            if not _is_json(_strip_fence(text)):
                out.append(JSON_INVALID)
        if self.check_code and not bad_args and text:
            why = check_code_blocks(text)
            if why:
                out.append(CODE_INVALID)
                self.detail = self.detail or why
        return out


_BAD = object()


def _load(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError, TypeError):
        return _BAD


def _is_json(s: str) -> bool:
    return _load(s) is not _BAD


def parse_flags(raw: Any) -> list[str]:
    if not raw or not isinstance(raw, str):
        return []
    return [x for x in raw.split(",") if x]

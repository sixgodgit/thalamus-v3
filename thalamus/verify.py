"""本地验证器：不调用任何模型，只做纯结构检查，只留标记不留内容。

存在的理由：整套路由里唯一没被计价的东西是"答错的代价"。要给它定价就需要一个
奖励信号，而 Bandits-with-Knapsacks、Weitzman 的潘多拉盒子、Chow's rule 这些定理
都要求奖励可观测。这里造的就是那个信号 —— 三个便宜、客观、不违反"不存 prompt"
承诺的检查：

  tool_schema_invalid  工具参数是合法 JSON，但不符合客户端声明的 schema
                       （函数名不存在 / 缺必填 / 类型不对 / enum 不在范围内）
  json_invalid         请求要求 JSON，返回的不是（已有，这里只是复用）
  code_invalid         明确标了语言的代码块解析不过（python / json / yaml）

设计取向是**宁可漏报不可误报**：模型输出片段式代码（带 `...` 省略、REPL 回显、
diff）是完全正常的，所以这些形态一律跳过。漏掉一些真实错误不要紧，误报会污染
学习信号，那才是真的坏。
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

import yaml

# ── 工具参数 schema 校验 ──────────────────────────────────────────

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _type_ok(value: Any, spec: Any) -> bool:
    """只判断能确定的情况；schema 写得复杂（anyOf/oneOf/$ref 等）就放过。"""
    names = [spec] if isinstance(spec, str) else spec if isinstance(spec, list) else []
    names = [n for n in names if isinstance(n, str) and n in _JSON_TYPES]
    if not names:
        return True
    for n in names:
        if n in ("number", "integer") and isinstance(value, bool):
            continue  # Python 里 bool 是 int 的子类，别把 true 当成数字
        if n == "integer":
            # JSON 不区分 int 与 float：3.0 是合法整数，3.5 不是
            if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                return True
            continue
        if isinstance(value, _JSON_TYPES[n]):
            return True
    return False


def check_tool_args(name: str, args: Any, tools: Any) -> str:
    """返回违规原因（英文短语，只进日志不回传给客户端），合规返回空串。"""
    if not isinstance(tools, list) or not tools:
        return ""  # 客户端没声明 tools 就无从判断
    fns = {}
    for t in tools:
        fn = (t or {}).get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            fns[fn["name"]] = fn
    if not fns:
        return ""
    if name not in fns:
        return f"unknown function {name[:40]!r}"
    if not isinstance(args, dict):
        return ""  # 参数不是对象：schema 也可能不是 object，交给 JSON 合法性那一关
    schema = fns[name].get("parameters")
    if not isinstance(schema, dict):
        return ""
    if schema.get("type") not in (None, "object"):
        return ""
    props = schema.get("properties")
    props = props if isinstance(props, dict) else {}
    for key in schema.get("required") or []:
        if isinstance(key, str) and key not in args:
            return f"missing required {key[:40]!r}"
    if props and schema.get("additionalProperties") is False:
        extra = [k for k in args if k not in props]
        if extra:
            return f"unexpected {extra[0][:40]!r}"
    for key, value in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        if not _type_ok(value, spec.get("type")):
            return f"{key[:40]!r} should be {spec.get('type')}"
        enum = spec.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            return f"{key[:40]!r} not in enum"
    return ""


# ── 代码块语法检查 ───────────────────────────────────────────────

_FENCE = re.compile(r"```([A-Za-z0-9_+.\-]*)[ \t]*\r?\n(.*?)```", re.S)
# 这些形态说明这段本来就不是完整可解析的代码，跳过
_ELIDED = re.compile(r"^\s*(?:\.\.\.|…|#\s*\.\.\.|//\s*\.\.\.|/\*\s*\.\.\.|<\.\.\.>|#\s*(?:rest|snip|省略))", re.M | re.I)
_REPL = re.compile(r"^\s*(?:>>>|\.\.\.\s|\$\s|#\s*\$|In \[\d+\])", re.M)
_DIFFISH = re.compile(r"^[+-]{1,3}\s", re.M)
_PY = {"python", "py", "python3"}
_YAML = {"yaml", "yml"}
# 这些报错说明这段是被截断的片段，不是"模型写错了代码"
_TRUNCATED = ("was never closed", "unexpected eof", "incomplete input", "unterminated")


def check_code_blocks(text: str, limit: int = 12) -> str:
    """明确标了语言的代码块解析不过就返回原因。宁可漏报不可误报。"""
    if "```" not in text:
        return ""
    for i, m in enumerate(_FENCE.finditer(text)):
        if i >= limit:
            break
        lang, body = m.group(1).lower(), m.group(2)
        if not lang or not body.strip():
            continue
        if _ELIDED.search(body) or _REPL.search(body) or _DIFFISH.search(body):
            continue
        if lang in _PY:
            if body.count("\n") < 1:
                continue
            try:
                ast.parse(body)
            except SyntaxError as e:
                msg = str(e)
                if any(h in msg.lower() for h in _TRUNCATED):
                    continue
                return f"python block: {msg[:60]}"
            except (ValueError, RecursionError, MemoryError):
                return ""
        elif lang == "json":
            try:
                json.loads(body)
            except ValueError as e:
                return f"json block: {str(e)[:60]}"
        elif lang in _YAML:
            try:
                yaml.safe_load(body)
            except yaml.YAMLError as e:
                return f"yaml block: {str(e)[:60]}"
    return ""

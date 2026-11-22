"""极简 .env 加载器。

不依赖 python-dotenv。支持引号、`export ` 前缀和行尾注释（值与 # 之间需有空白）。
已经存在的环境变量不会被覆盖，这样 systemd / docker 注入的值优先。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_INLINE_COMMENT = re.compile(r"\s+#")


def load_env(path: str | os.PathLike[str]) -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] in "\"'" and val[-1] == val[0]:
            val = val[1:-1]
        else:
            val = _INLINE_COMMENT.split(val, 1)[0].strip()
        if key and key not in os.environ:
            os.environ[key] = val
            n += 1
    return n

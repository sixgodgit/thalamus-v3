"""运行时密钥。

控制台添加的渠道需要在运行期存下 API key，不能要求改 .env 再重启。
存放在 data/secrets.json，权限 0600（Windows 上 chmod 无效，靠目录权限）。
数据库备份走 VACUUM INTO，只包含 .db，所以密钥不会进每日备份。

查找顺序：运行时密钥 → 环境变量。也就是说 .env 里的值永远可以被控制台覆盖，
而控制台没设过的渠道仍按原来的方式从环境变量取。
"""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

log = logging.getLogger("thalamus.secrets")

NAME_MAX = 120


def mask(value: str) -> str:
    """只在管理接口回显这种形式，完整值永远不返回。"""
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}…{value[-4:]}"


class SecretStore:
    def __init__(self, path: str | os.PathLike[str] | None):
        self.path = Path(path) if path else None
        self._v: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._v = {str(k): v for k, v in raw.items() if isinstance(v, str) and v}
        except (OSError, ValueError):
            log.exception("读取 %s 失败，运行时密钥按空处理", self.path)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._v, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:  # Windows / 特殊文件系统
            pass

    def get(self, name: str) -> str | None:
        return self._v.get(name) or None

    def set(self, name: str, value: str) -> None:
        self._v[name] = value
        self._save()

    def delete(self, name: str) -> bool:
        if name not in self._v:
            return False
        del self._v[name]
        self._save()
        return True

    def names(self) -> list[str]:
        return sorted(self._v)

    def masked(self) -> dict[str, str]:
        return {k: mask(v) for k, v in sorted(self._v.items())}

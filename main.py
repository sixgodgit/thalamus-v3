"""Thalamus v3 启动入口。

    python main.py            # 读取同目录 .env，监听 THALAMUS_HOST:THALAMUS_PORT
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from thalamus.env import load_env

HERE = Path(__file__).resolve().parent
load_env(HERE / ".env")

logging.basicConfig(
    level=os.getenv("THALAMUS_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # 否则每个上游请求都打一行

from thalamus.app import create_app  # noqa: E402  (必须在 load_env 之后导入)

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("THALAMUS_HOST", "127.0.0.1"),
        port=int(os.getenv("THALAMUS_PORT", "9880")),
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("THALAMUS_TRUSTED_PROXIES", "127.0.0.1"),
    )

"""命令行管理 key（服务器上没有浏览器时用）。

    python scripts/manage.py create --name "张三" --daily 800 [--plan pro] [--rpm 60] [--concurrency 4]
    python scripts/manage.py list
    python scripts/manage.py set <key_id> --daily 1200
    python scripts/manage.py disable <key_id>
    python scripts/manage.py enable <key_id>

直接写数据库；网关进程在运行时，key 缓存最多 10 秒后生效。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thalamus.env import load_env  # noqa: E402

load_env(ROOT / ".env")

from thalamus.config import load_settings  # noqa: E402
from thalamus.quota import day_str  # noqa: E402
from thalamus.store import Store  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    s_ = sub.add_parser("set")
    s_.add_argument("key_id")
    for p in (c, s_):
        p.add_argument("--daily", type=float)
        p.add_argument("--rpm", type=int)
        p.add_argument("--concurrency", type=int)
        p.add_argument("--max-output", type=int)
        p.add_argument("--plan", choices=["standard", "pro"])
        p.add_argument("--exhaust", choices=["economy", "reject"])
        p.add_argument("--note")
    sub.add_parser("list")
    for name in ("disable", "enable"):
        sub.add_parser(name).add_argument("key_id")
    args = ap.parse_args()

    settings = load_settings()
    store = Store(settings.db_path)
    await store.connect()
    try:
        fields = {}
        if args.cmd in ("create", "set"):
            fields = {
                "daily_credits": args.daily, "rpm": args.rpm, "max_concurrency": args.concurrency,
                "max_output_tokens": args.max_output, "plan": args.plan, "exhaust_mode": args.exhaust, "note": args.note,
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            if fields.get("daily_credits", 0) < 0:
                sys.exit("--daily 不能为负（0 = 冻结）")
            for f_ in ("rpm", "max_concurrency"):
                if f_ in fields and fields[f_] < 1:
                    sys.exit(f"{f_} 至少为 1")
        if args.cmd == "create":
            k = settings.policy.keys
            defaults = {"daily_credits": k.daily_credits, "rpm": k.rpm, "max_concurrency": k.max_concurrency,
                        "max_output_tokens": k.max_output_tokens, "plan": k.plan, "exhaust_mode": k.exhaust_mode}
            key, raw = await store.create_key(args.name, defaults, **fields)
            print(f"已创建 {key.id}（{key.name}），每日 {key.daily_credits} credits")
            print(f"API key（只显示这一次）：{raw}")
        elif args.cmd == "list":
            usage = await store.usage_for_day(day_str(settings.policy.timezone))
            for key in await store.list_keys():
                u = usage.get(key.id, {})
                state = "启用" if key.enabled else "停用"
                print(f"{key.id}  {key.prefix}…  {key.name:<16} {state}  {key.plan:<8} 今日 {u.get('used', 0):.1f}/{key.daily_credits:g}  请求 {u.get('requests', 0)}")
        elif args.cmd == "set":
            key = await store.update_key(args.key_id, fields)
            print("未找到" if key is None else f"已更新 {key.id}")
        else:
            key = await store.update_key(args.key_id, {"enabled": args.cmd == "enable"})
            print("未找到" if key is None else f"{key.id} 已{'启用' if key.enabled else '停用'}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())

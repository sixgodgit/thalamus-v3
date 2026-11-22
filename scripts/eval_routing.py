"""路由分类离线评测（免费、秒级）。

    python scripts/eval_routing.py                          # 默认评测集
    python scripts/eval_routing.py eval/my_cases.jsonl --min 0.9 --json report.json

任务准确率低于 --min 时退出码为 1，可以接进 CI。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thalamus.evaluation import format_routing_report, load_jsonl, run_routing_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="?", default=str(ROOT / "eval" / "routing_cases.jsonl"))
    ap.add_argument("--min", type=float, default=0.0, help="最低任务准确率，低于则退出码 1")
    ap.add_argument("--json", help="把完整报告写成 JSON")
    ap.add_argument("--long-threshold", type=int, default=120_000)
    args = ap.parse_args()

    report = run_routing_eval(load_jsonl(args.cases), args.long_threshold)
    print(format_routing_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["task_accuracy"] >= args.min else 1


if __name__ == "__main__":
    sys.exit(main())

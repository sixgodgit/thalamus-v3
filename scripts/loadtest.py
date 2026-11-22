"""压测：真实网关进程 + 本地模拟上游，测网关在并发下附加的延迟与吞吐。

    python scripts/loadtest.py
    python scripts/loadtest.py --levels 1,50,200 --requests 600 --ttft-ms 400 --chunks 60 --chunk-ms 10 --mix chat

做法：
  1. 在本进程起一个模拟上游（OpenAI /chat/completions 与 Anthropic /messages 两种流式格式），
     首 token 延迟、分片数、分片间隔可配
  2. 复制 config/ 到临时目录，把所有渠道的 base_url 指向模拟上游，另起一个真实网关子进程
  3. 同样的请求分别直连模拟上游（基线）和经过网关，逐并发档位对比首 token 与总耗时
网关开销 ≈ 经过网关的首 token − 基线首 token。
客户端、模拟上游、网关是三个独立进程（Python 单进程只能用一个核）。表里的 CPU% 是该进程占一个核的比例：
接近 100% 说明它先饱和了——如果是"客户端"饱和，那一档的数字不可信，请把客户端放到另一台机器。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ───────────────────────────── 模拟上游 ─────────────────────────────


def mock_app(ttft: float, chunks: int, chunk_gap: float) -> Starlette:
    async def openai(request: Request):
        body = await request.json()
        model = body.get("model", "m")

        async def gen():
            await asyncio.sleep(ttft)
            for i in range(chunks):
                d = {"choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}], "model": model}
                yield f"data: {json.dumps(d)}\n\n"
                if chunk_gap:
                    await asyncio.sleep(chunk_gap)
            yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield f'data: {{"choices":[],"usage":{{"prompt_tokens":1000,"completion_tokens":{chunks},"prompt_cache_hit_tokens":800,"prompt_cache_miss_tokens":200}}}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def anthropic(request: Request):
        await request.body()

        def ev(t, obj):
            return f"event: {t}\ndata: {json.dumps({'type': t, **obj})}\n\n"

        async def gen():
            yield ev("message_start", {"message": {"usage": {"input_tokens": 200, "cache_read_input_tokens": 800}}})
            await asyncio.sleep(ttft)
            yield ev("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
            for i in range(chunks):
                yield ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": f"tok{i} "}})
                if chunk_gap:
                    await asyncio.sleep(chunk_gap)
            yield ev("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": chunks}})
            yield ev("message_stop", {})

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[
        Route("/chat/completions", openai, methods=["POST"]),
        Route("/messages", anthropic, methods=["POST"]),
    ])


# ───────────────────────────── 请求体 ─────────────────────────────

TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "do something " * 8,
          "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}} for i in range(20)]
BODIES = {
    "chat": {"messages": [{"role": "user", "content": "你好，今天过得怎么样？"}]},
    "code": {"messages": [{"role": "user", "content": "这段代码为什么报错\n```python\n" + "def f(x):\n    return g(x)\n" * 60 + "```"}]},
    "agent": {"messages": [{"role": "system", "content": "你是助手。" * 15000},
                           {"role": "user", "content": "整理下周日程"},
                           {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "tool_1", "arguments": "{}"}}]},
                           {"role": "tool", "tool_call_id": "c", "content": "结果" * 1500}], "tools": TOOLS},
}
MIXES = {"chat": ["chat"], "mixed": ["chat"] * 7 + ["code"] * 2 + ["agent"]}


# ───────────────────────────── 客户端 ─────────────────────────────


async def one(client: httpx.AsyncClient, url: str, headers: dict, body: dict) -> tuple[float | None, float, str | None]:
    """返回 (首 token 秒, 总秒, 错误描述或 None)。"""
    t0 = time.perf_counter()
    ttft = None
    try:
        async with client.stream("POST", url, json=body, headers=headers) as r:
            if r.status_code != 200:
                raw = (await r.aread()).decode("utf-8", "replace")
                return None, time.perf_counter() - t0, f"HTTP {r.status_code}: {raw[:160]}"
            async for line in r.aiter_lines():
                if ttft is None and line.startswith("data: {") and '"content"' in line:
                    ttft = time.perf_counter() - t0
                if line.startswith("data: {\"error\""):
                    return ttft, time.perf_counter() - t0, f"stream error: {line[6:160]}"
        return ttft, time.perf_counter() - t0, None if ttft is not None else "no content in stream"
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - t0, f"{type(e).__name__}: {e}"[:160]


async def run_level(url: str, headers: dict, concurrency: int, n: int, mix: list[str], model: str | None) -> dict:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency + 10)
    results: list[tuple[float | None, float, str | None]] = []
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        async def task(i: int) -> None:
            body = dict(BODIES[mix[i % len(mix)]], stream=True)
            if model:
                body["model"] = model
            async with sem:
                results.append(await one(client, url, headers, body))

        t0 = time.perf_counter()
        await asyncio.gather(*(task(i) for i in range(n)))
        wall = time.perf_counter() - t0
    ok = [r for r in results if r[2] is None]
    err_samples = [r[2] for r in results if r[2] is not None][:5]
    ttfts = sorted(r[0] for r in ok)
    totals = sorted(r[1] for r in ok)

    def pct(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(len(xs) * p))] * 1000 if xs else float("nan")

    return {
        "ok": len(ok), "errors": n - len(ok), "err_samples": err_samples, "rps": len(ok) / wall,
        "ttft_p50": pct(ttfts, .5), "ttft_p95": pct(ttfts, .95), "ttft_p99": pct(ttfts, .99),
        "total_p50": pct(totals, .5), "total_p99": pct(totals, .99),
        "ttft_mean": statistics.mean(ttfts) * 1000 if ttfts else float("nan"),
    }


# ───────────────────────────── 主流程 ─────────────────────────────


async def overhead_hist(gw: str, admin: str) -> dict[str, float]:
    """读取网关 /metrics 里路由开销直方图的累计桶。"""
    async with httpx.AsyncClient(timeout=10) as c:
        text = (await c.get(gw + "/metrics", headers={"Authorization": f"Bearer {admin}"})).text
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("thalamus_routing_overhead_seconds_bucket"):
            le = line.split('le="', 1)[1].split('"', 1)[0]
            out[le] = float(line.split()[-1])
    return out


def hist_quantile(before: dict[str, float], after: dict[str, float], q: float) -> str:
    buckets = sorted(after, key=lambda k: float("inf") if k == "+Inf" else float(k))
    total = after.get("+Inf", 0) - before.get("+Inf", 0)
    if total <= 0:
        return "—"
    for le in buckets:
        if after[le] - before.get(le, 0) >= q * total:
            return "∞" if le == "+Inf" else f"{float(le) * 1000:g}"
    return "∞"


def cpu_seconds(proc, children: bool = True) -> float:
    """进程（含子进程）累计 CPU 秒。Windows 的 venv python.exe 是启动器，真正的解释器是它的子进程。"""
    try:
        import psutil

        p = psutil.Process(proc if isinstance(proc, int) else proc.pid)
        procs = [p] + (p.children(recursive=True) if children else [])
        total = 0.0
        for q in procs:
            try:
                t = q.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total
    except Exception:  # noqa: BLE001 - 没装 psutil 就不报 CPU
        return float("nan")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,20,100")
    ap.add_argument("--requests", type=int, default=300, help="每档请求数")
    ap.add_argument("--ttft-ms", type=float, default=300)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--chunk-ms", type=float, default=5)
    ap.add_argument("--mix", choices=sorted(MIXES), default="mixed")
    ap.add_argument("--keep", action="store_true", help="保留临时目录（含网关日志与数据库）")
    ap.add_argument("--warm", action="store_true", help="网关启动时预热上游连接（生产默认开启）")
    ap.add_argument("--serve-mock", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.serve_mock:  # 子进程模式：只跑模拟上游
        await uvicorn.Server(uvicorn.Config(mock_app(args.ttft_ms / 1000, args.chunks, args.chunk_ms / 1000),
                                            host="127.0.0.1", port=args.serve_mock, log_level="warning",
                                            access_log=False)).serve()
        return 0

    levels = [int(x) for x in args.levels.split(",")]
    mock_port, gw_port = free_port(), free_port()
    mock_url = f"http://127.0.0.1:{mock_port}"
    mock_proc = subprocess.Popen(
        [sys.executable, __file__, "--serve-mock", str(mock_port), "--ttft-ms", str(args.ttft_ms),
         "--chunks", str(args.chunks), "--chunk-ms", str(args.chunk_ms)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    tmp = Path(tempfile.mkdtemp(prefix="thalamus-load-"))
    cfg = tmp / "config"
    cfg.mkdir()
    up = yaml.safe_load((ROOT / "config" / "upstreams.yaml").read_text(encoding="utf-8"))
    for ch in up["channels"].values():
        ch["base_url"] = mock_url
        ch["max_concurrency"] = 100000
        if ch.get("budget"):
            for w in ch["budget"]["windows"]:
                w["limit"] = 1e12  # 压测不应被订阅窗口限流
    (cfg / "upstreams.yaml").write_text(yaml.safe_dump(up, allow_unicode=True), encoding="utf-8")
    shutil.copy(ROOT / "config" / "policy.yaml", cfg / "policy.yaml")

    admin = secrets.token_hex(16)
    env = {**os.environ,
           "THALAMUS_CONFIG_DIR": str(cfg), "THALAMUS_DB": str(tmp / "load.db"), "THALAMUS_PORT": str(gw_port),
           "THALAMUS_HOST": "127.0.0.1", "THALAMUS_ADMIN_TOKEN": admin, "THALAMUS_WARM": "1" if args.warm else "0",
           "THALAMUS_LOG_LEVEL": "WARNING", "DEEPSEEK_API_KEY": "load", "GLM_PLAN_API_KEY": "load",
           "OPENCODE_GO_API_KEY": "load"}
    log_file = open(tmp / "gateway.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    gw = f"http://127.0.0.1:{gw_port}"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            for url in (mock_url + "/nope", gw + "/health"):
                for _ in range(100):
                    try:
                        await c.get(url)
                        break
                    except httpx.HTTPError:
                        await asyncio.sleep(0.2)
                else:
                    print("启动失败，网关日志：", (tmp / "gateway.log").read_text(encoding="utf-8")[-2000:])
                    return 1
            r = await c.post(gw + "/admin/api/keys", headers={"Authorization": f"Bearer {admin}"},
                             json={"name": "load", "daily_credits": 1e12, "rpm": 10_000_000, "max_concurrency": 100000,
                                   "max_output_tokens": 0})
            key = r.json()["api_key"]

        mix = MIXES[args.mix]
        print(f"模拟上游: 首 token {args.ttft_ms:g}ms，{args.chunks} 片 × {args.chunk_ms:g}ms；流量组成: {args.mix}；每档 {args.requests} 个请求\n")

        # 冷启动：进程启动后的第一个请求（首次查 key、加载账本、建立上游连接；压测关闭了连接预热）
        H = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(timeout=60) as c:
            cold = await one(c, gw + "/v1/chat/completions", H, dict(BODIES["chat"], stream=True))
            warm_base = await one(c, mock_url + "/chat/completions", {}, dict(BODIES["chat"], stream=True, model="m"))
            for i in range(10):  # 预热，不计入统计
                await one(c, gw + "/v1/chat/completions", H, dict(BODIES[mix[i % len(mix)]], stream=True))
                await one(c, mock_url + "/chat/completions", {}, dict(BODIES[mix[i % len(mix)]], stream=True, model="m"))
        if cold[0] is not None and warm_base[0] is not None:
            print(f"冷启动首个请求：首 token {cold[0] * 1000:.0f} ms（同条件直连上游 {warm_base[0] * 1000:.0f} ms）\n")
        print(f"{'并发':>4} | {'基线首token p50/p99':>19} | {'网关首token p50/p95/p99':>25} | {'开销 p50/p99':>13} | "
              f"{'RPS':>5} | {'错误':>2} | {'CPU 网关/上游/客户端':>18}")
        print("-" * 116)
        me = os.getpid()
        any_errors = 0
        for level in levels:
            base = await run_level(mock_url + "/chat/completions", {}, level, args.requests, mix, "m")
            h0 = await overhead_hist(gw, admin)
            g0, m0, c0, t0 = cpu_seconds(proc), cpu_seconds(mock_proc), cpu_seconds(me, False), time.perf_counter()
            gwr = await run_level(gw + "/v1/chat/completions", {"Authorization": f"Bearer {key}"}, level, args.requests, mix, None)
            wall = time.perf_counter() - t0
            gcpu = (cpu_seconds(proc) - g0) / wall * 100
            mcpu = (cpu_seconds(mock_proc) - m0) / wall * 100
            ccpu = (cpu_seconds(me, False) - c0) / wall * 100
            flag = "  ← 客户端/上游饱和，本档数字不可信" if max(mcpu, ccpu) > 80 else ""
            any_errors += gwr["errors"]
            print(f"{level:>6} | {base['ttft_p50']:>7.1f} / {base['ttft_p99']:>7.1f} ms | "
                  f"{gwr['ttft_p50']:>7.1f} / {gwr['ttft_p95']:>6.1f} / {gwr['ttft_p99']:>6.1f} ms | "
                  f"{gwr['ttft_p50'] - base['ttft_p50']:>5.1f} / {gwr['ttft_p99'] - base['ttft_p99']:>5.1f} ms | "
                  f"{gwr['rps']:>5.1f} | {gwr['errors']:>4} | {gcpu:>5.0f}% / {mcpu:>4.0f}% / {ccpu:>4.0f}%{flag}")
            h1 = await overhead_hist(gw, admin)
            p50, p99 = hist_quantile(h0, h1, 0.5), hist_quantile(h0, h1, 0.99)
            print(f"       网关内部路由开销（进入→发出上游请求）p50 ≤ {p50} ms，p99 ≤ {p99} ms")
            for s in gwr["err_samples"]:
                print(f"       错误样本: {s}")

        return 1 if any_errors else 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        mock_proc.terminate()
        try:
            mock_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock_proc.kill()
        if args.keep:
            print(f"临时目录: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

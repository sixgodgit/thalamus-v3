"""延迟分解：把"慢"拆成网络距离 / 上游自己 / 网关 / 反代缓冲四段。

    # 在网关那台机器上（能读 .env，直接量上游网络距离）
    python scripts/latency.py

    # 再从你真正用客户端的那台机器上量一遍（看反代有没有攒包）
    python scripts/latency.py --via https://api.example.com --key sk-thl-xxx

两次结果的差就是"你和网关之间"那一段。只读不写，不产生可观的费用
（上游探测只打 GET /models；端到端那条生成约 60 token）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thalamus.env import load_env  # noqa: E402

load_env(ROOT / ".env")

from thalamus.config import load_settings  # noqa: E402

PROMPT = "从 1 数到 20，每个数字之间用空格隔开，不要解释。"


async def probe_upstreams(rounds: int) -> None:
    """量网络距离：connect / TLS / 首字节，各打几次取中位数。

    这一段网关无法优化（除了复用连接），它是地理距离。荷兰 → 北京单程 RTT
    约 80-120ms，TLS 握手还要再两个往返；荷兰 → 美东约 40ms。
    """
    import httpx

    s = load_settings()
    chans = [c for c in s.channels.values() if c.configured and c.enabled]
    if not chans:
        print("没有配置好的渠道（.env 里缺上游 key），跳过网络距离探测\n")
        return
    print(f"【① 网络距离】网关 → 各上游 × {rounds} 次取中位数")
    print(f"  {'渠道':<14}{'区域':<6}{'TCP':>7}{'TLS':>7}{'冷连接首字节':>14}{'复用连接':>10}  base_url")
    for ch in chans:
        url = httpx.URL(ch.base_url)
        host, port = url.host, url.port or (443 if url.scheme == "https" else 80)
        tcp: list[float] = []
        tls: list[float] = []
        cold: list[float] = []
        warm: list[float] = []
        err = ""
        for _ in range(rounds):
            try:
                # TCP 与 TLS 分开量：跨洲时 TLS 握手要再两个往返，是冷连接最大的一笔
                t0 = time.perf_counter()
                reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 20)
                t1 = time.perf_counter()
                writer.close()
                await writer.wait_closed()
                tcp.append((t1 - t0) * 1000)
                if url.scheme == "https":
                    import ssl

                    t0 = time.perf_counter()
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port, ssl=ssl.create_default_context(),
                                                server_hostname=host), 20)
                    tls.append((time.perf_counter() - t0) * 1000)
                    writer.close()
                    await writer.wait_closed()

                h = {"Authorization": f"Bearer {ch.api_key}", "User-Agent": ch.user_agent}
                async with httpx.AsyncClient(base_url=ch.base_url, timeout=20.0) as cli:
                    t0 = time.perf_counter()
                    await cli.get("/models", headers=h)        # 冷：含 TCP+TLS
                    cold.append((time.perf_counter() - t0) * 1000)
                    t0 = time.perf_counter()
                    await cli.get("/models", headers=h)        # 热：复用同一条连接
                    warm.append((time.perf_counter() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:60]}"
                break

        def med(xs):
            return statistics.median(xs) if xs else float("nan")

        if err and not warm:
            print(f"  {ch.id:<14}{ch.region:<6}  探测失败：{err}")
            continue
        print(f"  {ch.id:<14}{ch.region:<6}{med(tcp):>6.0f}ms{med(tls):>6.0f}ms"
              f"{med(cold):>13.0f}ms{med(warm):>9.0f}ms  {ch.base_url}")
    print("  线上走的是最后一列（连接池 keepalive 120s，启动预热、每 45 秒保温）。")
    print("  这一段是地理距离，网关只能复用连接、不能缩短它。\n")


async def end_to_end(via: str, key: str, rounds: int) -> None:
    """量端到端，并按分片到达时刻判断中间有没有攒包。"""
    import httpx

    print(f"【② 端到端】{via}  × {rounds} 次")
    body = {"model": "thalamus/auto", "stream": True, "messages": [{"role": "user", "content": PROMPT}]}
    async with httpx.AsyncClient(base_url=via.rstrip("/"), timeout=httpx.Timeout(120.0)) as cli:
        for i in range(rounds):
            t0 = time.perf_counter()
            first = None
            gaps: list[float] = []
            last = None
            served, route_ms = "", ""
            try:
                async with cli.stream("POST", "/v1/chat/completions", json=body, headers={
                        "Authorization": f"Bearer {key}", "Content-Type": "application/json"}) as r:
                    served = r.headers.get("x-thalamus-model", "?")
                    route_ms = r.headers.get("x-thalamus-route-ms", "?")
                    if r.status_code >= 400:
                        print(f"  第 {i + 1} 次  HTTP {r.status_code}: {(await r.aread())[:200]!r}")
                        continue
                    async for line in r.aiter_lines():
                        if not line.startswith("data: ") or line.endswith("[DONE]"):
                            continue
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        elif last is not None:
                            gaps.append((now - last) * 1000)
                        last = now
            except Exception as e:  # noqa: BLE001
                print(f"  第 {i + 1} 次  失败：{type(e).__name__}: {str(e)[:80]}")
                continue
            if first is None:
                print(f"  第 {i + 1} 次  一个分片都没收到")
                continue
            ttft, total = (first - t0) * 1000, (time.perf_counter() - t0) * 1000
            n = len(gaps) + 1
            med = statistics.median(gaps) if gaps else 0.0
            # 分片之间几乎没有间隔 = 它们是一起到的 = 中间有人攒包
            buffered = n >= 8 and med < 1.5
            flag = "  ⚠ 分片是一起到的，中间有人在缓冲（反代 / CDN）" if buffered else ""
            print(f"  第 {i + 1} 次  首字 {ttft:>6.0f}ms  全程 {total:>6.0f}ms  分片 {n:>3}  "
                  f"分片间隔中位数 {med:>5.1f}ms  路由判断 {route_ms}ms  由 {served} 服务{flag}")
    print("  首字 − 路由判断 − 网络距离 ≈ 上游自己的思考时间，网关管不到那一段。")
    print("  分片间隔中位数应该在 10-80ms（真的在逐字流）。接近 0 且分片很多 = 被攒包了：")
    print("    nginx 要 proxy_buffering off（见 deploy/nginx.conf）；Caddy 要 flush_interval -1。\n")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--via", help="网关地址；给了就做端到端测量")
    ap.add_argument("--key", help="Thalamus API key，--via 时必填")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--skip-upstreams", action="store_true", help="不量网络距离（没有上游 key 时用）")
    args = ap.parse_args()

    if not args.skip_upstreams:
        await probe_upstreams(args.rounds)
    if args.via:
        if not args.key:
            sys.exit("--via 需要 --key")
        await end_to_end(args.via, args.key, args.rounds)
    elif args.skip_upstreams:
        sys.exit("什么都没要做：给 --via/--key，或者去掉 --skip-upstreams")


if __name__ == "__main__":
    asyncio.run(main())

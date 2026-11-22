"""Prometheus 文本格式指标（无第三方依赖）。

GET /metrics，需 Bearer THALAMUS_METRICS_TOKEN（未设置时用管理口令）。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Engine

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, math.inf)
OVERHEAD_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, math.inf)

Labels = tuple[tuple[str, str], ...]


def _labels(d: dict[str, object]) -> Labels:
    return tuple(sorted((k, str(v)) for k, v in d.items() if v is not None))


def _fmt(labels: Iterable[tuple[str, str]]) -> str:
    items = [f'{k}="{_esc(v)}"' for k, v in labels]
    return "{" + ",".join(items) + "}" if items else ""


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Histogram:
    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float) -> None:
        self.sum += v
        self.n += 1
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1


HELP = {
    "thalamus_requests_total": "Requests by final status, task and model",
    "thalamus_credits_total": "Credits charged to keys (list-price based)",
    "thalamus_upstream_usd_total": "Real metered upstream spend in USD",
    "thalamus_subscription_units_total": "Subscription budget units consumed",
    "thalamus_tokens_total": "Tokens by kind",
    "thalamus_attempts_total": "Upstream attempts started",
    "thalamus_outcome_flags_total": "Quality/outcome flags by model",
    "thalamus_ttft_seconds": "Time to first token seen by the client",
    "thalamus_routing_overhead_seconds": "Gateway time from request start to first upstream dispatch",
}


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, dict[Labels, float]] = defaultdict(lambda: defaultdict(float))
        self.hists: dict[str, dict[Labels, Histogram]] = defaultdict(dict)

    def inc(self, name: str, value: float = 1.0, **labels: object) -> None:
        self.counters[name][_labels(labels)] += value

    def observe(self, name: str, value: float, buckets: tuple[float, ...], **labels: object) -> None:
        key = _labels(labels)
        h = self.hists[name].get(key)
        if h is None:
            h = self.hists[name][key] = Histogram(buckets)
        h.observe(value)

    def render(self, engine: "Engine") -> str:
        lines: list[str] = []
        for name, series in sorted(self.counters.items()):
            lines.append(f"# HELP {name} {HELP.get(name, name)}")
            lines.append(f"# TYPE {name} counter")
            for labels, v in sorted(series.items()):
                lines.append(f"{name}{_fmt(labels)} {v:g}")
        for name, series in sorted(self.hists.items()):
            lines.append(f"# HELP {name} {HELP.get(name, name)}")
            lines.append(f"# TYPE {name} histogram")
            for labels, h in sorted(series.items(), key=lambda kv: kv[0]):
                for b, c in zip(h.buckets, h.counts):
                    le = "+Inf" if math.isinf(b) else f"{b:g}"
                    lines.append(f"{name}_bucket{_fmt(labels + (('le', le),))} {c}")
                lines.append(f"{name}_sum{_fmt(labels)} {h.sum:.6f}")
                lines.append(f"{name}_count{_fmt(labels)} {h.n}")

        # 实时状态（gauge）
        lines += ["# TYPE thalamus_inflight gauge"]
        for ch, n in sorted(engine.channel_load.items()):
            lines.append(f'thalamus_inflight{{channel="{_esc(ch)}"}} {n}')
        lines += ["# TYPE thalamus_sessions gauge", f"thalamus_sessions {len(engine.sessions)}"]
        lines += ["# TYPE thalamus_breaker_open gauge"]
        for key, h in sorted(engine.health.snapshot().items()):
            lines.append(f'thalamus_breaker_open{{offer="{_esc(key)}"}} {int(h["state"] == "open")}')
        lines += ["# TYPE thalamus_subscription_window_ratio gauge"]
        for scope, snap in sorted(engine.capacity.snapshot().items()):
            for w in snap["windows"]:
                lines.append(
                    f'thalamus_subscription_window_ratio{{scope="{_esc(scope)}",hours="{w["hours"]:g}"}} {w["ratio"]}'
                )
        lines += ["# TYPE thalamus_metered_usd_today gauge"]
        for ch, usd in sorted(engine.metered_by_channel().items()):
            lines.append(f'thalamus_metered_usd_today{{channel="{_esc(ch)}"}} {usd:.6f}')
        return "\n".join(lines) + "\n"

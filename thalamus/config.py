"""配置加载与校验。

config/upstreams.yaml  渠道（DeepSeek / GLM Coding Plan / OpenCode Go）、模型能力、每个渠道提供哪些模型
config/policy.yaml     任务价值、配额默认值、订阅节奏、熔断、学习
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("thalamus.config")

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("utility", "chat", "code", "agent", "reason", "vision", "long")
DIALECTS = {"openai_chat": "/chat/completions", "anthropic": "/messages"}


class ConfigError(Exception):
    pass


# 运行时密钥解析器（由 app 启动时注入 SecretStore.get）。控制台加的渠道靠它拿 key，
# 不必改 .env 重启；没注入或没设过的名字仍回落到环境变量。
_secret_resolver: Any = None


def set_secret_resolver(fn: Any) -> None:
    global _secret_resolver
    _secret_resolver = fn


def _one_of(value: Any, allowed: tuple[str, ...], where: str) -> str:
    v = str(value).strip().lower()
    if v not in allowed:
        raise ConfigError(f"{where} 只能是 {'/'.join(allowed)}，收到 {value!r}")
    return v


def _truthy(value: str | None) -> bool:
    """环境变量的布尔约定：1/true/yes/on 为真，其余（含空）为假。"""
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def fmt_ctx(tokens: int) -> str:
    """上下文长度的人类写法：262144 → 262k，1000000 → 1M。"""
    return f"{tokens / 1_000_000:g}M" if tokens >= 1_000_000 else f"{round(tokens / 1000)}k"


# ───────────────────────────── 时间系数 ─────────────────────────────


@dataclass(frozen=True)
class TimeWindow:
    weekdays: frozenset[int]
    start: int
    end: int
    multiplier: float


@dataclass
class TimeFactor:
    """按 UTC 星期与小时给出倍率。用于峰谷价格与 GLM 的非高峰积分减半。"""

    default: float = 1.0
    windows: list[TimeWindow] = field(default_factory=list)

    def at(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        for w in self.windows:
            if now.weekday() in w.weekdays and w.start <= now.hour < w.end:
                return w.multiplier
        return self.default


def _time_factor(raw: dict[str, Any] | None) -> TimeFactor:
    if not raw:
        return TimeFactor()
    wins = []
    for w in raw.get("windows") or []:
        start, end = w["hours_utc"]
        wins.append(
            TimeWindow(
                frozenset(w.get("weekdays", range(7))),
                int(start),
                int(end),
                float(w["multiplier"]),
            )
        )
    return TimeFactor(float(raw.get("default", 1.0)), wins)


# ───────────────────────────── 上游 ─────────────────────────────


@dataclass(frozen=True)
class Price:
    """USD / 1M tokens。用作积分系数时单位是"每 token 积分 ×10000"。"""

    input: float
    cached: float
    output: float

    def per_million(self, miss: float, cached: float, out: float) -> float:
        return (miss * self.input + cached * self.cached + out * self.output) / 1_000_000


def _price(raw: dict[str, Any] | None) -> Price | None:
    if not raw:
        return None
    return Price(float(raw["input"]), float(raw.get("cached", raw["input"])), float(raw["output"]))


@dataclass
class BudgetWindow:
    hours: float
    limit: float


@dataclass
class Budget:
    unit: str  # points | percent
    scope: str  # shared | per_model
    windows: list[BudgetWindow]
    time: TimeFactor
    exhausted_cooldown_s: float


@dataclass
class ModelSpec:
    key: str
    display: str
    context: int
    max_output: int
    vision: bool
    tools: bool
    json_mode: bool  # 支持 response_format（json_object / json_schema）
    reasoning: str  # none | hybrid | always
    tier: str  # standard | pro
    ttft_s: float
    tps: float
    quality: dict[str, float]
    # 思考型模型里有一类要求把上一轮的 reasoning_content 原样回传，否则 400
    # （GLM："The reasoning_content in the thinking mode must be passed back to the API"）。
    # 而 reasoning_content 不是 OpenAI 规范里的字段，多数客户端不会存、更不会回传，
    # 于是多轮对话从第二轮起必然失败。默认跟 reasoning: always 走，可在 upstreams.yaml 里逐个覆盖。
    needs_reasoning_echo: bool = False

    def q(self, task: str) -> float:
        if task in self.quality:
            return self.quality[task]
        return self.quality.get("chat", 6.0) - 0.5


@dataclass
class Channel:
    id: str
    kind: str  # metered | subscription
    base_url: str
    api_key_env: str
    dialect: str
    region: str
    max_concurrency: int
    user_agent: str
    headers: dict[str, str]
    pricing_time: TimeFactor
    budget: Budget | None
    enabled: bool
    connect_timeout_s: float

    @property
    def api_key(self) -> str:
        if _secret_resolver is not None:
            v = _secret_resolver(self.api_key_env)
            if v:
                return v
        return os.getenv(self.api_key_env, "")

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.api_key)


@dataclass
class Offer:
    """某个渠道上的某个模型 —— 路由的最小候选单位。"""

    channel: Channel
    model: ModelSpec
    upstream: str
    dialect: str
    path: str
    price: Price  # 按量渠道 = 实付价；订阅渠道 = 公开标价（用于配额折算与影子价格）
    pricing_time: TimeFactor
    points: Price | None
    monthly_usd: float | None
    think_tags: bool
    min_max_tokens: int
    extra_body: dict[str, Any]
    enabled: bool

    @property
    def key(self) -> str:
        return f"{self.model.key}@{self.channel.id}"

    @property
    def usable(self) -> bool:
        return self.enabled and self.channel.configured

    @property
    def metered(self) -> bool:
        return self.channel.kind == "metered"

    @property
    def budget_scope(self) -> str:
        b = self.channel.budget
        if b is None or b.scope == "shared":
            return self.channel.id
        return self.key

    def list_usd(self, miss: float, cached: float, out: float, now: datetime | None = None) -> float:
        return self.price.per_million(miss, cached, out) * self.pricing_time.at(now)

    def real_usd(self, miss: float, cached: float, out: float, now: datetime | None = None) -> float:
        return self.list_usd(miss, cached, out, now) if self.metered else 0.0

    def units(self, miss: float, cached: float, out: float, now: datetime | None = None) -> float:
        """订阅额度消耗（GLM：积分；OpenCode Go：占月额度的百分比）。"""
        b = self.channel.budget
        if b is None:
            return 0.0
        if b.unit == "points":
            p = self.points or Price(0, 0, 0)
            raw = (miss * p.input + cached * p.cached + out * p.output) / 10_000
            return raw * b.time.at(now)
        if not self.monthly_usd:
            return 0.0
        return self.list_usd(miss, cached, out, now) / self.monthly_usd * 100.0


# ───────────────────────────── 策略 ─────────────────────────────


@dataclass
class TaskPolicy:
    name: str
    need: list[float]
    usd_per_point: list[float]
    usd_per_sec: float
    est_output: list[int]
    first_token_timeout_s: float
    deadline_s: float = 0.0  # 预计总耗时超过它的候选先排除；0 = 不设死线

    def at(self, values: list[Any], difficulty: int) -> Any:
        return values[max(1, min(difficulty, len(values))) - 1]


@dataclass
class KeyDefaults:
    daily_credits: float = 500.0
    rpm: int = 60
    max_concurrency: int = 4
    max_output_tokens: int = 0
    plan: str = "standard"
    exhaust_mode: str = "economy"
    soft_ratio: float = 0.8
    grace_ratio: float = 0.15
    economy_max_credits: float = 3.0
    allowed_regions: list[str] | None = None  # 新 key 的默认区域白名单；None = 不限


@dataclass
class Policy:
    credit_usd: float
    timezone: str
    keys: KeyDefaults
    tasks: dict[str, TaskPolicy]
    daily_metered_usd_cap: float = 0.0
    max_request_bytes: int = 32 * 1024 * 1024
    auth_fail_per_min: int = 20
    long_context_tokens: int = 120_000
    min_context_tokens: int = 0  # 只对接原生上下文 ≥ 这个值的模型；0 = 不限
    # 这些任务不套用上面那条门槛：输入本来就小，要求 1M 只是白白少掉又快又便宜的候选。
    # 真正的正确性由 router 里那条按本次请求算的 room 检查保证（装不下就是装不下）。
    min_context_exempt_tasks: frozenset[str] = frozenset()
    unknown_model: str = "reject"  # 客户端点名未知模型：reject = 404；route = 退回自动路由
    prefill_tps: float = 25_000.0  # 预填充速度估计，用于把首 token 超时按输入长度放大
    stream_idle_timeout_s: float = 120.0
    max_attempts: int = 3
    session_ttl_s: float = 1800.0
    pacing_floor: float = 0.05
    pacing_knee: float = 0.5
    pacing_full: float = 0.85
    pacing_safety: float = 0.97
    pacing_single_share: float = 0.15  # 单次请求占订阅窗口的比例超过这个数就不再给地板价
    pacing_adaptive: bool = True  # 对偶镜像下降自适应价格水平
    pacing_eta: float = 0.05  # 学习率
    pacing_lam_min: float = 0.2
    pacing_lam_max: float = 5.0
    breaker_threshold: int = 3
    breaker_base_cooldown_s: float = 30.0
    breaker_max_cooldown_s: float = 600.0
    learning_enabled: bool = True
    regenerate_window_s: float = 180.0
    correction_window_s: float = 900.0
    learn_penalty: float = 0.15
    learn_max_down: float = 1.5
    learn_max_up: float = 0.3
    hedge_enabled: bool = True
    hedge_tasks: frozenset[str] = frozenset({"chat", "utility"})
    hedge_min_delay_s: float = 0.8
    hedge_ttft_factor: float = 2.0
    hedge_max_credits: float = 2.0
    verify_retry: bool = True
    verify_retry_flags: frozenset[str] = frozenset({"tool_args_invalid", "tool_schema_invalid", "json_invalid", "empty"})
    verify_retry_max_credits: float = 8.0
    verify_buffer_bytes: int = 64 * 1024


@dataclass
class Settings:
    channels: dict[str, Channel]
    models: dict[str, ModelSpec]
    offers: list[Offer]
    policy: Policy
    admin_token: str
    db_path: str
    # TOTP 二次验证（RFC 6238）。secret 为空 = 功能关闭；totp_only = 口令彻底失效。
    # 两者都从环境变量读，改完要重启进程（与熔断器设置同理）。
    totp_secret: str = ""
    totp_only: bool = False
    warnings: list[str] = field(default_factory=list)
    config_dir: str = ""
    managed: set[str] = field(default_factory=set)  # 来自 channels.d/ 的渠道 id，控制台可改可删

    def offer(self, key: str) -> Offer | None:
        return next((o for o in self.offers if o.key == key), None)

    def usable_offers(self) -> list[Offer]:
        return [o for o in self.offers if o.usable]

    def eligible_offers(self) -> list[Offer]:
        """路由真正会考虑的集合：可用，且原生上下文达到 min_context_tokens。"""
        need = self.policy.min_context_tokens
        return [o for o in self.usable_offers() if o.model.context >= need]

    def regions(self) -> set[str]:
        """配置里出现过的所有渠道区域，用于校验 key 的 allowed_regions。"""
        return {c.region for c in self.channels.values()}


# ───────────────────────────── 加载 ─────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _models(raw: dict[str, Any]) -> dict[str, ModelSpec]:
    out: dict[str, ModelSpec] = {}
    for key, m in (raw or {}).items():
        q = {k: float(v) for k, v in (m.get("quality") or {}).items()}
        bad = set(q) - set(TASKS)
        if bad:
            raise ConfigError(f"model {key}: unknown quality tasks {sorted(bad)}")
        out[key] = ModelSpec(
            key=key,
            display=m.get("display", key),
            context=int(m.get("context", 128_000)),
            max_output=int(m.get("max_output", 8192)),
            vision=bool(m.get("vision", False)),
            tools=bool(m.get("tools", True)),
            json_mode=bool(m.get("json_mode", True)),
            reasoning=m.get("reasoning", "none"),
            tier=m.get("tier", "standard"),
            ttft_s=float(m.get("ttft_s", 1.5)),
            tps=float(m.get("tps", 50)),
            quality=q,
            needs_reasoning_echo=bool(
                m.get("needs_reasoning_echo", m.get("reasoning", "none") == "always")
            ),
        )
    return out


def _channels(raw: dict[str, Any], models: dict[str, ModelSpec]) -> tuple[dict[str, Channel], list[Offer]]:
    channels: dict[str, Channel] = {}
    offers: list[Offer] = []
    for cid, c in (raw or {}).items():
        kind = c.get("kind", "metered")
        if kind not in ("metered", "subscription"):
            raise ConfigError(f"channel {cid}: kind must be metered|subscription")
        dialect = c.get("dialect", "openai_chat")
        if dialect not in DIALECTS:
            raise ConfigError(f"channel {cid}: unknown dialect {dialect}")
        budget = None
        if kind == "subscription":
            b = c.get("budget") or {}
            if b.get("unit") not in ("points", "percent"):
                raise ConfigError(f"channel {cid}: budget.unit must be points|percent")
            budget = Budget(
                unit=b["unit"],
                scope=b.get("scope", "shared"),
                windows=[BudgetWindow(float(w["hours"]), float(w["limit"])) for w in b.get("windows", [])],
                time=_time_factor(b.get("time")),
                exhausted_cooldown_s=float(b.get("exhausted_cooldown_s", 1800)),
            )
            if not budget.windows:
                raise ConfigError(f"channel {cid}: subscription needs budget.windows")
        ch = Channel(
            id=cid,
            kind=kind,
            base_url=str(c["base_url"]).rstrip("/"),
            api_key_env=c["api_key_env"],
            dialect=dialect,
            region=c.get("region", "global"),
            max_concurrency=int(c.get("max_concurrency", 64)),
            user_agent=c.get("user_agent", "thalamus-gateway/3.0"),
            headers={str(k): str(v) for k, v in (c.get("headers") or {}).items()},
            pricing_time=_time_factor(c.get("pricing_time")),
            budget=budget,
            enabled=bool(c.get("enabled", True)),
            connect_timeout_s=float(c.get("connect_timeout_s", 8)),
        )
        channels[cid] = ch
        for mkey, o in (c.get("offers") or {}).items():
            model = models.get(mkey)
            if model is None:
                raise ConfigError(f"channel {cid}: offer references unknown model {mkey}")
            odialect = o.get("dialect", dialect)
            if odialect not in DIALECTS:
                raise ConfigError(f"offer {mkey}@{cid}: unknown dialect {odialect}")
            price = _price(o.get("price") or o.get("list_price"))
            if price is None:
                raise ConfigError(f"offer {mkey}@{cid}: price/list_price required")
            points = _price(o.get("points"))
            if budget and budget.unit == "points" and points is None:
                raise ConfigError(f"offer {mkey}@{cid}: points coefficients required")
            monthly = o.get("monthly_usd")
            if budget and budget.unit == "percent" and not monthly:
                raise ConfigError(f"offer {mkey}@{cid}: monthly_usd required")
            offers.append(
                Offer(
                    channel=ch,
                    model=model,
                    upstream=o.get("upstream", mkey),
                    dialect=odialect,
                    path=o.get("path", DIALECTS[odialect]),
                    price=price,
                    pricing_time=_time_factor(o["pricing_time"]) if "pricing_time" in o else ch.pricing_time,
                    points=points,
                    monthly_usd=float(monthly) if monthly else None,
                    think_tags=bool(o.get("think_tags", False)),
                    min_max_tokens=int(o.get("min_max_tokens", 0)),
                    extra_body=dict(o.get("extra_body") or {}),
                    enabled=bool(o.get("enabled", True)),
                )
            )
    return channels, offers


def _policy(raw: dict[str, Any]) -> Policy:
    tasks: dict[str, TaskPolicy] = {}
    for name in TASKS:
        t = (raw.get("tasks") or {}).get(name)
        if t is None:
            raise ConfigError(f"policy.tasks.{name} missing")
        tp = TaskPolicy(
            name=name,
            need=[float(x) for x in t["need"]],
            usd_per_point=[float(x) for x in t["usd_per_point"]],
            usd_per_sec=float(t.get("usd_per_sec", 0.0)),
            est_output=[int(x) for x in t["est_output"]],
            first_token_timeout_s=float(t.get("first_token_timeout_s", 20)),
            deadline_s=float(t.get("deadline_s", 0)),
        )
        for f_ in ("need", "usd_per_point", "est_output"):
            if len(getattr(tp, f_)) != 3:
                raise ConfigError(f"policy.tasks.{name}.{f_} must have 3 values (difficulty 1/2/3)")
        tasks[name] = tp

    k = raw.get("keys") or {}
    keys = KeyDefaults(**{f_: k[f_] for f_ in KeyDefaults.__dataclass_fields__ if f_ in k})
    if keys.allowed_regions is not None:
        if not isinstance(keys.allowed_regions, list) or not all(isinstance(r, str) for r in keys.allowed_regions):
            raise ConfigError("policy.keys.allowed_regions 必须是字符串列表，或留空表示不限")
        keys.allowed_regions = sorted({r.strip() for r in keys.allowed_regions if r.strip()}) or None
    g = raw.get("global") or {}
    p = raw.get("subscription_pacing") or {}
    br = raw.get("breaker") or {}
    le = raw.get("learning") or {}
    hd = raw.get("hedge") or {}
    hedge_tasks = frozenset(hd.get("tasks", ["chat", "utility"]))
    if hedge_tasks - set(TASKS):
        raise ConfigError(f"hedge.tasks 含未知任务 {sorted(hedge_tasks - set(TASKS))}")

    from .outcome import ALL_FLAGS

    vr = raw.get("verify_retry") or {}
    retry_flags = frozenset(vr.get("flags", ["tool_args_invalid", "tool_schema_invalid", "json_invalid", "empty"]))
    if retry_flags - set(ALL_FLAGS):
        raise ConfigError(f"verify_retry.flags 含未知标记 {sorted(retry_flags - set(ALL_FLAGS))}")
    return Policy(
        credit_usd=float(raw.get("credit_usd", 0.001)),
        timezone=raw.get("timezone", "UTC"),
        keys=keys,
        tasks=tasks,
        daily_metered_usd_cap=float(g.get("daily_metered_usd_cap", 0)),
        max_request_bytes=int(g.get("max_request_mb", 32) * 1024 * 1024),
        auth_fail_per_min=int(g.get("auth_fail_per_min", 20)),
        long_context_tokens=int(g.get("long_context_tokens", 120_000)),
        min_context_tokens=int(g.get("min_context_tokens", 0)),
        min_context_exempt_tasks=frozenset(g.get("min_context_exempt_tasks") or ()),
        unknown_model=_one_of(g.get("unknown_model", "reject"), ("reject", "route"), "global.unknown_model"),
        prefill_tps=max(1.0, float(g.get("prefill_tps", 25_000))),
        stream_idle_timeout_s=float(g.get("stream_idle_timeout_s", 120)),
        max_attempts=int(g.get("max_attempts", 3)),
        session_ttl_s=float(g.get("session_ttl_s", 1800)),
        pacing_floor=float(p.get("floor", 0.05)),
        pacing_knee=float(p.get("knee", 0.5)),
        pacing_full=float(p.get("full", 0.85)),
        pacing_safety=float(p.get("safety", 0.97)),
        pacing_single_share=float(p.get("single_share", 0.15)),
        pacing_adaptive=bool(p.get("adaptive", True)),
        pacing_eta=float(p.get("eta", 0.05)),
        pacing_lam_min=float(p.get("lam_min", 0.2)),
        pacing_lam_max=float(p.get("lam_max", 5.0)),
        breaker_threshold=int(br.get("threshold", 3)),
        breaker_base_cooldown_s=float(br.get("base_cooldown_s", 30)),
        breaker_max_cooldown_s=float(br.get("max_cooldown_s", 600)),
        learning_enabled=bool(le.get("enabled", True)),
        regenerate_window_s=float(le.get("regenerate_window_s", 180)),
        correction_window_s=float(le.get("correction_window_s", 900)),
        learn_penalty=float(le.get("penalty", 0.15)),
        learn_max_down=float(le.get("max_down", 1.5)),
        learn_max_up=float(le.get("max_up", 0.3)),
        hedge_enabled=bool(hd.get("enabled", True)),
        hedge_tasks=hedge_tasks,
        hedge_min_delay_s=float(hd.get("min_delay_ms", 800)) / 1000,
        hedge_ttft_factor=float(hd.get("ttft_factor", 2.0)),
        hedge_max_credits=float(hd.get("max_extra_credits", 2.0)),
        verify_retry=bool(vr.get("enabled", True)),
        verify_retry_flags=retry_flags,
        verify_retry_max_credits=float(vr.get("max_credits", 8.0)),
        verify_buffer_bytes=int(float(vr.get("buffer_kb", 64)) * 1024),
    )


def validate(settings: Settings) -> list[str]:
    """启动时检查"某些请求根本无路可走"的配置错误，返回警告列表。"""
    warns: list[str] = []
    usable = settings.usable_offers()
    if not usable:
        warns.append("没有任何可用渠道：请在 .env 中配置至少一个上游 API key")
        return warns

    # 原生上下文门槛：把"我只对接 1M 模型"变成启动时可验证的约束，而不是配置时的自觉
    need_ctx = settings.policy.min_context_tokens
    if need_ctx:
        short = sorted(f"{o.key}({fmt_ctx(o.model.context)})" for o in usable if o.model.context < need_ctx)
        usable = [o for o in usable if o.model.context >= need_ctx]
        if short:
            exempt = sorted(settings.policy.min_context_exempt_tasks)
            extra = f"；但 {'/'.join(exempt)} 任务不套用这道门槛，它们仍可用这些 offer" if exempt else ""
            warns.append(
                f"只接 {fmt_ctx(need_ctx)} 上下文：排除了 {len(short)} 个 offer —— {', '.join(short)}{extra}")
        if not usable:
            warns.append(f"没有任何模型的原生上下文达到 {fmt_ctx(need_ctx)}，所有请求都会被拒绝")
            return warns
        dead = sorted({c.id for c in settings.channels.values() if c.configured} - {o.channel.id for o in usable})
        if dead:
            warns.append(f"渠道 {dead} 在 {fmt_ctx(need_ctx)} 门槛下没有任何可用 offer，等于没接")
        # 豁免的任务用的是过滤前的集合，所以要看那一份才知道某个方言到底会不会被走到
        reachable = settings.usable_offers() if settings.policy.min_context_exempt_tasks else usable
        if not any(o.dialect == "anthropic" for o in reachable):
            warns.append("门槛过滤后没有 anthropic 方言的 offer，该适配器在线上不会被走到")

    # 区域白名单：默认值写错或该区域没渠道，属于"所有请求都会失败"，必须在启动时就看见
    default_regions = settings.policy.keys.allowed_regions
    if default_regions:
        unknown = sorted(set(default_regions) - settings.regions())
        if unknown:
            warns.append(f"keys.allowed_regions 含未知区域 {unknown}；配置里的区域只有 {sorted(settings.regions())}")
        usable = [o for o in usable if o.channel.region in default_regions]
        if not usable:
            warns.append(f"keys.allowed_regions={default_regions} 下没有任何可用渠道，新建 key 的所有请求都会失败")
            return warns
        warns.append(f"默认区域白名单生效：新建 key 只能用 {default_regions} 的渠道（{len(usable)} 个 offer）")

    for task in TASKS:
        std = [o for o in usable if o.model.tier == "standard"]
        if task == "vision":
            std = [o for o in std if o.model.vision]
        if not std:
            warns.append(f"任务 {task} 没有 standard 档可用模型，standard key 的此类请求会失败")
    if not any(o.metered for o in usable):
        warns.append("没有按量渠道：订阅额度用完或被封时将无兜底")
    if not any(o.model.tools for o in usable):
        warns.append("没有支持工具调用的模型，agent 请求会失败")
    econ = settings.policy.keys.economy_max_credits
    cheap = [o for o in usable if o.model.tier == "standard"]
    if econ > 0 and cheap:
        cheapest = min(o.list_usd(2000, 0, 300) for o in cheap) / settings.policy.credit_usd
        if cheapest > econ:
            warns.append(
                f"经济档上限 {econ} credits 低于最便宜模型一次普通请求的约 {cheapest:.2f} credits，超额 key 将被拒绝"
            )
    return warns


MANAGED_DIR = "channels.d"


def load_settings(config_dir: str | os.PathLike[str] | None = None) -> Settings:
    cdir = Path(config_dir or os.getenv("THALAMUS_CONFIG_DIR", ROOT / "config"))
    up = _load_yaml(cdir / "upstreams.yaml")
    pol = _load_yaml(cdir / "policy.yaml")
    raw_models: dict[str, Any] = dict(up.get("models") or {})
    raw_channels: dict[str, Any] = dict(up.get("channels") or {})

    # 控制台添加的渠道各自一个文件，避免改写 upstreams.yaml（会冲掉里面的注释）
    managed: set[str] = set()
    mdir = cdir / MANAGED_DIR
    if mdir.is_dir():
        for p in sorted(mdir.glob("*.yaml")):
            doc = _load_yaml(p)
            for mkey, m in (doc.get("models") or {}).items():
                raw_models.setdefault(mkey, m)  # 不覆盖主文件里已有的模型定义
            for cid, c in (doc.get("channels") or {}).items():
                if cid in raw_channels:
                    raise ConfigError(f"{p.name}: 渠道 {cid} 与 upstreams.yaml 重名")
                raw_channels[cid] = c
                managed.add(cid)

    models = _models(raw_models)
    channels, offers = _channels(raw_channels, models)
    settings = Settings(
        channels=channels,
        models=models,
        offers=offers,
        policy=_policy(pol),
        admin_token=os.getenv("THALAMUS_ADMIN_TOKEN", ""),
        totp_secret=os.getenv("THALAMUS_TOTP_SECRET", "").strip(),
        totp_only=_truthy(os.getenv("THALAMUS_TOTP_ONLY", "")),
        db_path=os.getenv("THALAMUS_DB", str(ROOT / "data" / "thalamus.db")),
        config_dir=str(cdir),
        managed=managed,
    )
    settings.warnings = validate(settings)
    for w in settings.warnings:
        log.warning("config: %s", w)
    return settings

"""FastAPI 应用工厂。"""
from __future__ import annotations

import asyncio
import glob
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .adapters import ConnectionPool
from .api_admin import router as admin_router
from .api_v1 import router as v1_router
from .config import Settings, load_settings, set_secret_resolver
from .engine import Engine
from .secrets import SecretStore
from .store import Store
from .types import GatewayError

log = logging.getLogger("thalamus")


async def _janitor(app: FastAPI) -> None:
    tick = 0
    while True:
        await asyncio.sleep(60)
        tick += 1
        engine: Engine = app.state.engine
        try:
            engine.sessions.sweep()
            engine.reap_calls()
            await engine.flush()
            if tick % 10 == 0:
                engine.ledger.sweep()
                await engine.store.sweep_reservations()
                horizon = engine.capacity.max_seconds()
                if horizon:
                    await engine.store.prune_channel_usage(time.time() - horizon - 3600)
                await engine.store.prune_requests(30 * 86400)
            if tick % 60 == 1:
                await _daily_backup(engine.store)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("janitor error")


async def _keep_warm(app: FastAPI, interval: float) -> None:
    """定时轻量请求各上游，让长连接不因空闲过期（连接池 keepalive_expiry=120s）。

    压测实测：冷连接时首个请求首 token 多出约 550ms（本机回环），跨洲 TCP+TLS 只会更多。
    """
    while True:
        await asyncio.sleep(interval)
        try:
            s: Settings = app.state.settings
            app.state.pool_status = await app.state.pool.warm([c for c in s.channels.values() if c.configured])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("keep-warm error")


async def _daily_backup(store: Store, keep: int = 7) -> None:
    if store.path == ":memory:":
        return
    folder = os.path.join(os.path.dirname(os.path.abspath(store.path)), "backups")
    dest = os.path.join(folder, f"thalamus-{datetime.now():%Y%m%d}.db")
    if os.path.exists(dest):
        return
    await store.backup(dest)
    for old in sorted(glob.glob(os.path.join(folder, "thalamus-*.db")))[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass
    log.info("backup written: %s", dest)


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    warm: bool | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    warm = (os.getenv("THALAMUS_WARM", "1") == "1") if warm is None else warm

    async def startup(app: FastAPI) -> None:
        store = Store(settings.db_path)
        await store.connect()
        # 从这里开始任何一步失败都必须先把数据库连接关掉再抛出去：aiosqlite 的连接跑在
        # 非守护线程（daemon=False）上，不关的话解释器在 threading._shutdown 里 join
        # 它直到永远 —— 一个本该"报错退出"的配置问题会表现成 systemctl 卡住 90 秒
        # 然后被 SIGKILL，真正的 traceback 根本看不到。
        try:
            # 运行时密钥与数据库同目录；注册后 Channel.api_key 会优先用它，其次环境变量
            secrets = SecretStore(
                None if settings.db_path == ":memory:"
                else os.path.join(os.path.dirname(os.path.abspath(settings.db_path)), "secrets.json")
            )
            set_secret_resolver(secrets.get)
            app.state.secrets = secrets
            pool = ConnectionPool(transport)
            engine = Engine(settings, store, pool)
            await engine.start()
            app.state.settings = settings
            app.state.store = store
            app.state.pool = pool
            app.state.engine = engine
            app.state.admin_sessions = {}
            app.state.login_failures = {}
            app.state.pool_status = {}
            configured = [c.id for c in settings.channels.values() if c.configured]
            log.info("thalamus %s ready, channels: %s", __version__, configured or "NONE")
            app.state.keep_warm = None
            if warm and configured:
                app.state.pool_status = await pool.warm([c for c in settings.channels.values() if c.configured])
                interval = float(os.getenv("THALAMUS_KEEPWARM_S", "45"))
                if interval > 0:
                    app.state.keep_warm = asyncio.create_task(_keep_warm(app, interval))
            app.state.janitor = asyncio.create_task(_janitor(app))
        except BaseException:
            await store.close()
            raise

    async def shutdown(app: FastAPI) -> None:
        for name in ("janitor", "keep_warm"):
            task = getattr(app.state, name, None)
            if task:
                task.cancel()
        # 每一步都要走到：前一步抛异常不能连带让数据库连接关不掉（同 startup 的理由）。
        # 刷盘失败只是丢最近 250ms 的日志与用量，卡住退出是要被 SIGKILL 的。
        engine = getattr(app.state, "engine", None)
        pool = getattr(app.state, "pool", None)
        store_ = getattr(app.state, "store", None)
        for name, step in (("engine", engine and engine.stop), ("pool", pool and pool.aclose),
                           ("store", store_ and store_.close)):
            if not step:
                continue
            try:
                await step()
            except Exception:
                log.exception("shutdown: %s 关闭失败，继续关后面的", name)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await startup(app)
        try:
            yield
        finally:
            await shutdown(app)

    app = FastAPI(title="Thalamus", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.startup = lambda: startup(app)
    app.state.shutdown = lambda: shutdown(app)

    @app.exception_handler(GatewayError)
    async def _gateway_error(_, exc: GatewayError):
        return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)

    @app.get("/health")
    async def health():
        s: Settings = app.state.settings
        return {
            "status": "ok",
            "version": __version__,
            "channels_configured": sum(1 for c in s.channels.values() if c.configured),
            "uptime_s": int(time.time() - app.state.engine.started_at),
        }

    @app.get("/metrics")
    async def metrics(request: Request):
        """Prometheus 抓取端点。Bearer THALAMUS_METRICS_TOKEN（未设置则用管理口令）。"""
        token = os.getenv("THALAMUS_METRICS_TOKEN") or app.state.settings.admin_token
        auth = request.headers.get("authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not token or not given or not hmac.compare_digest(given, token):
            raise GatewayError(401, "unauthorized", "metrics token required")
        engine: Engine = app.state.engine
        return PlainTextResponse(engine.metrics.render(engine), media_type="text/plain; version=0.0.4")

    app.include_router(v1_router, prefix="/v1")
    app.include_router(admin_router, prefix="/admin")
    return app

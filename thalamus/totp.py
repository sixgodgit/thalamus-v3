"""TOTP（RFC 6238）——用标准库实现，给管理台做免口令登录 / 二次验证。

Google Authenticator、Aegis、1Password、Bitwarden 等都认这套参数：
HMAC-SHA1、30 秒步长、6 位数字、base32 密钥。这里不引入第三方依赖，
因为登录是唯一的入口，多一个依赖就多一个供应链面。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote, urlencode

DIGITS = 6
PERIOD = 30
DEFAULT_WINDOW = 1  # 允许前后各一个步长的时钟漂移


def new_secret(nbytes: int = 20) -> str:
    """生成 base32 密钥（无填充），可直接手工录入 Authenticator。"""
    return base64.b32encode(secrets.token_bytes(nbytes)).decode().rstrip("=")


def normalize(secret: str) -> str:
    """去掉空格/连字符/填充并转大写——手工录入常带这些。"""
    return "".join(secret.split()).replace("-", "").upper().rstrip("=")


def _key(secret: str) -> bytes:
    s = normalize(secret)
    if not s:
        raise ValueError("empty TOTP secret")
    pad = "=" * ((8 - len(s) % 8) % 8)
    try:
        return base64.b32decode(s + pad, casefold=True)
    except Exception:  # noqa: BLE001
        raise ValueError("TOTP secret is not valid base32") from None


def _hotp(key: bytes, counter: int, digits: int) -> str:
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    val = struct.unpack(">I", mac[off : off + 4])[0] & 0x7FFFFFFF
    return str(val % (10**digits)).zfill(digits)


def code(secret: str, at: float | None = None, digits: int = DIGITS, period: int = PERIOD) -> str:
    """某一时刻的验证码（测试与命令行用）。"""
    counter = int((time.time() if at is None else at) // period)
    return _hotp(_key(secret), counter, digits)


def counter_for(secret: str, given: str, at: float | None = None, window: int = DEFAULT_WINDOW,
                digits: int = DIGITS, period: int = PERIOD, min_counter: int = 0) -> int | None:
    """校验验证码。通过时返回命中的计数器，否则 None。

    返回计数器而不是布尔值，是为了让调用方做防重放：同一个验证码在它的
    时间窗内只应被接受一次，否则被截获的 6 位数字还能再用几十秒。
    """
    g = "".join(str(given).split())
    # 必须 ASCII 数字：全角数字的 isdigit() 也为真，但会让 compare_digest 抛 TypeError
    if not (g.isascii() and g.isdigit()) or len(g) != digits:
        return None
    base = int((time.time() if at is None else at) // period)
    key = _key(secret)
    for c in range(base - window, base + window + 1):
        if c <= min_counter:
            continue
        if hmac.compare_digest(_hotp(key, c, digits), g):
            return c
    return None


def seconds_left(at: float | None = None, period: int = PERIOD) -> int:
    """当前验证码还剩几秒——登录页可以据此提示/自动刷新。"""
    t = time.time() if at is None else at
    return int(period - (t % period))


def otpauth_uri(secret: str, label: str, issuer: str = "Thalamus") -> str:
    """给 Authenticator 扫码/识别用的标准 URI。"""
    q = urlencode({
        "secret": normalize(secret),
        "issuer": issuer,
        "algorithm": "SHA1",
        "digits": DIGITS,
        "period": PERIOD,
    })
    return f"otpauth://totp/{quote(issuer)}:{quote(label)}?{q}"

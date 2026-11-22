"""管理台 TOTP 登录：RFC 6238 向量、时钟漂移、防重放，以及登录路径本身。"""
import asyncio
import time

import pytest
from helpers import ADMIN, gateway, load_test_settings

from thalamus import totp

# RFC 6238 附录 B 的 SHA1 向量（密钥为 ASCII "12345678901234567890"）
RFC_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
RFC_VECTORS = [
    (59, "94287082"),
    (1111111109, "07081804"),
    (1111111111, "14050471"),
    (1234567890, "89005924"),
    (2000000000, "69279037"),
    (20000000000, "65353130"),
]


def run(coro):
    asyncio.run(coro)


@pytest.mark.parametrize("at,expect", RFC_VECTORS)
def test_matches_rfc6238_vectors(at, expect):
    assert totp.code(RFC_SECRET, at=at, digits=8) == expect


def test_secret_is_scannable_and_tolerant_of_typing():
    s = totp.new_secret()
    assert len(s) == 32 and "=" not in s
    assert totp.normalize(s.lower()) == s                    # 大小写
    assert totp.normalize(" ".join(s[i:i + 4] for i in range(0, 32, 4))) == s  # 每 4 位一组
    assert totp.normalize(s + "==") == s                     # 多打的填充
    assert totp.counter_for(s, totp.code(s)) is not None


def test_drift_window_is_one_step_each_way():
    s = totp.new_secret()
    now = time.time()
    assert totp.counter_for(s, totp.code(s, at=now - 30), at=now) is not None
    assert totp.counter_for(s, totp.code(s, at=now + 30), at=now) is not None
    assert totp.counter_for(s, totp.code(s, at=now - 60), at=now) is None
    assert totp.counter_for(s, totp.code(s, at=now + 60), at=now) is None


def test_wrong_and_malformed_codes_are_rejected():
    s = totp.new_secret()
    good = totp.code(s)
    assert totp.counter_for(s, "0" * 6 if good != "0" * 6 else "1" * 6) is None
    for junk in ("", "12345", "1234567", "abcdef", "12 34 56x", "１２３４５６"):
        assert totp.counter_for(s, junk) is None


def test_same_code_cannot_be_used_twice():
    s = totp.new_secret()
    now = time.time()
    c = totp.code(s, at=now)
    hit = totp.counter_for(s, c, at=now)
    assert hit is not None
    assert totp.counter_for(s, c, at=now, min_counter=hit) is None       # 同一窗口重放
    assert totp.counter_for(s, c, at=now, min_counter=hit - 1) == hit    # 只比它旧才拦


def test_bad_secret_is_a_clear_error():
    with pytest.raises(ValueError):
        totp.counter_for("not!!base32!!", "123456")


def test_otpauth_uri_shape():
    s = totp.new_secret()
    u = totp.otpauth_uri(s, "admin", issuer="Thalamus")
    assert u.startswith("otpauth://totp/Thalamus:admin?")
    assert f"secret={s}" in u and "digits=6" in u and "period=30" in u and "algorithm=SHA1" in u


# ───────────────────────── 管理台登录路径 ─────────────────────────


def test_login_with_totp_alone_and_token_closed(tmp_path):
    async def go():
        s = load_test_settings(tmp_path)
        s.totp_secret, s.totp_only = totp.new_secret(), True
        async with gateway(tmp_path, settings=s) as (_app, client, _):
            assert (await client.post("/admin/api/login", json={"totp": "000000"})).status_code == 401
            # only 模式下口令彻底失效，包括 Bearer 头
            assert (await client.post("/admin/api/login", json={"token": "admin-test-token"})).status_code == 401
            assert (await client.get("/admin/api/me", headers=ADMIN)).status_code == 401

            r = await client.post("/admin/api/login", json={"totp": totp.code(s.totp_secret)})
            assert r.status_code == 200
            sid = r.headers["set-cookie"].split(";")[0].split("=", 1)[1]
            C = {"Cookie": f"thl_admin={sid}"}
            me = (await client.get("/admin/api/me", headers=C)).json()
            assert me["totp"] == {"enabled": True, "only": True, "period": 30}
            assert (await client.get("/admin/api/overview", headers=C)).status_code == 200
    run(go())


def test_totp_replay_is_refused_by_the_gateway(tmp_path):
    async def go():
        s = load_test_settings(tmp_path)
        s.totp_secret = totp.new_secret()
        async with gateway(tmp_path, settings=s) as (_app, client, _):
            code = totp.code(s.totp_secret)
            assert (await client.post("/admin/api/login", json={"totp": code})).status_code == 200
            assert (await client.post("/admin/api/login", json={"totp": code})).status_code == 401
    run(go())


def test_token_still_works_when_totp_is_optional(tmp_path):
    async def go():
        s = load_test_settings(tmp_path)
        s.totp_secret, s.totp_only = totp.new_secret(), False
        async with gateway(tmp_path, settings=s) as (_app, client, _):
            assert (await client.post("/admin/api/login", json={"token": "admin-test-token"})).status_code == 200
            assert (await client.get("/admin/api/me", headers=ADMIN)).status_code == 200
            # 空验证码回落到口令比较，不会误放行
            assert (await client.post("/admin/api/login", json={"totp": ""})).status_code == 401
    run(go())


def test_login_page_renders_the_configured_mode(tmp_path):
    async def go():
        s = load_test_settings(tmp_path)
        s.totp_secret, s.totp_only = totp.new_secret(), True
        async with gateway(tmp_path, settings=s) as (_app, client, _):
            html = (await client.get("/admin")).text
            assert "const TOTP_ENABLED = true, TOTP_ONLY = true;" in html
            assert "__TOTP_ENABLED__" not in html and "__TOTP_ONLY__" not in html

        async with gateway(tmp_path) as (_app, client, _):
            html = (await client.get("/admin")).text
            assert "const TOTP_ENABLED = false, TOTP_ONLY = false;" in html
    run(go())

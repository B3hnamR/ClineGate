"""Device-code login flow: authorize -> poll -> register -> snapshot+upsert."""

from __future__ import annotations

import json

import httpx
import pytest

from cline_gateway.device_auth import (
    AUTHENTICATE, CLIENT_ID, DEVICE_AUTHORIZE, REGISTER, DeviceLogin,
)

WORKOS_APPROVED = {
    "user": {"id": "user_X", "email": "new@x.com", "external_id": "usr-NEW1"},
    "access_token": "eyJNEW.HEADER.SIG",
    "refresh_token": "REF25",
    "authentication_method": "GoogleOAuth",
}

REGISTER_OK = {
    "data": {
        "accessToken": "eyJCLINE.HEADER.SIG",
        "refreshToken": "CLINEREF",
        "tokenType": "Bearer",
        "expiresAt": "2026-09-22T02:52:30Z",
        "userInfo": {"clineUserId": "usr-NEW1", "email": "new@x.com"},
    },
    "success": True,
}

DEVICE = {
    "device_code": "D" * 64,
    "user_code": "ABCD-EFGH",
    "verification_uri": "https://authkit.cline.bot/device",
    "verification_uri_complete": "https://authkit.cline.bot/device?user_code=ABCD-EFGH",
    "expires_in": 300,
    "interval": 5,
}


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _router(*, approve_after=1, device_error=None, register_error=None):
    """Scripted WorkOS + Cline backend. approve_after=N polls before 200."""
    calls = {"auth": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == DEVICE_AUTHORIZE:
            if device_error:
                return httpx.Response(device_error, json={"error": "boom"})
            assert request.content.decode() == f"client_id={CLIENT_ID}"
            return httpx.Response(200, json=DEVICE)
        if url == AUTHENTICATE:
            body = request.content.decode()
            assert "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code" in body
            assert f"device_code={'D' * 64}" in body
            calls["auth"] += 1
            if calls["auth"] >= approve_after:
                return httpx.Response(200, json=WORKOS_APPROVED)
            return httpx.Response(400, json={"error": "authorization_pending"})
        if url == REGISTER:
            payload = json.loads(request.content.decode())
            assert payload == {"accessToken": "eyJNEW.HEADER.SIG",
                               "refreshToken": "REF25"}
            if register_error:
                return httpx.Response(register_error, json={"error": "no"})
            return httpx.Response(200, json=REGISTER_OK)
        raise AssertionError(f"unexpected call: {url}")

    return handler, calls


async def test_happy_path(tmp_path):
    handler, _ = _router(approve_after=2)
    seen = []

    async def collect(acc):
        seen.append(acc)

    login = DeviceLogin(tmp_path, client=_client(handler), on_account=collect)
    await login.run()

    assert login.state == "done"
    assert login.user_code == "ABCD-EFGH"
    assert login.verification_url.endswith("user_code=ABCD-EFGH")
    assert login.account_email == "new@x.com"

    assert len(seen) == 1
    acc = seen[0]
    assert acc.id == "usr-NEW1"
    assert acc.access_token == "workos:eyJCLINE.HEADER.SIG"   # stored form
    assert acc.refresh_token == "CLINEREF"
    assert acc.expires_at > 0
    assert acc.source == "device-login"

    # snapshot written to the accounts dir and parseable
    assert login.snapshot_path is not None and login.snapshot_path.is_file()
    from cline_gateway.accounts_dir import parse_snapshot
    fields = parse_snapshot(login.snapshot_path)
    assert fields["email"] == "new@x.com"
    assert fields["access_token"] == "workos:eyJCLINE.HEADER.SIG"


async def test_pending_then_approval_is_normal(tmp_path):
    handler, calls = _router(approve_after=3)
    login = DeviceLogin(tmp_path, client=_client(handler))
    await login.run()
    assert login.state == "done"
    assert calls["auth"] == 3       # two pendings, then the 200


async def test_denied_surfaces_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == DEVICE_AUTHORIZE:
            return httpx.Response(200, json=DEVICE)
        if url == AUTHENTICATE:
            return httpx.Response(400, json={"error": "access_denied"})
        raise AssertionError(url)

    login = DeviceLogin(tmp_path, client=_client(handler))
    await login.run()
    assert login.state == "error"
    assert "denied" in login.error


async def test_expired_token_state(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == DEVICE_AUTHORIZE:
            return httpx.Response(200, json=DEVICE)
        if url == AUTHENTICATE:
            return httpx.Response(400, json={"error": "expired_token"})
        raise AssertionError(url)

    login = DeviceLogin(tmp_path, client=_client(handler))
    await login.run()
    assert login.state == "expired"


async def test_register_failure(tmp_path):
    handler, _ = _router(approve_after=1, register_error=500)
    login = DeviceLogin(tmp_path, client=_client(handler))
    await login.run()
    assert login.state == "error"
    assert "register" in login.error


async def test_cancel_during_pending(tmp_path):
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == DEVICE_AUTHORIZE:
            return httpx.Response(200, json=DEVICE)
        if url == AUTHENTICATE:
            return httpx.Response(400, json={"error": "authorization_pending"})
        raise AssertionError(url)

    login = DeviceLogin(tmp_path, client=_client(handler))
    task = asyncio.create_task(login.run())
    await asyncio.sleep(0.2)          # let it start polling
    login.cancel()
    await asyncio.wait_for(task, timeout=3)
    assert login.state == "cancelled"


def test_routes_registered_and_gated():
    from fastapi.testclient import TestClient

    from cline_gateway.app import create_app
    from cline_gateway.config import load_config

    cfg = load_config()
    cfg.update.enabled = False
    app = create_app(cfg)
    with TestClient(app) as client:
        for path in ("/admin/dash/auth/login/start",
                     "/admin/dash/auth/login/cancel"):
            assert client.post(path).status_code in (401, 403)
        assert client.get("/admin/dash/auth/login/status").status_code in (401, 403)
        ok = {"Authorization": f"Bearer {cfg.server.admin_key}"}
        idle = client.get("/admin/dash/auth/login/status", headers=ok)
        assert idle.status_code == 200 and idle.json()["state"] == "idle"

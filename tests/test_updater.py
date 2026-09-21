"""Auto-updater: version compare, asset pick, check() shaping, route auth."""

from __future__ import annotations

import httpx
import pytest

from cline_gateway.updater import (
    UpdateChecker, is_newer, parse_version, pick_asset,
)


def test_parse_version_common_forms():
    assert parse_version("v0.3.0") == (0, 3, 0)
    assert parse_version("0.10.2") == (0, 10, 2)
    assert parse_version("1.3.0.0") == (1, 3, 0, 0)
    assert parse_version("") == ()


def test_is_newer():
    assert is_newer("v0.3.1", "0.3.0")
    assert is_newer("1.0.0", "0.9.9")
    assert not is_newer("v0.3.0", "0.3.0")     # equal is not newer
    assert not is_newer("v0.2.9", "0.3.0")
    assert not is_newer("garbage", "0.3.0")    # unparseable tag
    assert not is_newer("v0.3.0", "")          # unparseable current


def test_pick_asset():
    assets = [
        {"name": "checksums.txt", "browser_download_url": "https://github.com/x/s"},
        {"name": "ClineGateway.exe",
         "browser_download_url": "https://github.com/x/ClineGateway.exe",
         "size": 32_000_000},
    ]
    assert pick_asset(assets)["name"] == "ClineGateway.exe"
    assert pick_asset(assets[:1]) is None
    assert pick_asset([]) is None


class _StubClient:
    """Stands in for httpx.AsyncClient; returns a canned GitHub response."""

    def __init__(self, payload, status=200):
        self._payload = payload
        self._status = status

    async def get(self, url):
        # raise_for_status() needs a bound request on the response
        return httpx.Response(self._status, json=self._payload,
                              request=httpx.Request("GET", url))

    async def aclose(self):
        return None


def _release(tag="v0.4.0", with_asset=True):
    return {
        "tag_name": tag,
        "html_url": "https://github.com/B3hnamR/ClineGate/releases/tag/" + tag,
        "body": "what changed",
        "published_at": "2026-09-22T00:00:00Z",
        "assets": ([{"name": "ClineGateway.exe",
                     "browser_download_url": "https://github.com/a/ClineGateway.exe",
                     "size": 32_000_000}] if with_asset else []),
    }


async def test_check_reports_newer_release():
    checker = UpdateChecker("B3hnamR/ClineGate", "0.3.0",
                            client=_StubClient(_release()))
    status = await checker.check()
    assert status["update_available"] is True
    assert status["latest"] == "0.4.0"
    assert status["current"] == "0.3.0"
    assert status["download_url"].startswith("https://github.com/")
    assert status["notes"] == "what changed"


async def test_check_no_update_when_current():
    checker = UpdateChecker("B3hnamR/ClineGate", "0.4.0",
                            client=_StubClient(_release()))
    assert (await checker.check())["update_available"] is False


async def test_check_no_update_without_exe_asset():
    checker = UpdateChecker("B3hnamR/ClineGate", "0.3.0",
                            client=_StubClient(_release(with_asset=False)))
    assert (await checker.check())["update_available"] is False


async def test_check_404_first_release_is_not_an_error():
    checker = UpdateChecker("B3hnamR/ClineGate", "0.3.0",
                            client=_StubClient({}, status=404))
    status = await checker.check()
    assert status["update_available"] is False
    assert "error" not in status


async def test_check_failure_is_cached_not_raised():
    class _Boom:
        async def get(self, url):
            raise httpx.ConnectError("nope")

        async def aclose(self):
            return None

    checker = UpdateChecker("B3hnamR/ClineGate", "0.3.0", client=_Boom())
    status = await checker.check()
    assert status["update_available"] is False
    assert status["error"] == "ConnectError"


async def test_disabled_checker_never_calls_network():
    class _Explode:
        async def get(self, url):
            raise AssertionError("must not be called")

        async def aclose(self):
            return None

    checker = UpdateChecker("B3hnamR/ClineGate", "0.3.0", enabled=False,
                            client=_Explode())
    status = await checker.check()
    assert status["update_available"] is False
    assert status["disabled"] is True


def test_routes_registered_and_auth_gated():
    from fastapi.testclient import TestClient

    from cline_gateway.app import create_app
    from cline_gateway.config import load_config

    cfg = load_config()
    cfg.update.enabled = False
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/admin/dash/update/status").status_code in (401, 403)
        assert client.post("/admin/dash/update/check").status_code in (401, 403)
        assert client.post("/admin/dash/update/download").status_code in (401, 403)
        assert client.post("/admin/dash/update/apply").status_code in (401, 403)
        ok = {"Authorization": f"Bearer {cfg.server.admin_key}"}
        status = client.get("/admin/dash/update/status", headers=ok)
        assert status.status_code == 200
        assert status.json()["update_available"] is False
        # downloads/applies are impossible without an available update
        assert client.post("/admin/dash/update/apply", headers=ok).status_code \
            == 400

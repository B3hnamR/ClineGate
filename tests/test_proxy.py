"""upstream.proxy must reach every httpx client that talks to Cline."""

from __future__ import annotations

import asyncio

import httpx
import yaml

from cline_gateway.config import Config, UpstreamConfig, load_config
from cline_gateway.model_catalog import ModelCatalog
from cline_gateway.registry import Registry
from cline_gateway.tokens import TokenManager
from cline_gateway.upstream import UpstreamClient

SOCKS = "socks5h://127.0.0.1:1080"
HTTP = "http://127.0.0.1:10809"


def _proxy_url(client: httpx.AsyncClient) -> str:
    """The proxy a client mounts, or "" when it goes direct."""
    for transport in client._mounts.values():
        url = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
        if url is not None:
            return f"{url.scheme.decode()}://{url.host.decode()}:{url.port}"
    return ""


def test_upstream_client_mounts_config_proxy():
    up = UpstreamClient(UpstreamConfig(proxy=SOCKS))
    assert _proxy_url(up._client) == SOCKS


def test_upstream_client_goes_direct_by_default(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    up = UpstreamClient(UpstreamConfig())
    assert _proxy_url(up._client) == ""


def test_token_refresh_client_mounts_proxy():
    cfg = Config(upstream=UpstreamConfig(proxy=HTTP))
    mgr = TokenManager(cfg, pool=None)  # type: ignore[arg-type]
    assert _proxy_url(mgr._client) == HTTP
    asyncio.run(mgr._client.aclose())


def test_model_catalog_mounts_proxy():
    cat = ModelCatalog("https://api.cline.bot/api/v1", Registry(), proxy=SOCKS)
    assert _proxy_url(cat._client) == SOCKS


def test_config_yaml_and_env_override(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"upstream:\n  proxy: {SOCKS}\n", encoding="utf-8")
    assert load_config(cfg_file).upstream.proxy == SOCKS

    monkeypatch.setenv("CLINE_GATEWAY_UPSTREAM__PROXY", HTTP)
    assert load_config(cfg_file).upstream.proxy == HTTP


# --------------------------------------------------------------------------- #
# settings-save validation: reject a bad proxy before it reaches config.yaml
# (a value httpx cannot construct on would stop the gateway booting at all)
# --------------------------------------------------------------------------- #

def _put_proxy(tmp_path, monkeypatch, value: str):
    """PUT upstream.proxy through the dashboard settings endpoint."""
    from fastapi.testclient import TestClient

    from cline_gateway.app import create_app

    monkeypatch.chdir(tmp_path)
    conf = tmp_path / "config.yaml"
    conf.write_text("server:\n  port: 9997\n", encoding="utf-8")
    cfg = load_config(conf)
    # hermetic: no real accounts/store/logs, no background pollers
    cfg.update.enabled = False
    cfg.pool.balance_poll_seconds = 0
    cfg.accounts.source = "pool_file"
    cfg.accounts.pool_file = str(tmp_path / "empty.json")
    (tmp_path / "empty.json").write_text('{"accounts": []}', encoding="utf-8")
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")

    app = create_app(cfg)
    with TestClient(app) as client:
        resp = client.put("/admin/dash/settings",
                          headers={"Authorization":
                                   f"Bearer {cfg.server.admin_key}"},
                          json={"values": {"upstream.proxy": value}})
    return resp, conf


def test_settings_reject_unsupported_proxy(tmp_path, monkeypatch):
    resp, conf = _put_proxy(tmp_path, monkeypatch, "ftp://127.0.0.1:1080")
    assert resp.status_code == 400
    assert "scheme" in resp.json()["detail"]
    written = yaml.safe_load(conf.read_text(encoding="utf-8"))
    assert "proxy" not in (written.get("upstream") or {})   # rejected pre-write


def test_settings_reject_proxy_without_scheme(tmp_path, monkeypatch):
    resp, _ = _put_proxy(tmp_path, monkeypatch, "127.0.0.1:1080")
    assert resp.status_code == 400
    assert "scheme" in resp.json()["detail"]


def test_settings_accepts_proxy_schemes_and_empty(tmp_path, monkeypatch):
    for value in ("socks5h://127.0.0.1:1080", "http://127.0.0.1:10809", ""):
        resp, conf = _put_proxy(tmp_path, monkeypatch, value)
        assert resp.status_code == 200, resp.text
        assert resp.json()["changed"] == ["upstream.proxy"]
        written = yaml.safe_load(conf.read_text(encoding="utf-8"))
        assert written["upstream"]["proxy"] == value

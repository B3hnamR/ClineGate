"""The visible model list follows Cline's curated feed, not old chat captures."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from cline_gateway.app import create_app
from cline_gateway.config import ClientKey, Config
from cline_gateway.model_catalog import ModelCatalog
from cline_gateway.registry import Registry


def _feed(*, free: str = "vendor/current-free") -> dict:
    return {
        "recommended": [{"id": "vendor/current-paid", "name": "Current Paid"}],
        "free": [{"id": free, "name": "Current Free"}],
        "clinePass": [{"id": "cline-pass/current"}],
        "clineCloud": [{"id": "cline-cloud/current"}],
    }


def test_live_feed_replaces_snapshot_and_free_lane():
    reg = Registry()
    assert "cline-free/kimi-k3" not in {m["id"] for m in reg.catalogue()}
    reg.use_live_catalogue(_feed())
    assert {m["id"] for m in reg.catalogue()} == {
        "vendor/current-paid", "vendor/current-free",
        "cline-pass/current", "cline-cloud/current"}
    assert reg.lane_for("vendor/current-free") == "free"
    assert reg.lane_for("vendor/current-paid") == "usage"
    reg.use_live_catalogue(_feed(free="vendor/replacement-free"))
    assert reg.lane_for("vendor/current-free") == "usage"
    assert reg.lane_for("vendor/replacement-free") == "free"


def test_bad_feed_preserves_last_good_catalogue():
    reg = Registry()
    reg.use_live_catalogue(_feed())
    with pytest.raises(ValueError):
        reg.use_live_catalogue({"free": []})
    assert "vendor/current-free" in {m["id"] for m in reg.catalogue()}


@pytest.mark.asyncio
async def test_catalog_refresh_and_failure_keep_last_good_data():
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/api/v1/ai/cline/recommended-models"
        if calls == 1:
            return httpx.Response(200, json=_feed())
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        reg = Registry()
        catalog = ModelCatalog("https://api.cline.bot/api/v1", reg, client=client)
        assert (await catalog.refresh_if_due())["source"] == "live"
        await catalog.refresh_if_due()
        assert calls == 1  # cached for five minutes
        status = await catalog.refresh_if_due(force=True)
        assert status["source"] == "live"
        assert status["error"] == "HTTPStatusError"
        assert "vendor/current-free" in {m["id"] for m in reg.catalogue()}


def test_models_endpoints_share_the_live_feed(tmp_path):
    cfg = Config()
    cfg.server.admin_key = "adm-test"
    cfg.server.client_keys = [ClientKey(key="ck-test")]
    cfg.accounts.source = "pool_file"
    cfg.accounts.pool_file = str(tmp_path / "empty.json")
    (tmp_path / "empty.json").write_text('{"accounts": []}', encoding="utf-8")
    cfg.store.sqlite_path = str(tmp_path / "gateway.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")
    cfg.update.enabled = False
    cfg.pool.balance_poll_seconds = 0

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_feed())

    with TestClient(create_app(cfg)) as api:
        catalog = api.app.state.app_state.catalog
        catalog._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        catalog._owns_client = True
        matrix = api.get("/admin/models/availability",
                         headers={"Authorization": "Bearer adm-test"})
        models = api.get("/v1/models", headers={"Authorization": "Bearer ck-test"})
        assert matrix.status_code == 200
        assert models.status_code == 200
        assert matrix.json()["catalog"]["source"] == "live"
        expected = {"vendor/current-paid", "vendor/current-free",
                    "cline-pass/current", "cline-cloud/current"}
        assert {m["model"] for m in matrix.json()["models"]} == expected
        assert {m["id"] for m in models.json()["data"]} == expected
        assert "cline-free/kimi-k3" not in expected
        sync = api.post("/admin/models/catalog/refresh",
                        headers={"Authorization": "Bearer adm-test"})
        assert sync.status_code == 200
        assert sync.json()["source"] == "live"

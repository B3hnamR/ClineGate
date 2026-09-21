"""Streaming routes must report open-time failures with a real HTTP status.

Regression for a live symptom: a streaming request whose upstream could not be
opened (no credit / no account) reached the client as
"connection reset by server", because the failure was raised inside the
StreamingResponse generator - after 200 + headers had already been sent.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from cline_gateway.app import create_app
from cline_gateway.config import Config


def _app_with_no_accounts(tmp_path):
    cfg = Config()
    cfg.server.require_client_key = False
    cfg.accounts.source = "pool_file"
    cfg.accounts.pool_file = str(tmp_path / "empty.json")
    (tmp_path / "empty.json").write_text('{"accounts": []}', encoding="utf-8")
    cfg.store.sqlite_path = str(tmp_path / "g.db")
    cfg.logging.capture_dir = str(tmp_path / "logs")
    return create_app(cfg)


def test_streaming_open_failure_returns_an_error_status(tmp_path):
    app = _app_with_no_accounts(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "cline-free/solar-pro4",
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
        )

    # the important part: a status we can read, not a dropped connection
    assert r.status_code >= 400, f"expected an error status, got {r.status_code}"
    body = r.json()
    assert "error" in body
    assert "account" in json.dumps(body).lower()


def test_anthropic_streaming_open_failure_returns_an_error_status(tmp_path):
    app = _app_with_no_accounts(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/v1/messages",
            json={"model": "cline-free/solar-pro4", "max_tokens": 32,
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
        )

    assert r.status_code >= 400
    body = r.json()
    assert body.get("type") == "error"

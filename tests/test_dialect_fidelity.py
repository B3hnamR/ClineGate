"""Body-variant and header regression tests against the Phase-1 capture.

These are the guard rails: if the gateway stops speaking the captured dialect,
these fail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cline_gateway.config import Fingerprint
from cline_gateway.registry import CAPTURED_MODEL_VARIANTS, Registry
from cline_gateway.upstream import (
    VARIANTS,
    REQUIRED_HEADER_NAMES,
    build_headers,
    build_upstream_body,
)

CAPTURE_DIR = Path(os.environ.get(
    "CLINE_CAPTURE_DIR",
    Path(__file__).resolve().parents[2] / "capture" / "captures",
))


def load_captured_calls() -> list[dict]:
    """Every real-client chat call from the Phase-1 capture."""
    records = []
    if not CAPTURE_DIR.is_dir():
        return records
    for path in sorted(CAPTURE_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            req = rec.get("request", {})
            if "chat/completions" not in req.get("path", ""):
                continue
            if "x-client-version" not in req.get("headers", {}):
                continue
            records.append(rec)
    return records


CAPTURED = load_captured_calls()
pytestmark = pytest.mark.skipif(not CAPTURED, reason="no capture fixtures present")


# --------------------------------------------------------------------------- #


def test_header_names_match_capture():
    captured_names = {
        k.lower() for k in CAPTURED[0]["request"]["headers"]
        if k.lower() not in ("host", "content-length")
    }
    assert captured_names == REQUIRED_HEADER_NAMES


def test_build_headers_emits_exact_name_set():
    headers = build_headers("workos:test", Fingerprint())
    assert {k.lower() for k in headers} == REQUIRED_HEADER_NAMES
    assert headers["authorization"] == "Bearer workos:test"


def test_header_values_match_capture():
    captured = {k.lower(): v for k, v in CAPTURED[0]["request"]["headers"].items()}
    built = build_headers("Bearer workos:test", Fingerprint())
    for key in ("user-agent", "x-client-version", "x-core-version", "x-platform",
                "x-platform-version", "x-client-type", "x-is-multiroot", "x-title",
                "http-referer", "content-type"):
        assert built[key] == captured[key], f"{key} drifted from the capture"


@pytest.mark.parametrize("record", CAPTURED, ids=lambda r: json.loads(
    r["request"]["body"])["model"] + "#" + str(abs(hash(r["request"]["body"])) % 1000))
def test_body_variant_matches_capture(record):
    """Rebuild the upstream body from the captured body and compare key sets."""
    import copy

    captured_body = json.loads(record["request"]["body"])
    model = captured_body["model"]
    variant = Registry().variant_for(model)

    payload = copy.deepcopy(captured_body)
    rebuilt = build_upstream_body(payload, variant)

    captured_keys = set(captured_body.keys())
    rebuilt_keys = set(rebuilt.keys())
    assert rebuilt_keys == captured_keys, (
        f"variant {variant} produced {sorted(rebuilt_keys)} "
        f"but capture has {sorted(captured_keys)} for {model}"
    )


def test_all_captured_models_have_a_variant():
    for model in CAPTURED_MODEL_VARIANTS:
        assert Registry().variant_for(model) in VARIANTS


def test_variant_rules_for_unknown_models():
    reg = Registry()
    assert reg.variant_for("anthropic/claude-future-9") == "anthropic"
    assert reg.variant_for("some-new-vendor/model") == "default"


def test_anthropic_variant_adds_cache_control():
    body = build_upstream_body({"model": "x", "messages": []}, "anthropic")
    assert body["cache_control"] == {"type": "ephemeral"}
    assert "max_tokens" in body


def test_nextgen_variant_swaps_max_tokens_field():
    body = build_upstream_body({"model": "x", "messages": [], "max_tokens": 4096},
                               "openai-nextgen")
    assert "max_completion_tokens" in body
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 4096


def test_reasoning_effort_passthrough():
    body = build_upstream_body(
        {"model": "x", "messages": [], "reasoning_effort": "xhigh"}, "default")
    assert body["reasoning_effort"] == "xhigh"

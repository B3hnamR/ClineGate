"""Reasoning controls from third-party clients (Kilo Code, Cline, others).

Two halves:

* the gateway accepts every spelling a client may send (snake_case, the
  camelCase that Kilo's generated variant bodies use, and the
  enable_thinking / thinking shapes its settings UI can emit)
* the client-sync module adds the `reasoning: true` flag that makes Kilo
  render its thinking picker at all (it never reads capabilities from
  /v1/models)
"""

from __future__ import annotations

import json
import re

import pytest

from cline_gateway.client_sync import (
    _provider_blocks,
    inspect_kilo,
    sync_kilo,
)
from cline_gateway.upstream import build_upstream_body


# --------------------------------------------------------------------------- #
# gateway: accepting client spellings
# --------------------------------------------------------------------------- #


def _k3(**extra):
    payload = {"model": "cline-free/kimi-k3",
               "messages": [{"role": "user", "content": "hi"}]}
    payload.update(extra)
    return build_upstream_body(payload, "reasoning")


def test_snake_case_effort_still_works():
    assert _k3(reasoning_effort="high")["reasoning_effort"] == "high"


def test_camel_case_effort_from_kilo_variants():
    # Kilo's generated variant bodies use { reasoningEffort: effort }
    assert _k3(reasoningEffort="xhigh")["reasoning_effort"] == "xhigh"


def test_kilo_fallback_ladder_values_are_normalised():
    # Kilo's default ladder is none/low/medium/high/xhigh/max
    assert _k3(reasoningEffort="none")["reasoning"] == {"enabled": False}
    assert _k3(reasoningEffort="max")["reasoning_effort"] == "xhigh"
    assert _k3(reasoningEffort="minimal")["reasoning_effort"] == "medium"
    assert _k3(reasoningEffort="low")["reasoning_effort"] == "medium"
    assert _k3(reasoningEffort="medium")["reasoning_effort"] == "medium"


def test_output_effort_key_is_accepted():
    assert _k3(effort="high")["reasoning_effort"] == "high"


def test_thinking_object_shapes():
    assert _k3(thinking={"type": "disabled"})["reasoning"] == {"enabled": False}
    assert _k3(thinking={"type": "enabled"})["reasoning_effort"] == "medium"
    assert _k3(thinking={"type": "adaptive"})["reasoning_effort"] == "medium"


def test_enable_thinking_flag():
    assert _k3(enable_thinking=False)["reasoning"] == {"enabled": False}
    assert _k3(enable_thinking=True)["reasoning_effort"] == "medium"


def test_nothing_requested_stays_off():
    body = _k3()
    assert body["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in body


def test_explicit_reasoning_object_is_preserved():
    body = _k3(reasoning={"enabled": True, "budget": 4096})
    assert body["reasoning"] == {"enabled": True, "budget": 4096}


def test_other_variants_accept_camel_case_too():
    payload = {"model": "zai/glm-5.3-flash",
               "messages": [{"role": "user", "content": "hi"}],
               "reasoningEffort": "high"}
    body = build_upstream_body(payload, "default")
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 32000


def test_other_variants_keep_the_low_default():
    body = build_upstream_body(
        {"model": "x", "messages": []}, "default")
    assert body["reasoning_effort"] == "low"


# --------------------------------------------------------------------------- #
# client sync
# --------------------------------------------------------------------------- #

KILO_CONFIG = """\
{
  // kilo config
  "$schema": "https://app.kilo.ai/config.json",
  "provider": {
    "clinegate": {
      "name": "Cline Gateway",
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": {
        "cline-free/kimi-k3": {
          "name": "Kimi K3",
        },
        "anthropic/claude-opus-5": {
          "name": "Opus",
        },
        "zai/glm-5.3-flash": {
          "name": "GLM flash",
        },
      },
    },
    "other": {
      "name": "other",
      "options": { "baseURL": "https://api.other.com/v1" },
      "models": {
        "some-model": { "name": "Some" },
      },
    },
  },
}
"""


@pytest.fixture()
def kilo_cfg(tmp_path):
    p = tmp_path / "kilo.jsonc"
    p.write_text(KILO_CONFIG, encoding="utf-8")
    return p


def _parse(path):
    text = re.sub(r"^\s*//.*$", "", path.read_text(encoding="utf-8"), flags=re.M)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def test_provider_blocks_are_detected():
    names = [n for n, _s, _e in _provider_blocks(KILO_CONFIG)]
    assert names == ["clinegate", "other"]


def test_inspect_finds_gateway_models_needing_the_flag(kilo_cfg):
    st = inspect_kilo(kilo_cfg)
    assert st.found is True
    assert st.gateway_providers == ["clinegate"]
    assert st.models_total == 3
    assert st.models_flagged == 0
    # glm flash is a known non-reasoning model: never flagged
    assert sorted(st.models_missing_flag) == [
        "anthropic/claude-opus-5", "cline-free/kimi-k3"]
    assert st.needs_sync is True


def test_sync_adds_flag_and_ladder(kilo_cfg):
    result = sync_kilo(kilo_cfg)
    assert result["error"] == ""
    assert result["changed"] == 2
    assert result["backup"] is not None

    data = _parse(kilo_cfg)
    models = data["provider"]["clinegate"]["models"]

    # kimi-k3 gets the explicit ladder
    assert models["cline-free/kimi-k3"]["reasoning"] is True
    ladder = models["cline-free/kimi-k3"]["variants"]
    assert set(ladder) == {"off", "medium", "high", "xhigh"}
    assert ladder["off"] == {"reasoning": {"enabled": False}}
    assert ladder["xhigh"] == {"reasoning_effort": "xhigh"}

    # a reasoning-capable model without a known ladder gets the flag only
    assert models["anthropic/claude-opus-5"]["reasoning"] is True
    assert "variants" not in models["anthropic/claude-opus-5"]

    # known non-reasoning model untouched
    assert models["zai/glm-5.3-flash"] == {"name": "GLM flash"}

    # other providers untouched
    assert data["provider"]["other"]["models"]["some-model"] == {"name": "Some"}
    # comments and other top-level keys survive
    assert data["$schema"] == "https://app.kilo.ai/config.json"


def test_sync_is_idempotent(kilo_cfg):
    sync_kilo(kilo_cfg)
    again = sync_kilo(kilo_cfg)
    assert again["changed"] == 0
    assert inspect_kilo(kilo_cfg).needs_sync is False


def test_sync_dry_run_writes_nothing(kilo_cfg):
    before = kilo_cfg.read_text(encoding="utf-8")
    result = sync_kilo(kilo_cfg, dry_run=True)
    assert result["changed"] == 0
    assert len(result["models"]) == 2
    assert kilo_cfg.read_text(encoding="utf-8") == before


def test_inspect_missing_file(tmp_path):
    st = inspect_kilo(tmp_path / "nope.jsonc")
    assert st.found is False
    assert st.needs_sync is False


def test_inspect_reports_broken_config(tmp_path):
    bad = tmp_path / "kilo.jsonc"
    bad.write_text("{ this is not json", encoding="utf-8")
    st = inspect_kilo(bad)
    assert st.found is True
    assert st.error != ""
    assert st.needs_sync is False


def test_sync_refuses_to_write_when_parse_would_break(kilo_cfg, monkeypatch):
    # simulate a broken insertion: force the model pattern to match inside a
    # string it should not touch, producing invalid json
    import cline_gateway.client_sync as cs

    def bad_insert(text, provider, models):
        return text + "\n{oops", 1

    monkeypatch.setattr(cs, "_insert_model_flags", bad_insert)
    before = kilo_cfg.read_text(encoding="utf-8")
    result = cs.sync_kilo(kilo_cfg)

    assert result["changed"] == 0
    assert "refusing to write" in result["error"]
    # original file is intact (the guard ran before any write)
    assert kilo_cfg.read_text(encoding="utf-8") == before

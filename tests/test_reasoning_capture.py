"""Reasoning vocabulary captured 2026-09-24 (cline-20260924-170429.jsonl) and
2026-09-26 (cline-20260926-132202.jsonl).

Captured per picker level on cline-free/gemini-3.8-flash:
    None   -> {"reasoning": {"enabled": false}}, no token key
    Low    -> reasoning_effort "low",   no token key
    Medium -> reasoning_effort "medium", no token key
    High   -> reasoning_effort "high",  no token key
    Extra  -> reasoning_effort "xhigh", no token key
and on stealth/space-bunny-alpha / cline-free/mimo-v2.6-flash /
cline-free/deepseek-v4.1-flash / cline-free/muse-spark-1.3-contributor:
    High   -> max_tokens + reasoning_effort "high"

stealth/pixel-canary (captured 2026-09-26, full ladder None/Low/Medium/High/Extra):
    None   -> {"reasoning": {"enabled": false}}, no token key
    Low    -> reasoning_effort "low",  no token key
    Medium -> reasoning_effort "medium", no token key
    High   -> reasoning_effort "high", no token key
    Extra  -> reasoning_effort "xhigh", no token key
Streams delta.reasoning + reasoning_details [{type: reasoning.text}].
"""

from __future__ import annotations

from cline_gateway.registry import Registry
from cline_gateway.upstream import build_upstream_body


def _gemini(**extra):
    base = {"model": "cline-free/gemini-3.8-flash", "messages": []}
    base.update(extra)
    return base


def _bunny(**extra):
    base = {"model": "stealth/space-bunny-alpha", "messages": []}
    base.update(extra)
    return base


def _pixel(**extra):
    base = {"model": "stealth/pixel-canary", "messages": []}
    base.update(extra)
    return base


# --------------------------------------------------------------------------- #
# registry: the captured variant classification
# --------------------------------------------------------------------------- #


def test_gemini_is_the_reasoning_family():
    reg = Registry()
    assert reg.variant_for("cline-free/gemini-3.8-flash") == "reasoning"
    assert reg.knows("cline-free/gemini-3.8-flash") is True
    # pass/cloud ids of the family resolve the same way
    assert reg.variant_for("cline-pass/gemini-3.8-flash") == "reasoning"


def test_other_new_models_stay_default():
    reg = Registry()
    assert reg.variant_for("stealth/space-bunny-alpha") == "default"
    assert reg.variant_for("cline-free/mimo-v2.6-flash") == "default"


def test_pixel_canary_is_the_reasoning_family():
    reg = Registry()
    assert reg.variant_for("stealth/pixel-canary") == "reasoning"
    assert reg.knows("stealth/pixel-canary") is True
    assert reg.lane_for("stealth/pixel-canary") == "free"


# --------------------------------------------------------------------------- #
# gemini: captured body shapes per picker level
# --------------------------------------------------------------------------- #


def test_gemini_picker_levels_match_the_capture():
    none = build_upstream_body(_gemini(reasoning_effort="none"), "reasoning")
    assert none["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in none

    for level, wire in (("low", "low"), ("medium", "medium"),
                        ("high", "high"), ("extra", "xhigh")):
        body = build_upstream_body(_gemini(reasoning_effort=level), "reasoning")
        assert body["reasoning_effort"] == wire, level
        assert "reasoning" not in body


def test_gemini_keeps_low_unlike_kimi():
    # kimi's floor is medium (never observed below); gemini's low is captured
    gemini = build_upstream_body(_gemini(reasoning_effort="low"), "reasoning")
    kimi = build_upstream_body(
        {"model": "cline-free/kimi-k3", "messages": [],
         "reasoning_effort": "low"}, "reasoning")
    assert gemini["reasoning_effort"] == "low"
    assert kimi["reasoning_effort"] == "medium"


def test_gemini_body_has_no_token_limit_key():
    body = build_upstream_body(_gemini(reasoning_effort="high"), "reasoning")
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


# --------------------------------------------------------------------------- #
# pixel-canary: full ladder captured 2026-09-26 (None/Low/Medium/High/Extra);
# same family shape as gemini-3.8-flash (no token key, effort-driven).
# --------------------------------------------------------------------------- #


def test_pixel_canary_captured_levels_match_the_capture():
    high = build_upstream_body(_pixel(reasoning_effort="high"), "reasoning")
    assert high["reasoning_effort"] == "high"
    assert "max_tokens" not in high
    assert "max_completion_tokens" not in high
    extra = build_upstream_body(_pixel(reasoning_effort="extra"), "reasoning")
    assert extra["reasoning_effort"] == "xhigh"     # captured as UI picker 'Extra'
    assert "max_tokens" not in extra

    none = build_upstream_body(_pixel(reasoning_effort="none"), "reasoning")
    assert none["reasoning"] == {"enabled": False}  # captured as UI picker 'None'
    assert "reasoning_effort" not in none

    for level in ("low", "medium"):
        body = build_upstream_body(_pixel(reasoning_effort=level), "reasoning")
        assert body["reasoning_effort"] == level    # captured verbatim
        assert "reasoning" not in body
        assert "max_tokens" not in body


# --------------------------------------------------------------------------- #
# default family: extra -> xhigh, None -> reasoning.enabled false
# --------------------------------------------------------------------------- #


def test_extra_maps_to_xhigh_on_the_default_family():
    body = build_upstream_body(_bunny(reasoning_effort="extra"), "default")
    assert body["reasoning_effort"] == "xhigh"        # was a 502 before
    assert body["max_tokens"] == 32000


def test_none_disables_thinking_via_the_reasoning_object():
    body = build_upstream_body(_bunny(reasoning_effort="none"), "default")
    assert body["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in body


def test_default_effort_is_still_low():
    body = build_upstream_body(_bunny(), "default")
    assert body["reasoning_effort"] == "low"


def test_client_reasoning_object_passes_through_on_default():
    body = build_upstream_body(
        _bunny(reasoning={"enabled": True, "effort": "high"}), "default")
    assert body["reasoning"] == {"enabled": True, "effort": "high"}
    assert "reasoning_effort" not in body

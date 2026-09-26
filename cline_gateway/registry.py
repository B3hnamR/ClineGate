"""Model registry: alias resolution and upstream body-variant mapping.

The variant map is derived directly from the Phase-1 capture
(`Cline/gateway-spec.json`). Four shapes have been observed:

  default         -> max_tokens
  anthropic       -> max_tokens + cache_control
  openai-nextgen  -> max_completion_tokens (no max_tokens)
  reasoning       -> NEITHER token key; drives thinking with
                     "reasoning_effort" (medium/high/xhigh) or
                     {"reasoning": {"enabled": false}}   (captured 2026-09-19,
                     cline-free/kimi-k3)
"""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any, Literal

Variant = Literal["default", "anthropic", "openai-nextgen", "reasoning"]

# --------------------------------------------------------------------------- #
# captured model -> variant
# --------------------------------------------------------------------------- #

CAPTURED_MODEL_VARIANTS: dict[str, Variant] = {
    # default shape (9 observed)
    "cline-free/muse-spark-1.3-contributor": "default",
    "cline-free/deepseek-v4.1-flash": "default",
    "cline-free/solar-pro4": "default",
    "zai/glm-5.3-flash": "default",
    "poolside/laguna-s-2.1:free": "default",
    "openai/gpt-6-astra": "default",
    "x-ai/grok-4.5": "default",
    "moonshotai/kimi-k3": "default",
    "~openai/gpt-sol-latest": "default",
    # anthropic shape (2 observed)
    "anthropic/claude-opus-5": "anthropic",
    "anthropic/claude-fable-5.1": "anthropic",
    # openai next-gen shape (1 observed)
    "openai/gpt-5.6-sol": "openai-nextgen",
    # reasoning shape (5 observed 2026-09-19: reasoning off/medium/high/xhigh)
    "cline-free/kimi-k3": "reasoning",
    # stealth/pixel-canary captured 2026-09-26: no token-limit key,
    # reasoning_effort-driven, streams delta.reasoning + reasoning_details
    "stealth/pixel-canary": "reasoning",
}

# Catalogued families from model-catalog.json (recommended-models endpoint).
# Not present in the Phase-1 chat capture, so their variant is inferred:
# every non-anthropic family observed so far uses the default shape.
KNOWN_MODEL_VARIANTS: dict[str, Variant] = {
    # free section, 2026-09-24
    # gemini-3.8-flash captured 2026-09-24: reasoning family (no token key)
    "cline-free/gemini-3.8-flash": "reasoning",
    "cline-free/mimo-v2.6-flash": "default",
    "stealth/space-bunny-alpha": "default",
    "cline-free/deepseek-v4.1-flash": "default",
    "cline-free/muse-spark-1.3-contributor": "default",
    # cline-pass
    "cline-pass/deepseek-v4-flash": "default",
    "cline-pass/qwen3.8-max": "default",
    "cline-pass/glm-5.2": "default",
    "cline-pass/deepseek-v4-pro": "default",
    "cline-pass/deepseek-v4.1-flash": "default",
    "cline-pass/glm-5.3-flash": "default",
    "cline-pass/kimi-k2.6": "default",
    "cline-pass/qwen3.7-max": "default",
    "cline-pass/minimax-m3": "default",
    "cline-pass/kimi-k2.7-code": "default",
    "cline-pass/glm-5.3": "default",
    "cline-pass/qwen3.7-plus": "default",
    "cline-pass/mimo-v2.5-pro": "default",
    "cline-pass/mimo-v2.5": "default",
    "cline-pass/mimo-v2.6-flash": "default",
    "cline-pass/mimo-v2.6-pro": "default",
    "cline-pass/muse-spark-1.3-contributor": "default",
    # same model family as the captured cline-free/kimi-k3: the request shape is
    # chosen by family, not by lane
    "cline-pass/kimi-k3": "reasoning",
    "cline-cloud/kimi-k3": "reasoning",
    "cline-cloud/deepseek-v4-flash": "default",
    "cline-cloud/glm-5.2": "default",
}

# The public recommended-models feed is the source for the displayed catalogue.
# This captured 2026-09-24 snapshot (free group updated 2026-09-26 with
# stealth/pixel-canary) is used until the first successful fetch and during an
# outage. Historical model ids above remain useful for request variant
# selection, but must not bring removed models back into the Models page.
FALLBACK_CATALOG_GROUPS: dict[str, tuple[str, ...]] = {
    "recommended": (
        "spacexai/grok-4.7", "openai/gpt-6-astra", "moonshotai/kimi-k3",
        "anthropic/claude-opus-5",
    ),
    "free": (
        "cline-free/gemini-3.8-flash", "stealth/space-bunny-alpha",
        "stealth/pixel-canary",
        "cline-free/mimo-v2.6-flash", "cline-free/deepseek-v4.1-flash",
        "cline-free/muse-spark-1.3-contributor",
    ),
    "clinePass": (
        "cline-pass/mimo-v2.6-flash", "cline-pass/mimo-v2.6-pro",
        "cline-pass/glm-5.3", "cline-pass/qwen3.8-max",
        "cline-pass/deepseek-v4-pro", "cline-pass/deepseek-v4.1-flash",
        "cline-pass/muse-spark-1.3-contributor", "cline-pass/kimi-k3",
        "cline-pass/glm-5.3-flash", "cline-pass/qwen3.7-plus",
        "cline-pass/minimax-m3", "cline-pass/qwen3.7-max",
        "cline-pass/mimo-v2.5-pro", "cline-pass/mimo-v2.5",
    ),
    "clineCloud": (
        "cline-cloud/glm-5.3", "cline-cloud/deepseek-v4.1-flash",
        "cline-cloud/kimi-k3",
    ),
}
CATALOG_GROUPS = tuple(FALLBACK_CATALOG_GROUPS)
CATALOG_FREE_MODELS: frozenset[str] = frozenset(FALLBACK_CATALOG_GROUPS["free"])


def _parse_catalogue(payload: dict[str, Any]) -> tuple[tuple[dict, ...], frozenset[str]]:
    """Validate the feed before replacing the entire visible catalogue."""
    if not isinstance(payload, dict) or any(
            not isinstance(payload.get(group), list) for group in CATALOG_GROUPS):
        raise ValueError("unexpected recommended-models response shape")
    entries: dict[str, dict] = {}
    free_ids: set[str] = set()
    for group in CATALOG_GROUPS:
        for item in payload[group]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("invalid recommended-models entry")
            model_id = item["id"].strip()
            if not model_id:
                raise ValueError("empty recommended-models id")
            if group == "free":
                free_ids.add(model_id)
            if model_id not in entries:
                entries[model_id] = {
                    "id": model_id,
                    "name": item.get("name") if isinstance(item.get("name"), str) else model_id,
                    "description": (item.get("description") if isinstance(
                        item.get("description"), str) else ""),
                    "tags": ([tag for tag in item["tags"] if isinstance(tag, str)]
                             if isinstance(item.get("tags"), list) else []),
                    "catalog_group": group,
                }
    if not entries:
        raise ValueError("recommended-models response is empty")
    return tuple(entries.values()), frozenset(free_ids)

# Pattern rules applied when a model is not in the tables above.
VARIANT_RULES: list[tuple[re.Pattern[str], Variant]] = [
    (re.compile(r"^anthropic/", re.I), "anthropic"),
    (re.compile(r"^claude", re.I), "anthropic"),
    (re.compile(r"^openai/gpt-5\.6-sol$", re.I), "openai-nextgen"),
    # kimi-k3 family: no token-limit key, reasoning_effort-driven
    # anchored: an unanchored substring match handed the reasoning variant to
    # any id merely containing "kimi-k3" (e.g. x/kimi-k3-old)
    (re.compile(r"(?:^|/)kimi-k3(?:$|[-.:])", re.I), "reasoning"),
    # gemini-3.8-flash: same family shape (no token key), captured 2026-09-24
    (re.compile(r"(?:^|/)gemini-3\.8-flash(?:$|[-.:])", re.I), "reasoning"),
]


class Registry:
    """Resolves client-supplied model names and their upstream body variant."""

    def __init__(self, aliases: dict[str, str] | None = None,
                 default: str = "cline-free/deepseek-v4.1-flash",
                 default_anthropic: str = "anthropic/claude-opus-5",
                 probe_unknown: bool = True) -> None:
        self.aliases = dict(aliases or {})
        self.default = default
        self.default_anthropic = default_anthropic
        self.probe_unknown = probe_unknown
        self._learned: dict[str, Variant] = {}
        snapshot = {
            group: [{"id": model_id} for model_id in ids]
            for group, ids in FALLBACK_CATALOG_GROUPS.items()
        }
        self._catalog_entries, self._catalog_free_ids = _parse_catalogue(snapshot)

    @property
    def free_models(self) -> frozenset[str]:
        return self._catalog_free_ids

    def use_live_catalogue(self, payload: dict[str, Any]) -> None:
        """Replace the visible models atomically after validating the full feed."""
        entries, free_ids = _parse_catalogue(payload)
        self._catalog_entries = entries
        self._catalog_free_ids = free_ids

    def lane_for(self, upstream_model: str) -> str:
        return model_lane(upstream_model, free_models=self.free_models)

    def is_free_model(self, upstream_model: str) -> bool:
        return self.lane_for(upstream_model) == "free"

    # ------------------------------------------------------------------ #

    def resolve(self, requested: str | None, dialect: str = "openai") -> str:
        """Map a client model name to an upstream Cline model id."""
        if not requested:
            return self.default_anthropic if dialect == "anthropic" else self.default

        if requested in self.aliases:
            return self.aliases[requested]

        # already a known/captured upstream id
        if requested in CAPTURED_MODEL_VARIANTS or requested in self._learned:
            return requested

        # bare claude-* names from anthropic SDK clients
        if dialect == "anthropic" and requested.lower().startswith("claude"):
            return self.default_anthropic

        return requested

    def knows(self, upstream_model: str) -> bool:
        """True when the variant for this model is already established."""
        return (upstream_model in CAPTURED_MODEL_VARIANTS
                or upstream_model in KNOWN_MODEL_VARIANTS
                or upstream_model in self._learned)

    def variant_for(self, upstream_model: str) -> Variant:
        if upstream_model in CAPTURED_MODEL_VARIANTS:
            return CAPTURED_MODEL_VARIANTS[upstream_model]
        if upstream_model in KNOWN_MODEL_VARIANTS:
            return KNOWN_MODEL_VARIANTS[upstream_model]
        if upstream_model in self._learned:
            return self._learned[upstream_model]
        for pattern, variant in VARIANT_RULES:
            if pattern.search(upstream_model):
                return variant
        return "default"

    def learn(self, upstream_model: str, variant: Variant) -> None:
        self._learned[upstream_model] = variant

    # ------------------------------------------------------------------ #

    @staticmethod
    def is_free(upstream_model: str) -> bool:
        """Offline credit-free classification; instances use the live free set."""
        m = upstream_model or ""
        return (m.lower().startswith("cline-free/")
                or m.lower().endswith(":free")
                or m in CATALOG_FREE_MODELS)

    # ------------------------------------------------------------------ #

    def catalogue(self) -> list[dict]:
        """Current curated model list, plus explicitly configured aliases."""
        out = []
        for item in self._catalog_entries:
            model_id = item["id"]
            captured = model_id in CAPTURED_MODEL_VARIANTS
            out.append({
                **item,
                "id": model_id,
                "object": "model",
                "owned_by": "cline",
                "cline_variant": self.variant_for(model_id),
                "captured": captured,
                "is_free": self.is_free_model(model_id),
            })
        # expose configured aliases as first-class ids too
        for alias, target in sorted(self.aliases.items()):
            if any(entry["id"] == alias for entry in out):
                continue
            out.append({
                "id": alias,
                "object": "model",
                "owned_by": "cline",
                "cline_variant": self.variant_for(target),
                "alias_of": target,
                "is_free": self.is_free_model(target),
            })
        return out

# --------------------------------------------------------------------------- #
# billing lanes
# --------------------------------------------------------------------------- #

# cline-pass/* and cline-cloud/* are gated by a SUBSCRIPTION, not by credits.
# Everything else that is not free is usage-billed against the credit balance.
PLAN_PREFIXES = ("cline-pass/", "cline-cloud/")


def model_lane(model: str, free_models: Collection[str] | None = None) -> str:
    """Which gate applies to this model?

      "free"  - credit-free, capped per model per account (daily window)
      "plan"  - needs a Cline Pass / Cloud subscription
      "usage" - billed against the account's credit balance
    """
    m = model or ""
    if (Registry.is_free(m) if free_models is None else
            (m.lower().startswith("cline-free/") or m.lower().endswith(":free")
             or m in free_models)):
        return "free"
    if any(m.lower().startswith(pref) for pref in PLAN_PREFIXES):
        return "plan"
    return "usage"

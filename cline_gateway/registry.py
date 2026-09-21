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
from typing import Literal

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
}

# Catalogued families from model-catalog.json (recommended-models endpoint).
# Not present in the Phase-1 chat capture, so their variant is inferred:
# every non-anthropic family observed so far uses the default shape.
KNOWN_MODEL_VARIANTS: dict[str, Variant] = {
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
    # same model family as the captured cline-free/kimi-k3: the request shape is
    # chosen by family, not by lane
    "cline-pass/kimi-k3": "reasoning",
    "cline-cloud/kimi-k3": "reasoning",
    "cline-cloud/deepseek-v4-flash": "default",
    "cline-cloud/glm-5.2": "default",
}

# --------------------------------------------------------------------------- #
# free-model classification
# --------------------------------------------------------------------------- #

# From model-catalog.json -> `free`. The catalogue spells one id "z-ai/..." while
# the client sends "zai/..." — both must classify as free, because a daily cap on
# a free model is not a credit exhaustion and must not retire the paid lane.
CATALOG_FREE_MODELS: frozenset[str] = frozenset({
    "cline-free/deepseek-v4.1-flash",
    "cline-free/muse-spark-1.3-contributor",
    "cline-free/solar-pro4",
    "cline-free/kimi-k3",
    "poolside/laguna-s-2.1:free",
    "z-ai/glm-5.3-flash",
    "zai/glm-5.3-flash",
})

# Pattern rules applied when a model is not in the tables above.
VARIANT_RULES: list[tuple[re.Pattern[str], Variant]] = [
    (re.compile(r"^anthropic/", re.I), "anthropic"),
    (re.compile(r"^claude", re.I), "anthropic"),
    (re.compile(r"^openai/gpt-5\.6-sol$", re.I), "openai-nextgen"),
    # kimi-k3 family: no token-limit key, reasoning_effort-driven
    # anchored: an unanchored substring match handed the reasoning variant to
    # any id merely containing "kimi-k3" (e.g. x/kimi-k3-old)
    (re.compile(r"(?:^|/)kimi-k3(?:$|[-.:])", re.I), "reasoning"),
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
        """Credit-free models.

        True for the `cline-free/*` prefix, any `:free` suffix, and every id in
        the catalogue's free section (which includes `zai/glm-5.3-flash` — a
        model with a *daily cap* rather than a credit cost).

        Why this matters: a 402 on the paid lane retires the account's paid lane,
        and a 429 daily-cap on a free model must not do anything of the sort.
        """
        m = upstream_model or ""
        return (m.lower().startswith("cline-free/")
                or m.lower().endswith(":free")
                or m in CATALOG_FREE_MODELS)

    # ------------------------------------------------------------------ #

    def catalogue(self) -> list[dict]:
        """Model list for GET /v1/models (and /v1/models for anthropic clients)."""
        all_variants: dict[str, Variant] = {}
        all_variants.update(KNOWN_MODEL_VARIANTS)
        all_variants.update(CAPTURED_MODEL_VARIANTS)
        all_variants.update(self._learned)
        out = []
        for model_id, variant in sorted(all_variants.items()):
            captured = model_id in CAPTURED_MODEL_VARIANTS
            out.append({
                "id": model_id,
                "object": "model",
                "owned_by": "cline",
                "cline_variant": variant,
                "captured": captured,
                "is_free": Registry.is_free(model_id),
            })
        # expose configured aliases as first-class ids too
        for alias, target in sorted(self.aliases.items()):
            out.append({
                "id": alias,
                "object": "model",
                "owned_by": "cline",
                "cline_variant": self.variant_for(target),
                "alias_of": target,
                "is_free": Registry.is_free(target),
            })
        return out

# --------------------------------------------------------------------------- #
# billing lanes
# --------------------------------------------------------------------------- #

# cline-pass/* and cline-cloud/* are gated by a SUBSCRIPTION, not by credits.
# Everything else that is not free is usage-billed against the credit balance.
PLAN_PREFIXES = ("cline-pass/", "cline-cloud/")


def model_lane(model: str) -> str:
    """Which gate applies to this model?

      "free"  - credit-free, capped per model per account (daily window)
      "plan"  - needs a Cline Pass / Cloud subscription
      "usage" - billed against the account's credit balance
    """
    m = model or ""
    if Registry.is_free(m):
        return "free"
    if any(m.lower().startswith(pref) for pref in PLAN_PREFIXES):
        return "plan"
    return "usage"

"""Service layer: one chat request -> account selection, upstream call, failover,
usage accounting, capture. Rendering (OpenAI vs Anthropic) happens at the edges;
everything here works on the internal OpenAI-shaped payload.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import httpx
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .config import Config
from .pool import Account, PoolManager
from .registry import Registry
from .store import JsonlCapture, Store
from .tokens import TokenManager
from .translate_anthropic import AnthropicStreamTranslator, extract_openai_usage
from .upstream import (
    ErrorKind,
    VARIANTS,
    edge_block_message,
    is_entitlement_code,
    sse_frame,
    UpstreamClient,
    UpstreamResponse,
    TransportError,
    build_headers,
    build_upstream_body,
    classify,
    iter_sse_events,
    parse_upstream_error,
)

log = logging.getLogger("cline_gateway.service")


def _sse_event(name: str, data: dict) -> bytes:
    """Anthropic-style named SSE event (event: + data:)."""
    return (f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            ).encode("utf-8")


def _split_sse_frames(pending: bytes) -> tuple[list[bytes], bytes]:
    """Split complete SSE frames off the buffer, keeping the partial tail.

    SSE allows both ``\\n\\n`` and ``\\r\\n\\r\\n`` as event separators; splitting
    on the literal ``\\n\\n`` alone left CRLF-framed streams buffered until the
    connection closed (no streaming, growing memory).
    """
    frames: list[bytes] = []
    while True:
        m = re.search(rb"\r\n\r\n|\n\n", pending)
        if not m:
            return frames, pending
        frames.append(pending[:m.start()])
        pending = pending[m.end():]

RETRY_PAUSE_SECONDS = 0.4
# A 403 entitlement error parks the model for this long (6h): a plan/subscription
# does not change minute to minute, and the account itself stays fully usable.
ENTITLEMENT_PARK_SECONDS = 6 * 3600


class NoAccountsAvailable(Exception):
    pass


class UpstreamFailure(Exception):
    def __init__(self, status: int, code: str, message: str,
                 release_at: str | None = None,
                 release_in_s: int | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        # model-scoped errors carry the absolute release time so clients can
        # schedule instead of guessing
        self.release_at = release_at
        self.release_in_s = release_in_s
        self.details = details or {}

    def extra(self) -> dict:
        out = {}
        if self.release_at:
            out["release_at"] = self.release_at
        if self.release_in_s is not None:
            out["release_in_s"] = self.release_in_s
        out.update(self.details)
        return out


@dataclass
class StreamHandle:
    """An opened upstream stream, before any client bytes are sent."""
    account: Account
    headers: dict
    body: dict
    resp: UpstreamResponse
    variant: str
    started: float


@dataclass
class CallResult:
    account: Account
    status: int
    raw: bytes
    duration_ms: float
    upstream_headers: dict


class ChatService:
    def __init__(self, cfg: Config, pool: PoolManager, tokens: TokenManager,
                 client: UpstreamClient, registry: Registry, store: Store,
                 capture: JsonlCapture) -> None:
        self.cfg = cfg
        self.pool = pool
        self.tokens = tokens
        self.client = client
        self.registry = registry
        self.store = store
        self.capture = capture

    # ------------------------------------------------------------------ #
    # upstream plumbing
    # ------------------------------------------------------------------ #

    def _build(self, payload: dict, variant: str) -> dict:
        return build_upstream_body(
            payload, variant, self.cfg.models.default_max_tokens,
            anthropic_cache_control=self.cfg.upstream.anthropic_cache_control)

    async def _send(self, account: Account, body: dict) -> UpstreamResponse:
        headers = build_headers(account.access_token,
                                self.cfg.upstream.fingerprint)
        return await self.client.send(headers, body, stream=True)

    async def _open(self, payload: dict, variant: str, *,
                    dialect: str = "openai", client_key: str = "anonymous",
                    model: str = "") -> tuple[Account, dict, dict, UpstreamResponse, str]:
        """Open a model, optionally advancing the configured free-model chain."""
        selected = payload.get("model", "")
        fallback = self.cfg.models.auto_free_fallback
        chain = list(dict.fromkeys(
            self.registry.resolve(candidate, dialect) for candidate in fallback.chain))
        selected = self.registry.resolve(selected, dialect)
        if not (fallback.enabled and self.registry.is_free_model(selected)
                and selected in chain):
            return await self._open_model(payload, variant, dialect=dialect,
                                          client_key=client_key, model=model)

        attempted: list[str] = []
        for candidate in chain[chain.index(selected):]:
            if not self.registry.is_free_model(candidate):
                continue
            candidate_payload = dict(payload)
            candidate_payload["model"] = candidate
            candidate_variant = self.registry.variant_for(candidate_payload["model"])
            attempted.append(candidate_payload["model"])
            try:
                return await self._open_model(
                    candidate_payload, candidate_variant, dialect=dialect,
                    client_key=client_key, model=candidate_payload["model"])
            except UpstreamFailure as exc:
                accounts = await self.pool.all()
                eligible = [account for account in accounts
                            if account.access_token
                            and account.state.value not in ("dead", "exhausted")]
                all_capped = bool(eligible) and all(
                    account.is_model_capped(candidate_payload["model"])
                    for account in eligible)
                is_cap = exc.code.upper() in {
                    "INFERENCE_CAP_ERROR", "DAILY_FREE_LIMIT", "FREE_TIER_LIMIT",
                    "MODEL_CAPPED", "MODEL_PARKED",
                } or is_entitlement_code(exc.code)
                if not (is_cap and all_capped):
                    raise
                if not chain[chain.index(candidate) + 1:]:
                    raise UpstreamFailure(
                        429, "AUTO_FREE_MODELS_EXHAUSTED",
                        "all configured free models are capped",
                        details={"attempted_models": attempted,
                                 "configured_models": chain})

        raise UpstreamFailure(429, "AUTO_FREE_MODELS_EXHAUSTED",
                              "all configured free models are capped",
                              details={"attempted_models": attempted,
                                       "configured_models": chain})

    async def _open_model(self, payload: dict, variant: str, *,
                    dialect: str = "openai", client_key: str = "anonymous",
                    model: str = "",
                    ) -> tuple[Account, dict, dict, UpstreamResponse, str]:
        """Acquire an account and open a 200 stream, handling failover.

        Rotation policy (see PLAN §4.5):
          402 insufficient_credits -> retire the account, move on
          401                      -> refresh token, retry once, else kill
          429                      -> cool the account, move on
          5xx / transport          -> transient: retry, only rotate if an
                                      alternative account actually exists
          4xx (other)              -> client error: surface, never burn a slot
                                      (unknown models get one alternate-variant
                                      probe retry first — see below)

        Returns (account, headers, body, response, variant_used). `variant_used`
        may differ from the caller's `variant` when the unknown-model probe
        switched shapes; the winner is cached in the registry.
        """
        last_error: tuple[int, str, str] | None = None
        max_attempts = self.cfg.upstream.max_attempts
        tried: set[str] = set()
        started = time.time()
        # details of the last failed exchange, recorded to the capture on the way out
        failure: tuple | None = None
        attempt = 0              # counts transient (transport/5xx) retries for backoff

        # Unknown-model probe: when the variant for this model is not
        # established, a client error (400/413) on the first shape triggers a
        # single retry with the alternate shape; whichever works is learned.
        upstream_model = payload.get("model", "")
        probing = (bool(self.cfg.models.probe_unknown)
                   and not self.registry.knows(upstream_model))
        # try the resolved shape first, then the other known shapes (bounded:
        # every probe costs an upstream round-trip)
        probe_order = [variant] + [v for v in VARIANTS if v != variant][:2]
        probe_i = 0

        # Credit-free models (cline-free/*, *:free) do not consume Cline Credits,
        # so a paid-lane 402 must not take the account out of rotation for them.
        is_free = self.registry.is_free_model(payload.get("model", ""))
        # Lane-aware routing: plan models (cline-pass/*, cline-cloud/*) are gated
        # by a subscription, not by the credit balance — an account whose usage
        # balance is exhausted must still be routable for them.
        require_paid = self.registry.lane_for(payload.get("model", "")) == "usage"

        held: Account | None = None      # reservation owned by this call
        transferred = False              # True once the caller takes it over
        try:
            for _ in range(max_attempts):
                account = await self.pool.acquire(
                    tried, require_paid=require_paid, model=upstream_model,
                    wait_seconds=(self.cfg.pool.acquire_wait_seconds
                                  if not tried else 0.0))
                if account is None and require_paid:
                    # every account is paid-exhausted: try anyway so the client gets
                    # the upstream 402 rather than a generic "no accounts"
                    account = await self.pool.acquire(tried, require_paid=False,
                                                      model=upstream_model,
                                                      wait_seconds=0.0)
                if account is None and tried:
                    # nothing untried left (e.g. a single-account pool) — allow reuse
                    account = await self.pool.acquire(set(), require_paid=require_paid,
                                                      model=upstream_model,
                                                      wait_seconds=0.0)
                if account is None:
                    # If the only obstacle is a parked model, answer with the real
                    # upstream reason instead of a generic "no usable account".
                    # Prefer the last classified error over the parked record: the
                    # parked message may be empty, and the chain handler keys on the
                    # code — normalise it so a cap/entitlement is always recognised
                    # as such (never surfaces as a raw 429 that stalls the chain).
                    if (last_error and last_error[1] and (
                            self._is_model_cap(
                                {"code": last_error[1], "message": last_error[2]},
                                True)
                            or is_entitlement_code(last_error[1]))):
                        parked = await self.pool.unavailable_reason(upstream_model)
                        self._record_failure(failure, client_key, dialect, model,
                                             variant, upstream_model)
                        raise UpstreamFailure(
                            last_error[0], last_error[1],
                            (last_error[2]
                             or (parked or {}).get("message")
                             or f"model {upstream_model} is temporarily unavailable"),
                            release_at=(parked or {}).get("release_at"),
                            release_in_s=(parked or {}).get("release_in_s"))
                    parked = await self.pool.unavailable_reason(upstream_model)
                    if parked:
                        self._record_failure(failure, client_key, dialect, model,
                                             variant, upstream_model)
                        raise UpstreamFailure(
                            parked["status"], parked["code"],
                            parked["message"] or f"model {upstream_model} is "
                                                 f"temporarily unavailable",
                            release_at=parked.get("release_at"),
                            release_in_s=parked.get("release_in_s"))
                    break

                held = account

                if account.needs_refresh(self.cfg.pool.refresh_lead_seconds):
                    await self.tokens.refresh(account)
                    # Unusable (refresh failed and nothing reloadable): rotate away
                    # instead of sending a known-expired token upstream.
                    if not account.access_token or account.expires_in() <= 0:
                        await self.pool.release(account)
                        held = None
                        tried.add(account.id)
                        continue

                body = self._build(payload, variant)
                try:
                    resp = await self._send(account, body)
                except TransportError as exc:
                    await self.pool.release(account)
                    held = None
                    tried.add(account.id)
                    last_error = (502, "transport_error", str(exc))
                    await asyncio.sleep(self._retry_pause(attempt))
                    attempt += 1
                    continue

                headers = build_headers(account.access_token,
                                        self.cfg.upstream.fingerprint)

                if resp.status_code == 200:
                    if probing:
                        self.registry.learn(upstream_model, variant)
                    transferred = True
                    return account, headers, body, resp, variant

                raw = await resp.aread()
                resp_headers = dict(resp.headers)
                no_retry = str(resp.headers.get("no-retry", "")).lower() == "true"
                await resp.aclose()
                await self.pool.release(account)
                held = None
                tried.add(account.id)

                kind = classify(resp.status_code, raw)
                info = parse_upstream_error(resp.status_code, raw)
                last_error = (resp.status_code, info["code"], info["message"])
                failure = (account, headers, body, resp.status_code, resp_headers,
                           raw.decode("utf-8", "replace"), (time.time() - started) * 1000)

                if kind is ErrorKind.INSUFFICIENT_CREDITS:
                    if is_free:
                        if info["code"] == "insufficient_credits":
                            # a credit error on a credit-free model is unexpected:
                            # treat it as transient rather than retiring the lane
                            log.warning("free-lane 402 from account %s (%s)",
                                        account.id, info["code"])
                            await self.pool.cool(account, reason="free_lane_402")
                        else:
                            # a non-credit 402 on a free model is most likely the
                            # (still-uncaptured) free-tier throttle: cool longer
                            log.warning("free-lane %s from account %s (%s)",
                                        resp.status_code, account.id, info["code"])
                            await self.pool.cool(
                                account,
                                seconds=self.cfg.pool.cooldown_seconds * 2,
                                reason=f"free_lane_{info['code']}")
                    else:
                        log.warning("paid lane exhausted on account %s (%s)",
                                    account.id, info["code"])
                        await self.pool.retire(account, reason=info["code"],
                                               paid_only=True)
                    continue

                if kind is ErrorKind.ENTITLEMENT:
                    # 403 ENTITLEMENT_ERROR: this account is not subscribed to this
                    # model's plan. Model-scoped — park the model, leave the account
                    # alone. A plan does not change minute to minute, so park longer.
                    seconds = self._parse_cap_seconds(info["message"])
                    if seconds <= 3600:
                        seconds = ENTITLEMENT_PARK_SECONDS
                    await self.pool.cap_model(account, upstream_model, seconds,
                                              reason=info["code"],
                                              message=info["message"])
                    log.warning("entitlement: %s unavailable on account %s (%s)",
                                upstream_model, account.id, info["code"])
                    self._record_failure(failure, client_key, dialect, model,
                                         variant, upstream_model)
                    failure = None       # recorded; do not record it twice on exit
                    last_error = (resp.status_code, info["code"], info["message"])
                    # Model-scoped, per account: another account may be entitled to
                    # this very model, so keep rotating. If every account is parked,
                    # the next acquire() finds nothing and `unavailable_reason`
                    # raises with this model's own reason + release time.
                    continue

                if kind is ErrorKind.UNAUTHORIZED:
                    # force=True: the upstream rejected a token our local expiry
                    # still likes (revoked early). Without it, refresh() would
                    # return True without calling the API and we would resend the
                    # exact rejected token.
                    ok = await self.tokens.refresh(account, force=True)
                    if not ok:
                        # a single refresh failure can be transient (network); only
                        # retire the account after repeated failures
                        if account.error_count >= 3:
                            await self.pool.kill(account, reason="refresh_failed")
                        else:
                            await self.pool.cool(account, reason="refresh_failed")
                        continue
                    # retry the same account once with the fresh token. The original
                    # reservation was released above (before classification), so
                    # take it back first — otherwise the final release in
                    # complete()/stream_from() would decrement another request's
                    # in-flight slot.
                    tried.discard(account.id)
                    await self.pool.reacquire(account)
                    held = account
                    try:
                        resp = await self._send(account, body)
                    except TransportError as exc:
                        await self.pool.cool(account, reason="transport after refresh")
                        await self.pool.release(account)
                        held = None
                        last_error = (502, "transport_error", str(exc))
                        continue
                    if resp.status_code == 200:
                        if probing:
                            self.registry.learn(upstream_model, variant)
                        transferred = True
                        return account, build_headers(account.access_token,
                                                      self.cfg.upstream.fingerprint), \
                               body, resp, variant
                    raw = await resp.aread()
                    await resp.aclose()
                    info = parse_upstream_error(resp.status_code, raw)
                    last_error = (resp.status_code, info["code"], info["message"])
                    # still 401 with a freshly refreshed token: do not kill the
                    # account on a transient upstream auth blip — cool and move on
                    await self.pool.cool(account, reason=info["code"])
                    await self.pool.release(account)
                    held = None
                    continue

                if kind is ErrorKind.RATE_LIMITED:
                    if self._is_model_cap(info, no_retry):
                        # A per-model daily cap (INFERENCE_CAP_ERROR on a free model).
                        # The account is fine and its other models still work, so this
                        # must NOT cool the account — only park the model.
                        seconds = self._parse_cap_seconds(info["message"])
                        await self.pool.cap_model(account, upstream_model, seconds,
                                                  reason=info["code"],
                                                  message=info["message"])
                        log.warning("model cap: %s parked on account %s for %.0fs (%s)",
                                    upstream_model, account.id, seconds, info["code"])
                        self._record_failure(failure, client_key, dialect, model,
                                             variant, upstream_model)
                        failure = None   # recorded; do not record it twice on exit
                        last_error = (resp.status_code, info["code"], info["message"])
                        # Model-scoped, per account: keep rotating — another account
                        # may still have quota for this model. When all are parked,
                        # `unavailable_reason` raises with the parsed release window.
                        continue
                    await self.pool.cool(account, reason=info["code"])
                    continue

                if kind is ErrorKind.EDGE_BLOCK:
                    # A 403 served as HTML by the CDN/edge, for every model. It is
                    # infrastructure, not the API, and not the account's fault - so it
                    # must not be surfaced as "you are forbidden", and it must not
                    # retire or kill anything. Back off, then let the client retry.
                    log.warning("edge block (403) on account %s: %s", account.id,
                                edge_block_message(raw))
                    await self.pool.cool(account,
                                         seconds=self.cfg.pool.cooldown_seconds * 4,
                                         reason="edge_block")
                    self._record_failure(failure, client_key, dialect, model,
                                         variant, upstream_model)
                    raise UpstreamFailure(503, "EDGE_BLOCK",
                                          edge_block_message(raw))

                if kind is ErrorKind.SERVER:
                    # upstream-side fault: retry, and only cool when we have a backup
                    other_ready = [a for a in await self.pool.all()
                                   if a.is_ready() and a.id != account.id]
                    if other_ready:
                        await self.pool.cool(account, reason=info["code"])
                    await asyncio.sleep(self._retry_pause(attempt))
                    attempt += 1
                    continue

                # client error — surface to the client, never rotate for this.
                # An unknown model earns exactly one alternate-variant probe retry,
                # unless the upstream marked the response no-retry.
                if probing and not no_retry and probe_i < len(probe_order) - 1:
                    probe_i += 1
                    variant = probe_order[probe_i]
                    log.info("probe: %s rejected variant %s (%s), retrying with %s",
                             upstream_model, probe_order[probe_i - 1],
                             info["code"], variant)
                    tried.discard(account.id)
                    continue
                self._record_failure(failure, client_key, dialect, model, variant,
                                     upstream_model)
                raise UpstreamFailure(resp.status_code, info["code"], info["message"])

        finally:
            if held is not None and not transferred:
                # An exception or cancellation escaped while a reservation
                # was still held (e.g. a mid-read httpx failure, client
                # disconnect). Release on a shielded task: a cancelled
                # caller must not swallow the release itself.
                try:
                    await asyncio.shield(self.pool.release(held))
                except Exception:
                    log.exception("failed to release leaked reservation for %s",
                                  held.id)
        status, code, message = last_error or (
            503, "no_accounts_available", "no usable Cline account in the pool")
        self._record_failure(failure, client_key, dialect, model, variant,
                             upstream_model)
        raise UpstreamFailure(status, code, message)

    # ------------------------------------------------------------------ #
    # error capture + cap parsing
    # ------------------------------------------------------------------ #

    # Captured 2026-09-16: HTTP 429, header `no-retry: true`, body
    # {"error":{"code":"INFERENCE_CAP_ERROR","message":"Error 429: Daily free
    #  limit reached on model zai/glm-5.3-flash. Try again in 11h 31m"}}
    CAP_CODES = ("INFERENCE_CAP_ERROR", "DAILY_FREE_LIMIT", "FREE_TIER_LIMIT")

    @classmethod
    def _is_model_cap(cls, info: dict, no_retry: bool) -> bool:
        code = (info.get("code") or "").upper()
        if code in cls.CAP_CODES:
            return True
        message = (info.get("message") or "").lower()
        return no_retry and "free limit" in message

    @staticmethod
    def _parse_cap_seconds(message: str) -> float:
        """'Try again in 11h 31m' -> seconds. Falls back to one hour.

        `\\bin\\b` matters: without the word boundaries the pattern matches the
        'in ' inside 'again', yielding an empty match and the 1-hour fallback.
        """
        match = re.search(r"\bin\b\s+(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?",
                          message or "", re.I)
        if match and (match.group(1) or match.group(2)):
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            return float(max(hours * 3600 + minutes * 60, 60))
        return 3600.0

    @staticmethod
    def _noop() -> None:      # keeps the helpers section obvious in diffs
        return None

    def _record_failure(self, failure, client_key: str, dialect: str,
                        model: str, variant: str, upstream_model: str) -> None:
        """Persist a failed exchange — the success path records in complete()/stream(),
        but a raise from _open would otherwise leave no capture behind."""
        if not failure:
            return
        account, headers, body, status, resp_headers, text, duration_ms = failure

        def _write() -> None:
            # stream=True: _open only ever opens streaming upstream connections,
            # so recording a failed open as non-stream skews the usage ledger
            self._record(account, headers, body, status, resp_headers, text,
                         duration_ms, client_key, dialect, upstream_model,
                         variant, stream=True)

        try:
            # sqlite I/O off the event loop; the raise below must not be delayed
            task = asyncio.get_running_loop().create_task(
                asyncio.to_thread(_write))

            def _log_write_error(t: asyncio.Task) -> None:
                exc = t.exception()
                if exc is not None:
                    log.exception("failed to record error exchange", exc_info=exc)

            task.add_done_callback(_log_write_error)
        except RuntimeError:
            # no loop (sync test context): write inline
            try:
                _write()
            except Exception:
                log.exception("failed to record error exchange")

    @staticmethod
    def _retry_pause(attempt: int) -> float:
        """Escalating backoff for transient (transport/5xx) retries.

        A fixed pause hammered the upstream when the pool had a single account
        and every attempt reused it. attempt is 0-based within one request.
        """
        return min(RETRY_PAUSE_SECONDS * (2 ** max(attempt, 0)), 5.0)

    # ------------------------------------------------------------------ #
    # non-streaming
    # ------------------------------------------------------------------ #

    async def complete(self, payload: dict, *, model: str, variant: str,
                       dialect: str, client_key: str) -> CallResult:
        started = time.time()
        account, headers, body, resp, variant = await self._open(
            payload, variant, dialect=dialect, client_key=client_key, model=model)
        try:
            try:
                raw = await resp.aread()
            except httpx.HTTPError as exc:
                # a mid-read failure is an upstream fault: surface it as the
                # dialect-shaped 502 the streaming path already uses instead of
                # letting a bare httpx exception reach the generic 500 handler
                raise UpstreamFailure(
                    502, "upstream_read_error",
                    f"upstream read failed: {exc.__class__.__name__}") from exc
            upstream_headers = resp.headers
        finally:
            # shielded: a cancelled or failing close must not skip the release
            await asyncio.shield(self._close_and_release(resp, account))

        duration_ms = (time.time() - started) * 1000
        # the upstream body is SSE even for a "non-stream" client (we always
        # stream upstream and aggregate locally), so usage lives in the final
        # SSE frame, not in a JSON document — parse it as SSE, not json.loads
        text = raw.decode("utf-8", "replace")
        # a 200 that carries an upstream error frame is not a success: the
        # routes answer 502 for this shape, so the pool ledger must agree
        # (mirrors the streaming path's stream_ok)
        ok = not self._any_error_event(text)
        if ok:
            await self.pool.on_success(account)
        usage = self._usage_from_sse(text)
        try:
            await asyncio.to_thread(
                self._record, account, headers, body,
                resp.status_code if ok else 502,
                upstream_headers, text, duration_ms, client_key,
                dialect, model, variant, stream=False, usage=usage)
        except Exception:
            # telemetry must never turn an already-successful upstream call
            # into a client-visible failure (disk full, locked sqlite, ...)
            log.exception("failed to record exchange")
        return CallResult(account, resp.status_code, raw, duration_ms,
                          upstream_headers)

    # ------------------------------------------------------------------ #
    # streaming
    # ------------------------------------------------------------------ #

    async def _close_and_release(self, resp, account) -> None:
        """Shielded teardown for non-streaming exchanges.

        Closing must never mask a successful call or skip the slot release:
        a failed close turns into a log line, not an exception.
        """
        try:
            await resp.aclose()
        except Exception:
            log.warning("failed to close upstream response for %s",
                        account.id, exc_info=True)
        try:
            await self.pool.release(account)
        except Exception:
            log.exception("failed to release account %s", account.id)

    async def open_stream(self, payload: dict, *, model: str, variant: str,
                          dialect: str, client_key: str) -> "StreamHandle":
        """Resolve the upstream connection BEFORE any response headers are sent.

        This exists because a streaming route cannot change its status code once
        the first byte is on the wire. Opening lazily inside the generator meant a
        failure here (no credit, no account, an edge block) surfaced to the client
        as a bare connection reset instead of a readable error. Callers open first,
        then hand the handle to `stream_from`.
        """
        account, headers, body, resp, used_variant = await self._open(
            payload, variant, dialect=dialect, client_key=client_key, model=model)
        return StreamHandle(account=account, headers=headers, body=body,
                            resp=resp, variant=used_variant,
                            started=time.time())

    async def stream(self, payload: dict, *, model: str, variant: str,
                     dialect: str, client_key: str,
                     anthropic: bool) -> AsyncIterator[bytes]:
        """Convenience wrapper: open, then stream. Prefer open_stream + stream_from
        in HTTP handlers so failures can still be reported with a real status."""
        handle = await self.open_stream(payload, model=model, variant=variant,
                                        dialect=dialect, client_key=client_key)
        async for chunk in self.stream_from(handle, model=model, dialect=dialect,
                                            client_key=client_key,
                                            anthropic=anthropic):
            yield chunk

    async def stream_from(self, handle: "StreamHandle", *, model: str,
                          dialect: str, client_key: str,
                          anthropic: bool) -> AsyncIterator[bytes]:
        account = handle.account
        headers = handle.headers
        body = handle.body
        resp = handle.resp
        variant = handle.variant
        started = handle.started

        # Bounded retention: capture only ever keeps a prefix (its JSONL writer
        # truncates at 200 KB anyway) and usage parsing only needs the final
        # chunks. Unbounded accumulation pinned the whole stream in memory for
        # the request lifetime even with capture disabled.
        CAPTURE_LIMIT = 256_000
        # the usage frame sits in the final chunk(s); keep a byte-bounded tail,
        # not N chunks — httpx chunks are unbounded in size
        TAIL_LIMIT = 65_536
        captured: list[str] = []
        captured_len = 0
        tail: deque[str] = deque()
        tail_len = 0
        translator = AnthropicStreamTranslator(model) if anthropic else None
        pending = b""
        status = resp.status_code
        # clean completion drives account state and the recorded status;
        # `error_emitted` stops the Anthropic translator from ending an
        # already-failed message with normal message_delta/message_stop events.
        stream_ok = True
        error_emitted = False

        try:
            if anthropic:
                yield translator._start()          # noqa: SLF001
                translator.sent_message_start = True

            try:
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    text = chunk.decode("utf-8", "replace")
                    if captured_len < CAPTURE_LIMIT:
                        captured.append(text)
                        captured_len += len(text)
                    tail.append(text)
                    tail_len += len(text)
                    while tail_len > TAIL_LIMIT and len(tail) > 1:
                        tail_len -= len(tail.popleft())

                    if anthropic:
                        pending += chunk
                        frames, pending = _split_sse_frames(pending)
                        for frame in frames:
                            for event in iter_sse_events(
                                    frame.decode("utf-8", "replace")):
                                if event == "[DONE]":
                                    continue
                                # an error frame inside a 200 stream must not be
                                # silently dropped - Anthropic clients understand
                                # an `error` event and stop cleanly
                                if isinstance(event, dict) and event.get("error"):
                                    err = event["error"]
                                    if isinstance(err, dict):
                                        code = str(err.get("code") or "stream_error")
                                        msg = err.get("message") or err
                                    else:
                                        code, msg = "stream_error", str(err)
                                    log.warning("upstream sent an error frame: %s", msg)
                                    stream_ok = False
                                    error_emitted = True
                                    yield _sse_event("error", {
                                        "type": "error",
                                        "error": {"type": "api_error",
                                                  "code": code,
                                                  "message": str(msg)},
                                    })
                                    continue
                                if error_emitted:
                                    # the message already failed; do not feed
                                    # further content after the error event
                                    continue
                                for out in translator.feed(event):
                                    yield out
                    else:
                        # byte-exact passthrough, but sniff the frames: an
                        # error frame inside a 200 stream must not be recorded
                        # as a success (the non-stream path maps the same
                        # shape to a 502)
                        pending += chunk
                        frames, pending = _split_sse_frames(pending)
                        for frame in frames:
                            if self._any_error_event(
                                    frame.decode("utf-8", "replace")):
                                stream_ok = False
                        yield chunk
            except httpx.HTTPError as exc:
                # The stream died after bytes were already sent, so there is no
                # retrying transparently. Emit a proper error frame in the
                # client's dialect instead of silently truncating the response
                # (a client-side "operation timed out" is exactly what a
                # truncated stream looks like from the other end).
                interrupted = f"upstream stream interrupted: {exc.__class__.__name__}"
                log.warning("%s (model=%s)", interrupted, model)
                stream_ok = False
                if anthropic:
                    yield _sse_event("error", {
                        "type": "error",
                        "error": {"type": "api_error", "message": interrupted},
                    })
                else:
                    yield sse_frame({
                        "error": {"message": interrupted,
                                  "type": "api_error",
                                  "code": "stream_interrupted"},
                    })
                    yield b"data: [DONE]\n\n"
                return

            if anthropic and not error_emitted:
                if pending.strip():
                    for event in iter_sse_events(
                            pending.decode("utf-8", "replace")):
                        if event != "[DONE]":
                            for out in translator.feed(event):
                                yield out
                for out in translator.finish():
                    yield out
            elif pending.strip() and self._any_error_event(
                    pending.decode("utf-8", "replace")):
                # an error frame split across the final chunks: the bytes were
                # already passed through, but this was not a success
                stream_ok = False

        finally:
            # shielded: cancellation from a client disconnect must not skip
            # the slot release, and a failing close must not mask the response
            await asyncio.shield(self._finish_stream(
                resp=resp, account=account, headers=headers, body=body,
                status=status, stream_ok=stream_ok, started=started,
                captured=captured, tail=tail, client_key=client_key,
                dialect=dialect, model=model, variant=variant))

    async def _finish_stream(self, *, resp, account, headers, body, status,
                             stream_ok, started, captured, tail, client_key,
                             dialect, model, variant) -> None:
        """Shielded teardown for a streaming exchange; never raises.

        Every step is best-effort: the response bytes are already on the wire,
        so a failed close or a failed release must not surface as a client
        error, and an error-carrying stream must not clear the account state.
        """
        sse_text = "".join(captured)
        try:
            await resp.aclose()
        except Exception:
            log.warning("failed to close upstream stream for %s",
                        account.id, exc_info=True)
        try:
            await self.pool.release(account)
        except Exception:
            log.exception("failed to release account %s after a stream",
                          account.id)
        duration_ms = (time.time() - started) * 1000
        if stream_ok:
            # only a clean completion clears error state / un-cools the
            # account; an interrupted or error-carrying stream must not
            # (it used to, hiding real failures from the pool).
            try:
                await self.pool.on_success(account)
            except Exception:
                log.exception("failed to mark stream success for %s", account.id)
        usage = self._usage_from_sse("".join(tail))
        try:
            await asyncio.to_thread(
                self._record, account, headers, body,
                status if stream_ok else 502, resp.headers, sse_text,
                duration_ms, client_key, dialect, model, variant,
                stream=True, usage=usage)
        except Exception:
            log.exception("failed to record stream exchange")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _any_error_event(text: str) -> bool:
        """True when an SSE buffer carries an upstream error frame.

        Captured shape: a 200 stream whose frames include
        {"error": {"code": "...", ...}} instead of choices. The non-stream path
        maps that shape to a 502; streaming must at least not treat it as a
        success (which would clear cooldowns and the error ledger).
        """
        return any(isinstance(e, dict) and e.get("error")
                   for e in iter_sse_events(text))

    @staticmethod
    def _usage_from_sse(text: str) -> dict:
        chunks = [e for e in iter_sse_events(text) if isinstance(e, dict)]
        return extract_openai_usage(chunks)

    def _record(self, account: Account, headers: dict, body: dict, status: int,
                upstream_headers: dict, response_text: str, duration_ms: float,
                client_key: str, dialect: str, model: str, variant: str,
                *, stream: bool, usage: dict | None = None) -> None:
        usage = usage or {}
        if not usage:
            try:
                parsed = json.loads(response_text)
                usage = parsed.get("usage") or {}
            except Exception:
                usage = {}

        self.store.record(
            account_id=account.id,
            client_key=client_key,
            dialect=dialect,
            model=model,
            variant=variant,
            stream=1 if stream else 0,
            status=status,
            duration_ms=duration_ms,
            prompt_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
            completion_tokens=(usage.get("completion_tokens")
                               or usage.get("output_tokens")),
            total_tokens=usage.get("total_tokens"),
            error_code=None if status == 200 else f"http_{status}",
        )

        self.capture.write(
            request_headers=headers,
            request_body=body,
            status=status,
            response_headers=upstream_headers,
            response_body=response_text,
            duration_ms=duration_ms,
            account_id=account.id,
            dialect=dialect,
            model=model,
            variant=variant,
        )


# --------------------------------------------------------------------------- #
# aggregation (stream -> non-stream for clients that don't stream)
# --------------------------------------------------------------------------- #


def aggregate_openai_stream(text: str, model: str, fallback_id: str) -> dict:
    """Collapse a captured SSE stream into a single chat.completion object."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    refusal: str | None = None
    finish_reason: str | None = None
    completion_id = fallback_id
    created = int(time.time())
    usage: dict = {}
    upstream_error: str | None = None

    for event in iter_sse_events(text):
        if not isinstance(event, dict):
            continue
        completion_id = event.get("id", completion_id)
        created = event.get("created", created)
        if event.get("model"):
            model = event["model"]
        if event.get("usage"):
            usage = event["usage"]
        if event.get("error"):
            upstream_error = str(event["error"])
        for choice in (event.get("choices") or []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            # Reasoning models stream their thinking in a `reasoning` delta
            # (captured 2026-09-19, cline-free/kimi-k3). Collect it so
            # non-streaming callers receive it instead of losing it.
            if delta.get("reasoning"):
                reasoning_parts.append(delta["reasoning"])
            # Providers signal a policy refusal with a `refusal` delta, not
            # `content`. Dropping it leaves the caller with an empty answer and
            # only a `content_filter` code, so they invent their own message.
            if delta.get("refusal"):
                refusal = delta["refusal"]
            for call in (delta.get("tool_calls") or []):
                idx = call.get("index", 0)
                slot = tool_calls.setdefault(idx, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    message: dict[str, Any] = {"role": "assistant",
                               "content": "".join(content_parts) or None}
    if reasoning_parts:
        # OpenRouter-style additive field: clients that understand reasoning
        # read it, everyone else ignores it
        message["reasoning"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls)]
    if refusal:
        # OpenAI's message object has a `refusal` field for exactly this
        message["refusal"] = refusal
        if finish_reason is None:
            finish_reason = "content_filter"
    if upstream_error and not content_parts and not tool_calls:
        message["content"] = upstream_error

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason or "stop",
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0},
    }

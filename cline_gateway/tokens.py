"""Token lifecycle: proactive refresh, single-flight per account, persistence.

The `/api/v1/auth/refresh` response shape was captured (2026-09-16);
`tests/test_token_refresh.py` pins it. Refreshing is deliberately defensive: on
failure it falls back to re-reading the source config (the desktop app may have
rotated the token itself). A refresh manages tokens only — account state
(COOLING / EXHAUSTED / DEAD) belongs to the pool and is never changed here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from .config import Config
from .pool import Account, AccountState, PoolManager, load_accounts

log = logging.getLogger("cline_gateway.tokens")

REFRESH_PATH = "/auth/refresh"
REFRESH_UA = "Bun/1.3.13"        # the client's UA on this endpoint differs from
                                 # the chat fingerprint (captured)
TOKEN_PREFIX = "workos:"         # stored tokens carry this; the refresh response does not


def _parse_expiry(value: Any) -> int:
    """Normalise an expiry to epoch milliseconds.

    Captured shape is an ISO 8601 string ("2026-09-16T02:52:30Z"); epoch seconds
    and epoch milliseconds are also accepted defensively.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        v = int(value)
        return v if v > 10_000_000_000 else v * 1000
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return _parse_expiry(int(text))
        try:
            from datetime import datetime, timezone
            iso = text.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return 0
    return 0


def _extract_token(payload: Any) -> dict[str, Any] | None:
    """Pull {access_token, refresh_token, expires_at} out of a response body.

    Captured shape:
        {"data": {"accessToken": "...", "tokenType": "Bearer",
                  "expiresAt": "2026-09-16T02:52:30Z",
                  "refreshToken": "...", "userInfo": {...}}, "success": true}
    """
    if not isinstance(payload, dict):
        return None

    # unwrap common envelopes
    for key in ("data", "result", "auth"):
        inner = payload.get(key)
        if isinstance(inner, dict):
            found = _extract_token(inner)
            if found:
                return found

    access = (payload.get("accessToken") or payload.get("access_token")
              or payload.get("token"))
    if not access:
        return None

    return {
        "access_token": access,
        "refresh_token": payload.get("refreshToken") or payload.get("refresh_token"),
        "expires_at": _parse_expiry(payload.get("expiresAt")
                                    or payload.get("expires_at")),
    }


class TokenManager:
    def __init__(self, cfg: Config, pool: PoolManager) -> None:
        self.cfg = cfg
        self.pool = pool
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0), headers={},
        )
        self._locks: dict[str, asyncio.Lock] = {}
        # bound concurrent refreshes: N accounts expiring together must not fire
        # N simultaneous HTTPS calls with the identical desktop UA
        self._refresh_sem = asyncio.Semaphore(4)
        self._stop = asyncio.Event()
        # in-flight refresh tasks spawned by the background loop, so shutdown
        # can cancel them instead of closing the HTTP client underneath them
        self._refresh_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ #

    def _lock_for(self, account_id: str) -> asyncio.Lock:
        lock = self._locks.get(account_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[account_id] = lock
        return lock

    def drop_lock(self, account_id: str) -> None:
        """Forget the per-account lock (called when the pool drops the account);
        an unlocked entry is never awaited again, so removal is safe."""
        lock = self._locks.pop(account_id, None)
        if lock is not None and lock.locked():
            # a refresh is mid-flight: keep it alive under a sentinel so the
            # running coroutine finishes against a valid lock object
            self._locks[f"{account_id}#zombie"] = lock

    async def aclose(self) -> None:
        self._stop.set()
        for task in list(self._refresh_tasks):
            task.cancel()
        if self._refresh_tasks:
            await asyncio.gather(*self._refresh_tasks, return_exceptions=True)
            self._refresh_tasks.clear()
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    # refresh
    # ------------------------------------------------------------------ #

    async def _refresh_bounded(self, account: Account) -> bool:
        async with self._refresh_sem:
            return await self.refresh(account)

    async def refresh(self, account: Account, force: bool = False) -> bool:
        """Single-flight refresh. Returns True if the account has a usable token.

        Captured contract:
            POST {base}/auth/refresh
            body: {"refreshToken": "...", "grantType": "refresh_token"}
            resp: {"data": {"accessToken","tokenType","expiresAt","refreshToken",
                            "userInfo"}, "success": true}
        The refresh token is not rotated, so the desktop client stays in sync.

        `force=True` refreshes even when the local expiry looks fine — used after
        an upstream 401, where the token was rejected (revoked early) while our
        local `expires_at` still claims it is valid. Concurrent forced refreshes
        for the same rejected token are deduplicated (one POST, not one each).
        """
        before = account.access_token
        async with self._lock_for(account.id):
            # a concurrent forced refresh may have already replaced the token
            # this caller was rejected with; re-POSTing for the same rejected
            # token is a refresh stampede (one POST per concurrent 401)
            if (force and account.access_token != before
                    and account.access_token
                    and account.expires_in() > 60):
                return True
            # another coroutine may have refreshed while we waited
            if (not force
                    and account.expires_in() > self.cfg.pool.refresh_lead_seconds
                    and account.access_token):
                return True
            if not account.refresh_token:
                log.warning("account %s has no refresh_token; cannot refresh", account.id)
                return await self._reload_from_source(account)

            url = self.cfg.upstream.base_url.rstrip("/") + REFRESH_PATH
            headers = {
                "content-type": "application/json",
                "accept": "*/*",
                "user-agent": REFRESH_UA,
            }
            body = {
                "refreshToken": account.refresh_token,
                "grantType": "refresh_token",
            }

            try:
                resp = await self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                log.warning("refresh transport error for %s: %s", account.id, exc)
                return await self._reload_from_source(account)

            if resp.status_code == 200:
                try:
                    parsed = _extract_token(resp.json())
                except Exception:
                    parsed = None
                if parsed:
                    if not parsed.get("expires_at"):
                        # No usable expiry in the response: keep the stale one and
                        # an already-expired account would "need refresh" again on
                        # every request (a refresh storm). Derive one from the
                        # JWT's own exp claim, with a conservative fallback.
                        parsed["expires_at"] = (
                            _jwt_exp_ms(parsed["access_token"])
                            or int((time.time() + 3300) * 1000))
                    self._apply(account, parsed)
                    await self._persist_refreshed(account)
                    log.info("refreshed token for %s (expires in %.0fs)",
                             account.id, account.expires_in())
                    return True

            log.warning("refresh failed for %s (HTTP %s); falling back to source",
                        account.id, resp.status_code)
            return await self._reload_from_source(account)

    def _apply(self, account: Account, parsed: dict[str, Any]) -> None:
        token = parsed["access_token"]
        # the stored token carries a "workos:" prefix; the refresh response does not
        if account.access_token.startswith(TOKEN_PREFIX) \
                and not token.startswith(TOKEN_PREFIX):
            token = TOKEN_PREFIX + token
        account.access_token = token

        if parsed.get("refresh_token"):
            account.refresh_token = parsed["refresh_token"]
        if parsed.get("expires_at"):
            account.expires_at = parsed["expires_at"]
        # Deliberately NOT touching state/error_count/last_error: a successful
        # token refresh must not un-cool an edge-blocked account or revive an
        # exhausted one. Pool transitions own those fields.

    async def _reload_from_source(self, account: Account) -> bool:
        """Re-read the credential source; the desktop app may have rotated it."""
        try:
            # blocking file I/O off the event loop (providers.json can be large
            # and AV-scanned on Windows)
            fresh = await asyncio.to_thread(load_accounts, self.cfg.accounts)
        except Exception as exc:
            log.debug("reload failed: %s", exc)
            return False
        for f in fresh:
            if f.id == account.id and f.access_token and f.expires_in() > 60:
                account.access_token = f.access_token
                account.refresh_token = f.refresh_token or account.refresh_token
                account.expires_at = f.expires_at
                # tokens only; pool state is not a token concern (see _apply)
                log.info("reloaded token for %s from %s", account.id, f.source)
                return True
        return False

    async def _persist_refreshed(self, account: Account) -> None:
        """Durably write refreshed credentials back to their source.

        Without this the new token exists only in memory: a restart reloads the
        stale source token, and if the refresh token ever rotates, the new one
        is lost permanently (account lockout). Best-effort by design.
        """
        try:
            src = self.cfg.accounts
            if (src.source == "accounts_dir"
                    and account.source.startswith("accounts/")):
                from .accounts_dir import parse_snapshot, write_snapshot
                path = Path(src.dir) / account.source.split("/", 1)[1]
                if path.is_file():
                    # blocking file I/O off the event loop
                    fields = await asyncio.to_thread(parse_snapshot, path)
                    fields["access_token"] = account.access_token
                    if account.refresh_token:
                        fields["refresh_token"] = account.refresh_token
                    fields["expires_at_ms"] = str(account.expires_at)
                    await asyncio.to_thread(write_snapshot, path, fields)
            elif src.source == "pool_file":
                persist_pool_snapshot(self.cfg, await self.pool.all())
            # providers_json is owned by the desktop app; do not rewrite it.
        except Exception:
            log.warning("could not persist refreshed token for %s",
                        account.id, exc_info=True)

    async def apply_credentials(self, account: Account, *,
                                access_token: str = "",
                                refresh_token: str = "",
                                expires_at: int = 0) -> None:
        """Update credentials under the account's refresh lock (admin reload).

        Mutating the Account directly raced concurrent requests building
        headers and in-flight refreshes; the per-account lock serializes them.
        """
        async with self._lock_for(account.id):
            if access_token:
                account.access_token = access_token
            if refresh_token:
                account.refresh_token = refresh_token
            if expires_at:
                account.expires_at = expires_at

    # ------------------------------------------------------------------ #
    # background workers
    # ------------------------------------------------------------------ #

    async def refresh_loop(self) -> None:
        """Refresh accounts before they expire."""
        lead = self.cfg.pool.refresh_lead_seconds
        while not self._stop.is_set():
            try:
                for account in await self.pool.all():
                    if account.state in (AccountState.DEAD,):
                        continue
                    if account.needs_refresh(lead):
                        task = asyncio.create_task(self._refresh_bounded(account))
                        self._refresh_tasks.add(task)
                        task.add_done_callback(self._refresh_tasks.discard)
            except Exception:
                log.exception("refresh loop iteration failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass

    async def check_account(self, account: Account) -> None:
        """Balance + paid-lane threshold + plan for one account, in one pass.

        The threshold logic used to live only in the periodic sweep, so the
        manual balance check and freshly added accounts (import, device login)
        showed an optimistic paid lane for up to one poll interval. Every path
        that learns a balance now settles the paid lane immediately.
        """
        bal = await self.fetch_balance(account)
        if bal is not None:
            if bal <= self.cfg.pool.min_balance_micro:
                if not account.paid_exhausted:
                    log.info("account %s balance=%s (<= threshold), "
                             "retiring", account.id, bal)
                    await self.pool.retire(
                        account, reason="balance_threshold")
            elif account.paid_exhausted or account.state is AccountState.EXHAUSTED:
                # topped up since the last check: back into rotation.
                log.info("account %s balance=%s recovered, "
                         "restoring paid lane", account.id, bal)
                await self.pool.restore(account)

        # a plan record gates cline-pass/* and cline-cloud/*; keep it fresh
        # (re-check hourly) so availability answers "no subscription"
        # instead of guessing — and a plan bought later is picked up.
        if account.has_plan is None or time.time() - (
                account.plan_checked_at or 0) > 3600:
            await self.fetch_plan(account)

    async def balance_sweep(self) -> None:
        """One pass over the pool: poll balances, retire/restore paid lanes, refresh plans."""
        for account in await self.pool.all():
            if not account.access_token:
                continue
            await self.check_account(account)

    async def balance_loop(self) -> None:
        """Poll per-account balance (drives quota_aware routing)."""
        interval = self.cfg.pool.balance_poll_seconds
        if interval <= 0:
            return
        while not self._stop.is_set():
            try:
                await self.balance_sweep()
            except Exception:
                log.exception("balance loop iteration failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def fetch_balance(self, account: Account) -> int | None:
        """GET /users/{id}/balance — captured shape: {"data":{"balance": -1615}}"""
        url = self.cfg.upstream.base_url.rstrip("/") + f"/users/{account.id}/balance"
        headers = {"authorization": f"Bearer {account.access_token}"}
        try:
            resp = await self._client.get(url, headers=headers)
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        inner = data.get("data", data)
        bal = inner.get("balance")
        if isinstance(bal, (int, float)):
            account.balance_micro = int(bal)
            account.balance_checked_at = time.time()
            return int(bal)
        return None

    async def fetch_plan(self, account: Account) -> bool | None:
        """GET /users/me/plan -> True if a plan record exists, False on the 404
        'no plan history found for user', None if the answer is unclear."""
        url = self.cfg.upstream.base_url.rstrip("/") + "/users/me/plan"
        headers = {"authorization": f"Bearer {account.access_token}",
                   "accept": "application/json"}
        try:
            resp = await self._client.get(url, headers=headers)
        except httpx.HTTPError:
            return None
        if resp.status_code == 200:
            account.has_plan = True
        elif resp.status_code == 404:
            account.has_plan = False
        else:
            return None
        account.plan_checked_at = time.time()
        return account.has_plan

    async def fetch_me(self, account: Account) -> dict | None:
        url = self.cfg.upstream.base_url.rstrip("/") + "/users/me"
        headers = {"authorization": f"Bearer {account.access_token}"}
        try:
            resp = await self._client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
        except httpx.HTTPError:
            pass
        return None


def _jwt_exp_ms(token: str) -> int | None:
    """exp claim (epoch seconds) of a JWT -> epoch milliseconds, best effort."""
    try:
        part = token.split(":", 1)[-1].split(".")[1]      # strip "workos:" prefix
        part += "=" * (-len(part) % 4)
        claims = json.loads(base64.urlsafe_b64decode(part))
        exp = claims.get("exp")
        return int(exp) * 1000 if exp else None
    except Exception as exc:
        log.warning("could not parse exp from JWT (%s); using fallback expiry",
                    exc.__class__.__name__)
        return None


def _atomic_write(path: Path, text: str) -> None:
    """Write via a sibling temp file + os.replace.

    Truncate-in-place can destroy a credential file mid-write (crash, disk
    full, AV lock), leaving a partial pool that fails to parse on the next
    start. os.replace is atomic on the same volume, including Windows.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def persist_pool_snapshot(cfg: Config, accounts: list[Account]) -> None:
    """Write the current pool to pool.json (secrets included — keep it restricted)."""
    path = Path(cfg.accounts.pool_file)
    payload = {
        "accounts": [
            {
                "id": a.id,
                "email": a.email,
                "access_token": a.access_token,
                "refresh_token": a.refresh_token,
                "expires_at": a.expires_at,
                "source": a.source,
            }
            for a in accounts
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(payload, indent=2))
    try:
        path.chmod(0o600)
    except OSError:
        pass

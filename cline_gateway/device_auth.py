"""Login with Cline: drive the WorkOS device-code flow the desktop app uses.

The user clicks "Login with Cline"; we request a device code from WorkOS, show
them the short code and the verification URL, poll until they approve in the
browser, exchange the WorkOS tokens for Cline tokens via /auth/register, and
drop the result into the accounts folder + live pool — no Cline install, no
manual token copying.

Wire shape (captured 2026-09-22, capture/cline-20260922-011813.jsonl):

    POST api.workos.com/user_management/authorize/device
        client_id=client_01K3...              -> device_code, user_code,
                                                 verification_uri(_complete),
                                                 expires_in, interval
    POST api.workos.com/user_management/authenticate   (every `interval` s)
        grant_type=urn:ietf:params:oauth:grant-type:device_code
        device_code=...&client_id=...         -> 400 authorization_pending
                                                 (while waiting) then 200 with
                                                 user, access_token, refresh_token
    POST api.cline.bot/api/v1/auth/register   (JSON, no fingerprint headers)
        {accessToken, refreshToken}           -> {data:{accessToken, refreshToken,
                                                 tokenType, expiresAt, userInfo}}

None of these calls need the chat fingerprint — only content-type + a plain UA.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .accounts_dir import write_snapshot
from .pool import Account, AccountState
from .tokens import _extract_token

log = logging.getLogger("cline_gateway.device_auth")

CLIENT_ID = "client_01K3A541FN8TA3EPPHTD2325AR"   # Cline's public WorkOS client id
DEVICE_AUTHORIZE = "https://api.workos.com/user_management/authorize/device"
AUTHENTICATE = "https://api.workos.com/user_management/authenticate"
REGISTER = "https://api.cline.bot/api/v1/auth/register"
UA = "Bun/1.3.13"          # same plain UA the desktop client uses here

# poll-loop guards
DEFAULT_INTERVAL = 5.0
MAX_INTERVAL = 15.0        # after a slow_down hint
DEFAULT_EXPIRES = 300.0


class DeviceLogin:
    """One login attempt: start -> poll -> register -> account.

    State machine: idle -> awaiting_user -> registering -> done | error |
    expired | cancelled. `status()` is a plain dict for the dashboard; the
    heavy lifting runs in `run()` as a task the caller owns.
    """

    def __init__(self, accounts_dir: Path,
                 client: httpx.AsyncClient | None = None,
                 on_account=None) -> None:
        self.accounts_dir = accounts_dir
        self.on_account = on_account      # async fn(Account), awaited on success
        self._own_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": UA, "Accept": "*/*"})
        self.state = "idle"
        self.error: str | None = None
        self.user_code: str | None = None
        self.verification_url: str | None = None
        self.account_email: str | None = None
        self.snapshot_path: Path | None = None
        self._device_code: str | None = None
        self._cancel = asyncio.Event()
        self._ready = asyncio.Event()      # set once the code (or the error) is known

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "user_code": self.user_code,
            "verification_url": self.verification_url,
            "error": self.error,
            "email": self.account_email,
        }

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """The whole flow. Sets self.state / self.error as it goes."""
        try:
            self.state = "awaiting_user"
            try:
                ok = await self._authorize()
            finally:
                self._ready.set()          # code known (or attempt failed)
            if not ok:
                return
            workos = await self._poll()
            if workos is None:
                return
            self.state = "registering"
            account = await self._register(workos)
            if account is None:
                return
            self._persist(account)
            if self.on_account is not None:
                await self.on_account(account)     # pool upsert (event loop)
            self.state = "done"
            log.info("device login complete for %s", account.email)
        except asyncio.CancelledError:
            self.state = "cancelled"
        except Exception as exc:                     # noqa: BLE001 - surface it
            log.exception("device login failed")
            self.state = "error"
            self.error = f"{exc.__class__.__name__}: {exc}"
        finally:
            if self._own_client:
                try:
                    await self.client.aclose()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #

    async def _authorize(self) -> bool:
        resp = await self.client.post(
            DEVICE_AUTHORIZE,
            content=f"client_id={CLIENT_ID}",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        resp.raise_for_status()
        data = resp.json()
        self._device_code = data.get("device_code")
        self.user_code = data.get("user_code")
        self.verification_url = (data.get("verification_uri_complete")
                                 or data.get("verification_uri"))
        if not (self._device_code and self.user_code):
            self.state = "error"
            self.error = f"unexpected authorize/device response: {data}"
            return False
        return True

    async def _poll(self) -> dict | None:
        """Poll /authenticate until approved, expired, or cancelled."""
        deadline = time.monotonic() + DEFAULT_EXPIRES
        interval = DEFAULT_INTERVAL
        body = ("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"
                f"&device_code={self._device_code}&client_id={CLIENT_ID}")
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                self.state = "cancelled"
                return None
            resp = await self.client.post(
                AUTHENTICATE,
                content=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"})
            if resp.status_code == 200:
                return resp.json()
            try:
                err = resp.json().get("error", "")
            except Exception:
                err = f"http_{resp.status_code}"
            if err == "authorization_pending":
                pass                                   # expected while waiting
            elif err == "slow_down":
                interval = min(interval + 5.0, MAX_INTERVAL)
            elif err == "expired_token":
                self.state = "expired"
                self.error = "the device code expired — start again"
                return None
            elif err == "access_denied":
                self.state = "error"
                self.error = "the login was denied in the browser"
                return None
            else:
                self.state = "error"
                self.error = f"authenticate failed: {err or resp.status_code}"
                return None
            # sleep in small bites so cancel is responsive
            waited = 0.0
            while waited < interval:
                if self._cancel.is_set():
                    self.state = "cancelled"
                    return None
                await asyncio.sleep(0.25)
                waited += 0.25
        self.state = "expired"
        self.error = "the device code expired — start again"
        return None

    async def _register(self, workos: dict) -> Account | None:
        """Trade the WorkOS tokens for a Cline account (same envelope as
        /auth/refresh, so one parser covers both)."""
        resp = await self.client.post(
            REGISTER,
            json={"accessToken": workos.get("access_token"),
                  "refreshToken": workos.get("refresh_token")},
            headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            self.state = "error"
            self.error = f"register failed: HTTP {resp.status_code}"
            return None
        parsed = _extract_token(resp.json())
        if not parsed:
            self.state = "error"
            self.error = "register returned an unrecognised envelope"
            return None
        user = (resp.json().get("data") or {}).get("userInfo") or {}
        account = Account(
            id=user.get("clineUserId") or "unknown",
            email=user.get("email") or "",
            access_token=("workos:" + parsed["access_token"]
                          if not parsed["access_token"].startswith("workos:")
                          else parsed["access_token"]),
            refresh_token=parsed.get("refresh_token") or "",
            expires_at=parsed.get("expires_at") or 0,
            source="device-login",
            state=AccountState.READY,
        )
        self.account_email = account.email or None
        return account

    def _persist(self, account: Account) -> None:
        """Snapshot to disk (survives restarts) — the pool upsert is done by
        the caller, which owns the pool lock."""
        workos_user = account.id
        fields = {
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "device-login",
            "account_id": account.id,
            "cline_user_id": account.id,
            "workos_user_id": workos_user if workos_user.startswith("user_") else "",
            "email": account.email or "",
            "token_prefix": "workos:",
            "expires_at_ms": str(account.expires_at or ""),
            "access_token": account.access_token,
            "refresh_token": account.refresh_token,
        }
        from .accounts_dir import snapshot_filename
        name = snapshot_filename(account.email or account.id, account.id, {})
        path = Path(self.accounts_dir) / name
        write_snapshot(path, fields)
        self.snapshot_path = path

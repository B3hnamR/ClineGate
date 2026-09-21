"""Detect the official Cline desktop app: installed, running, logged in.

Pure stdlib so the frozen exe gains no new dependencies. The GUI uses this to
show a status chip and to drive the import button with actionable guidance
("open Cline and sign in") instead of raw errors.

Paths are module constants for production use and injectable parameters for
tests.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path

HOME = Path.home()
CLINE_DIR = HOME / ".cline"
PROVIDERS_JSON = CLINE_DIR / "data" / "settings" / "providers.json"
GLOBAL_STATE_JSON = CLINE_DIR / "data" / "globalState.json"
HUB_LOCK = CLINE_DIR / "data" / "locks" / "hub" / "production.json"

CLINE_DOWNLOAD_URL = "https://cline.bot"

PROCESS_NAMES = ("cline-app.exe", "code-sidecar.exe")


def default_exe_candidates() -> list[Path]:
    """Known install locations for the Cline desktop app, best guess first."""
    from os import environ

    local = Path(environ.get("LOCALAPPDATA", ""))
    candidates: list[Path] = [
        local / "Cline" / "cline-app.exe",
        local / "Programs" / "Cline" / "cline-app.exe",
        local / "bot.cline.app" / "cline-app.exe",
    ]
    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        base = environ.get(var)
        if base:
            candidates.append(Path(base) / "Cline" / "cline-app.exe")
    return candidates


@dataclass
class ClineStatus:
    installed: bool = False
    running: bool = False
    logged_in: bool = False
    install_path: str = ""
    version: str = ""
    email: str = ""
    account_id: str = ""
    expires_at_ms: int = 0
    token_valid: bool = False
    error: str = ""
    # one human sentence for the status chip
    detail: str = "Cline not found"
    chip_kind: str = "off"        # off | installed | logged_in | running

    def as_chip(self) -> tuple[str, str]:
        return self.detail, self.chip_kind


def _registry_entry() -> tuple[str, str]:
    """(version, install_location) from the Windows uninstall registry."""
    try:
        import winreg
    except ImportError:
        return "", ""
    roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    subkeys = (r"Software\Microsoft\Windows\CurrentVersion\Uninstall",
               r"Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall")
    for root in roots:
        for sub in subkeys:
            try:
                top = winreg.OpenKey(root, sub)
            except OSError:
                continue
            with top:
                for i in count():
                    try:
                        name = winreg.EnumKey(top, i)
                    except OSError:
                        break
                    try:
                        with winreg.OpenKey(top, name) as k:
                            disp = winreg.QueryValueEx(k, "DisplayName")[0]
                            if "cline" not in str(disp).lower():
                                continue
                            try:
                                ver = str(winreg.QueryValueEx(k, "DisplayVersion")[0])
                            except OSError:
                                ver = ""
                            try:
                                loc = str(winreg.QueryValueEx(k, "InstallLocation")[0])
                            except OSError:
                                loc = ""
                            return ver, loc
                    except OSError:
                        continue
    return "", ""


# On a windowed (no-console) exe, spawning a console program like tasklist
# without this flag makes Windows allocate a fresh console window: the user
# sees a terminal flash open and close every status refresh.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def processes_running(names: tuple[str, ...] = PROCESS_NAMES) -> bool:
    """True if any of the named Cline processes is alive (tasklist)."""
    for name in names:
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {name}"],
                capture_output=True, text=True, timeout=10,
                creationflags=_NO_WINDOW,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if name.lower() in out.lower():
            return True
    return False


def jwt_claims(token: str) -> dict:
    """Claims of a workos-prefixed JWT, best effort ({} on any failure)."""
    raw = token.replace("workos:", "", 1)
    if raw.count(".") != 2:
        return {}
    try:
        payload = raw.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _read_logged_in(providers_path: Path) -> dict:
    """Best logged-in account info from providers.json ({} when none)."""
    if not providers_path.is_file():
        return {}
    try:
        data = json.loads(providers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Cline rewrites this file on login/logout; a torn read is transient
        return {}
    providers = data.get("providers") or {}
    best: dict = {}
    for slot, provider in providers.items():
        auth = (provider.get("settings") or {}).get("auth")
        if not isinstance(auth, dict) or not auth.get("accessToken"):
            continue
        claims = jwt_claims(auth["accessToken"])
        info = {
            "slot": slot,
            "account_id": auth.get("accountId") or claims.get("external_id", ""),
            "email": ((auth.get("metadata") or {}).get("userInfo") or {}).get("email", ""),
            "expires_at_ms": int(auth.get("expiresAt") or 0),
        }
        # prefer the main "cline" provider slot over pass/cloud variants
        if not best or slot == "cline":
            best = info
    return best


def detect_cline(exe_candidates: list[Path] | None = None,
                 providers_path: Path | None = None,
                 lock_path: Path | None = None,
                 check_processes: bool = True,
                 cline_dir: Path | None = None,
                 use_registry: bool = True) -> ClineStatus:
    """Assemble the full Cline status. All inputs injectable for tests."""
    if exe_candidates is None:
        exe_candidates = default_exe_candidates()
    if providers_path is None:
        providers_path = PROVIDERS_JSON
    if lock_path is None:
        lock_path = HUB_LOCK
    if cline_dir is None:
        cline_dir = CLINE_DIR

    status = ClineStatus()

    ver, reg_loc = _registry_entry() if use_registry else ("", "")
    status.version = ver

    exe_path = next((p for p in exe_candidates if p.is_file()), None)
    if exe_path is None and reg_loc:
        cand = Path(reg_loc) / "cline-app.exe"
        if cand.is_file():
            exe_path = cand
    if exe_path is not None:
        status.installed = True
        status.install_path = str(exe_path)
    elif cline_dir.is_dir():
        # config dir exists but no exe was found: still "installed" for login
        # purposes, but report no path rather than one we know does not exist
        status.installed = True

    if check_processes and status.installed:
        status.running = processes_running()

    logged = _read_logged_in(providers_path)
    if logged:
        status.logged_in = True
        status.account_id = logged.get("account_id", "")
        status.email = logged.get("email", "")
        status.expires_at_ms = logged.get("expires_at_ms", 0)
        status.token_valid = status.expires_at_ms > time.time() * 1000

    # chip text
    if not status.installed:
        status.detail = "Cline: not found"
        status.chip_kind = "off"
    elif not status.logged_in:
        status.detail = "Cline: installed, not logged in"
        status.chip_kind = "installed"
    elif status.running:
        who = status.email or status.account_id or "logged in"
        status.detail = f"Cline: running \u2014 {who}"
        status.chip_kind = "running"
    else:
        who = status.email or status.account_id or "logged in"
        status.detail = f"Cline: {who}"
        status.chip_kind = "logged_in"
    if status.logged_in and not status.token_valid:
        status.detail += " (token expired \u2014 gateway will refresh)"
    return status

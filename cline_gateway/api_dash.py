"""Dashboard-serving + the read/write admin surface the web UI needs.

Everything here is additive to the existing admin API. Settings writes go
through a whitelist of allowed keys — the config holds credentials and paths,
so free-form key editing is a footgun.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
import yaml

from .deps import admin_key, get_state
from .logbuffer import LogBuffer

log = logging.getLogger("cline_gateway.dash")

router = APIRouter(tags=["dashboard"])

def _dash_dir() -> Path:
    """Bundle-aware: PyInstaller unpacks data to sys._MEIPASS."""
    import sys
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "cline_gateway" / "dashboard"


DASH_DIR = _dash_dir()

# Settings the dashboard may edit (dotted config paths). Anything not listed
# is read-only from the UI — keeps credentials and upstream URLs out of it.
_EDITABLE = {
    "server.host": str, "server.port": int,
    "pool.strategy": str, "pool.min_balance_micro": int,
    "pool.refresh_lead_seconds": int, "pool.cooldown_seconds": int,
    "pool.max_in_flight_per_account": int, "pool.acquire_wait_seconds": float,
    "pool.balance_poll_seconds": int,
    "upstream.timeout_connect": float, "upstream.timeout_read": float,
    "upstream.max_attempts": int, "upstream.anthropic_cache_control": bool,
    "accounts.source": str, "accounts.dir": str, "accounts.pool_file": str,
    "logging.level": str, "logging.capture": bool,
    "models.default": str, "models.default_anthropic": str,
    "models.probe_unknown": bool,
    "update.enabled": bool, "update.interval_hours": float,
}


def _read_config_raw(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _config_path(state) -> Path | None:
    """Where the running config lives; None when not derivable."""
    root = getattr(state.cfg, "_root", None)
    if root:
        return root / "config.yaml"
    p = Path("config.yaml")
    return p if p.is_file() else None


@router.get("/dash", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    index = DASH_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse("dashboard not bundled", status_code=404)
    html = index.read_text(encoding="utf-8")
    # First-launch convenience: hand the page its own admin key so the login
    # screen does not gate the local tool (single-user, single-machine — the
    # Server tab displays the same key anyway). Loopback-only: never inject
    # when the server is reachable from other machines.
    try:
        cfg = get_state(request).cfg
        if (cfg.server.host or "").strip() in ("127.0.0.1", "localhost", "::1"):
            marker = 'sessionStorage.getItem("gw-admin-key") || ""'
            injected = ('sessionStorage.getItem("gw-admin-key") || '
                        + json.dumps(cfg.server.admin_key))
            html = html.replace(marker, injected, 1)
    except Exception:
        pass
    return HTMLResponse(html)


@router.get("/admin/dash/settings")
async def dash_settings(request: Request, key: str = Depends(admin_key)) -> dict:
    """Current settings + the raw editable values, plus server-side keys."""
    state = get_state(request)
    cfg = state.cfg
    cfg_path = _config_path(state)
    raw = _read_config_raw(cfg_path) if cfg_path else {}

    def _flat(path: str, default: Any = None) -> Any:
        cur: Any = raw
        for part in path.split("."):
            if not isinstance(cur, dict):
                return default
            cur = cur.get(part)
        return default if cur is None else cur

    return {
        "config_path": str(cfg_path) if cfg_path else "",
        "values": {k: _flat(k) for k in sorted(_EDITABLE)},
        "effective": {
            "server": cfg.server.model_dump(exclude={"client_keys", "admin_key"}),
            "pool": cfg.pool.model_dump(),
            "upstream": cfg.upstream.model_dump(exclude={"fingerprint"}),
            "accounts": cfg.accounts.model_dump(),
            "logging": cfg.logging.model_dump(),
            "models": cfg.models.model_dump(),
        },
        "admin_key": cfg.server.admin_key,
        "client_keys": [k.model_dump() for k in cfg.server.client_keys],
        "accounts_source": cfg.accounts.source,
    }


@router.put("/admin/dash/settings")
async def dash_settings_put(request: Request, body: dict = Body(...),
                            key: str = Depends(admin_key)) -> dict:
    """Write whitelisted settings keys back to config.yaml. A restart is
    required for most of them to take effect (pool/tokens are in memory)."""
    state = get_state(request)
    cfg_path = _config_path(state)
    if cfg_path is None:
        raise HTTPException(status_code=400, detail="no writable config.yaml found")

    values = body.get("values")
    if not isinstance(values, dict):
        raise HTTPException(status_code=400, detail="body must be {\"values\": {...}}")

    unknown = sorted(set(values) - set(_EDITABLE))
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"not editable: {', '.join(unknown)}")

    raw = _read_config_raw(cfg_path)
    changed: list[str] = []
    for dotted, val in values.items():
        cast = _EDITABLE[dotted]
        try:
            if cast is bool and isinstance(val, str):
                val = val.lower() in ("1", "true", "yes", "on")
            else:
                val = cast(val)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400,
                                detail=f"{dotted}: cannot cast {val!r}") from None
        cur = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        if cur.get(parts[-1]) != val:
            cur[parts[-1]] = val
            changed.append(dotted)

    if changed:
        cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False,
                                           allow_unicode=True),
                            encoding="utf-8")
    return {"changed": changed, "restart_required": bool(changed)}


@router.post("/admin/dash/import")
async def dash_import(request: Request, key: str = Depends(admin_key)) -> dict:
    """Snapshot every logged-in Cline account into the accounts folder."""
    state = get_state(request)
    try:
        from .gui import import_from_cline_config  # reuse the proven reader
    except Exception as exc:  # pragma: no cover - headless build
        raise HTTPException(status_code=500,
                            detail=f"import helper unavailable: {exc}") from exc
    try:
        out_dir = Path(state.cfg.accounts.dir)
        result = import_from_cline_config(output_dir=out_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.warning("cline import failed: %s", exc)
        raise HTTPException(status_code=500,
                            detail=f"import failed ({exc.__class__.__name__})") from exc
    return result


# ---- login with Cline (WorkOS device-code flow) --------------------------- #

def _device_login(request: Request):
    """The in-flight DeviceLogin for this app state, if any."""
    return getattr(get_state(request), "device_login", None)


@router.post("/admin/dash/auth/login/start")
async def login_start(request: Request, key: str = Depends(admin_key)) -> dict:
    """Begin a device-code login: returns the code + URL for the user."""
    state = get_state(request)
    if state.cfg.accounts.source != "accounts_dir":
        raise HTTPException(status_code=400,
                            detail="login writes account snapshots; set "
                                   "accounts.source=accounts_dir first")
    existing = _device_login(request)
    if existing and existing.state in ("awaiting_user", "registering"):
        return existing.status()          # one flow at a time; idempotent start
    from .device_auth import DeviceLogin

    async def _add_and_probe(account):
        """New account: join the pool, then settle balance/paid-lane/plan now —
        a login that replaced an exhausted account must not read 'available'."""
        await state.pool.upsert(account)
        try:
            await state.tokens.check_account(account)
        except Exception:
            log.warning("post-login account check failed for %s", account.id)

    login = DeviceLogin(Path(state.cfg.accounts.dir),
                        on_account=_add_and_probe)
    state.device_login = login
    task = asyncio.create_task(login.run(), name="device-login")
    login._task = task
    # wait (briefly) for the code so the UI can show it immediately
    try:
        await asyncio.wait_for(login._ready.wait(), timeout=20)
    except asyncio.TimeoutError:
        pass
    return login.status()


@router.get("/admin/dash/auth/login/status")
async def login_status(request: Request, key: str = Depends(admin_key)) -> dict:
    login = _device_login(request)
    if login is None:
        return {"state": "idle"}
    status = login.status()
    # a finished login has already been persisted + pooled inside run();
    # surface the resulting account id so the UI can refresh the table
    if login.state == "done" and login.snapshot_path is not None:
        status["snapshot"] = login.snapshot_path.name
    return status


@router.post("/admin/dash/auth/login/cancel")
async def login_cancel(request: Request, key: str = Depends(admin_key)) -> dict:
    login = _device_login(request)
    if login is None:
        return {"state": "idle"}
    login.cancel()
    task = getattr(login, "_task", None)
    if task is not None and not task.done():
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
    return login.status()


@router.get("/admin/dash/cline-status")
async def dash_cline_status(request: Request, key: str = Depends(admin_key)) -> dict:
    """Installed/running/logged-in status of the desktop Cline app."""
    get_state(request)
    from .cline_detect import detect_cline
    status = detect_cline()
    chip, kind = status.as_chip()
    return {
        "installed": status.installed,
        "running": status.running,
        "logged_in": status.logged_in,
        "install_path": status.install_path,
        "version": status.version,
        "email": status.email,
        "account_id": status.account_id,
        "expires_at_ms": status.expires_at_ms,
        "token_valid": status.token_valid,
        "detail": chip,
        "chip_kind": kind,
    }


@router.post("/admin/dash/shutdown")
async def dash_shutdown(request: Request, key: str = Depends(admin_key)) -> dict:
    """Stop the tool: answer, then exit the process a moment later.

    There is no other way to stop the exe build (no tray, no stop path, and
    closing the WebView2 window leaves the process — and its port and
    single-instance mutex — alive on non-daemon WebView2 threads). A hard
    exit is deliberate: uvicorn's should_exit would free the port but leave
    the windowless process running, which is exactly the bug this fixes.
    """
    get_state(request)
    threading.Timer(1.0, lambda: os._exit(0)).start()
    return {"stopping": True,
            "note": "the gateway process is exiting; you can close this window"}


@router.get("/admin/dash/update/status")
async def update_status(request: Request, key: str = Depends(admin_key)) -> dict:
    """Cached update check result (never hits GitHub from this route)."""
    state = get_state(request)
    updater = getattr(state, "updater", None)
    if updater is None:
        return {"update_available": False, "checked_at": None}
    return updater.status


@router.post("/admin/dash/update/check")
async def update_check(request: Request, key: str = Depends(admin_key)) -> dict:
    """Force a fresh check now (one GitHub round-trip)."""
    state = get_state(request)
    updater = getattr(state, "updater", None)
    if updater is None:
        raise HTTPException(status_code=503, detail="updater not running")
    return await updater.check()


@router.post("/admin/dash/update/download")
async def update_download(request: Request, key: str = Depends(admin_key)) -> dict:
    """Download the new exe next to the running one (exe builds only)."""
    state = get_state(request)
    updater = getattr(state, "updater", None)
    status = updater.status if updater else {}
    url = status.get("download_url")
    if not (status.get("update_available") and url):
        raise HTTPException(status_code=400, detail="no update available")
    from . import updater as updater_mod
    try:
        path = await updater_mod.download_update(url)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"download failed ({exc.__class__.__name__})") \
            from exc
    return {"downloaded": str(path), "size": path.stat().st_size}


@router.post("/admin/dash/update/apply")
async def update_apply(request: Request, key: str = Depends(admin_key)) -> dict:
    """Swap in the downloaded exe and relaunch (exe builds only)."""
    get_state(request)
    from . import updater as updater_mod
    try:
        updater_mod.apply_update(updater_mod.exe_dir() / updater_mod.NEW_EXE_NAME)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # the swapper waits for this process to die; answer first, then exit
    threading.Timer(1.0, lambda: os._exit(0)).start()
    return {"applying": True, "note": "the gateway exits, swaps the exe, and "
                                      "relaunches itself"}


@router.get("/admin/dash/logs")
async def dash_logs(request: Request, limit: int = 300, level: str = "",
                    key: str = Depends(admin_key)) -> dict:
    """Recent log lines from the in-process ring buffer."""
    state = get_state(request)
    buf: LogBuffer | None = getattr(state, "log_buffer", None)
    if buf is None:
        return {"records": [], "note": "log buffer not attached"}
    return {"records": buf.tail(limit=max(1, min(limit, 1000)), level=level)}


@router.get("/admin/dash/server")
async def dash_server(request: Request, key: str = Depends(admin_key)) -> dict:
    """Server + config info the dashboard header and Settings tab need."""
    state = get_state(request)
    cfg = state.cfg
    cfg_path = _config_path(state)
    return {
        "host": cfg.server.host, "port": cfg.server.port,
        "config_path": str(cfg_path) if cfg_path else "",
        "accounts_dir": cfg.accounts.dir,
        "accounts_source": cfg.accounts.source,
        "strategy": cfg.pool.strategy,
        "version": __import__("cline_gateway").__version__,
        "endpoints": {
            "openai": "/v1/chat/completions",
            "anthropic": "/v1/messages",
            "models": "/v1/models",
        },
        "admin_key": cfg.server.admin_key,
        "client_keys": [k.model_dump() for k in cfg.server.client_keys],
        "cwd": os.getcwd(),
        "started_hint": "the gateway process; restart applies settings edits",
    }

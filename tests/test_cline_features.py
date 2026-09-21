"""Tests for the Cline detection + import + first-launch bootstrap features."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path


from cline_gateway.accounts_dir import load_from_accounts_dir, snapshot_filename
from cline_gateway.cline_detect import detect_cline, jwt_claims, processes_running


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(obj).encode()).rstrip(b"=").decode()


def _jwt(exp: int, sub: str = "user_123") -> str:
    return f"workos:{_b64({'alg': 'RS256'})}.{_b64({'exp': exp, 'sub': sub})}.sig"


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def _providers_file(tmp_path, providers: dict) -> Path:
    p = tmp_path / "providers.json"
    p.write_text(json.dumps({"providers": providers}), encoding="utf-8")
    return p


def _provider(token: str, email: str = "u@x.com") -> dict:
    return {"settings": {"auth": {
        "accessToken": token,
        "refreshToken": "R",
        "expiresAt": int(time.time() * 1000) + 3_600_000,
        "accountId": "usr-1",
        "metadata": {"userInfo": {"email": email}},
    }}}


def test_detect_not_installed(tmp_path):
    status = detect_cline(
        exe_candidates=[],                    # no exe anywhere
        providers_path=tmp_path / "none.json",
        lock_path=tmp_path / "none.lock",
        check_processes=False,
        cline_dir=tmp_path / "no-cline-dir",  # ignore the real ~/.cline
        use_registry=False)
    assert status.installed is False
    assert status.logged_in is False
    assert status.chip_kind == "off"


def test_detect_installed_and_logged_in(tmp_path):
    exe = tmp_path / "cline-app.exe"
    exe.write_text("", encoding="utf-8")
    providers = _providers_file(tmp_path, {"cline": _provider(_jwt(2_000_000_000))})

    status = detect_cline(
        exe_candidates=[exe], providers_path=providers,
        lock_path=tmp_path / "none.lock", check_processes=False)

    assert status.installed is True
    assert status.logged_in is True
    assert status.email == "u@x.com"
    assert status.account_id == "usr-1"
    assert status.token_valid is True
    assert status.chip_kind == "logged_in"    # not running (no processes check)


def test_detect_installed_not_logged_in(tmp_path):
    exe = tmp_path / "cline-app.exe"
    exe.write_text("", encoding="utf-8")

    status = detect_cline(
        exe_candidates=[exe],
        providers_path=_providers_file(tmp_path, {}),
        lock_path=tmp_path / "none.lock", check_processes=False)

    assert status.installed is True
    assert status.logged_in is False
    assert status.chip_kind == "installed"


def test_detect_torn_providers_read_is_not_logged_in(tmp_path):
    exe = tmp_path / "cline-app.exe"
    exe.write_text("", encoding="utf-8")
    torn = tmp_path / "providers.json"
    torn.write_text('{"providers": {"cline": {"set', encoding="utf-8")  # mid-write

    status = detect_cline(
        exe_candidates=[exe], providers_path=torn,
        lock_path=tmp_path / "none.lock", check_processes=False)
    assert status.logged_in is False          # graceful, not an exception


def test_detect_expired_token_flagged(tmp_path):
    exe = tmp_path / "cline-app.exe"
    exe.write_text("", encoding="utf-8")
    prov = {"settings": {"auth": {
        "accessToken": _jwt(1_000_000),       # long past
        "expiresAt": 1000,
        "accountId": "usr-1",
        "metadata": {"userInfo": {"email": "u@x.com"}},
    }}}
    status = detect_cline(
        exe_candidates=[exe],
        providers_path=_providers_file(tmp_path, {"cline": prov}),
        lock_path=tmp_path / "none.lock", check_processes=False,
        cline_dir=tmp_path / "no-cline-dir", use_registry=False)
    assert status.token_valid is False
    assert "expired" in status.detail


def test_jwt_claims_parses_workos_prefixed_token():
    claims = jwt_claims(_jwt(1234, "sub_1"))
    assert claims["exp"] == 1234
    assert claims["sub"] == "sub_1"
    assert jwt_claims("garbage") == {}
    assert jwt_claims("") == {}


def test_processes_running_never_raises():
    # tasklist may be unavailable in CI; must return a bool either way
    assert isinstance(processes_running(), bool)


def test_processes_running_suppresses_console_window(monkeypatch):
    """Windowed-exe guard: tasklist must run with CREATE_NO_WINDOW, or the
    user sees a console flash on every Cline status refresh."""
    import subprocess as sp

    from cline_gateway import cline_detect

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        captured["cmd"] = cmd
        class R:
            stdout = ""
        return R()

    monkeypatch.setattr(sp, "run", fake_run)
    cline_detect.processes_running()
    flags = captured.get("creationflags", 0)
    no_window = getattr(sp, "CREATE_NO_WINDOW", 0)
    assert no_window & flags == no_window or flags == 0, \
        "tasklist spawned without CREATE_NO_WINDOW -> console flash"


# --------------------------------------------------------------------------- #
# collision-safe snapshot naming + import
# --------------------------------------------------------------------------- #


def test_snapshot_filename_collisions_get_account_suffix():
    used: dict[str, str] = {}
    first = snapshot_filename("same@x.com", "usr-A", used)
    used[first] = "usr-A"
    second = snapshot_filename("same@x.com", "usr-B", used)

    assert first == "same@x.com.txt"
    assert second != first                    # no overwrite
    assert second.startswith("same@x.com.")
    assert "usr-B" in second


def test_snapshot_filename_same_account_keeps_name():
    used: dict[str, str] = {}
    first = snapshot_filename("same@x.com", "usr-A", used)
    used[first] = "usr-A"
    again = snapshot_filename("same@x.com", "usr-A", used)
    assert again == first                     # re-import stays stable


def test_snapshot_filename_avoids_existing_suffix_collision():
    used = {"same@x.com.txt": "usr-A", "same@x.com.usr-B.txt": "usr-C"}
    name = snapshot_filename("same@x.com", "usr-B", used)
    assert name == "same@x.com.usr-B.2.txt"


def test_import_from_cline_config(tmp_path, monkeypatch):
    from cline_gateway import gui as gui_mod

    token = _jwt(2_000_000_000)
    providers = {
        "cline": _provider(token, "main@x.com"),
        "cline-pass": _provider(token, "main@x.com"),   # same email, other slot
        "empty": {"settings": {"auth": {}}},
    }
    prov_file = tmp_path / "providers.json"
    prov_file.write_text(json.dumps({"providers": providers}), encoding="utf-8")
    global_file = tmp_path / "globalState.json"
    global_file.write_text(json.dumps({"actModeApiProvider": "cline"}),
                           encoding="utf-8")
    accounts = tmp_path / "accounts"
    accounts.mkdir()

    monkeypatch.setattr(gui_mod, "CLINE_PROVIDERS", prov_file)
    monkeypatch.setattr(gui_mod, "CLINE_GLOBAL", global_file)
    monkeypatch.setattr(gui_mod, "find_accounts_dir", lambda: accounts)

    result = gui_mod.import_from_cline_config()

    # two distinct slots, same account id -> one snapshot (dedupe, no overwrite)
    assert result["written"] == 1
    assert result["new_ids"] == ["usr-1"]
    assert result["emails"] == ["main@x.com"]

    loaded = load_from_accounts_dir(accounts)
    assert len(loaded) == 1
    assert loaded[0].id == "usr-1"
    assert loaded[0].access_token == token


def test_import_writes_to_explicit_configured_directory(tmp_path, monkeypatch):
    from cline_gateway import gui as gui_mod

    configured = tmp_path / "configured"
    discovered = tmp_path / "discovered"
    configured.mkdir()
    discovered.mkdir()
    prov_file = _providers_file(tmp_path, {"cline": _provider(_jwt(2_000_000_000))})
    monkeypatch.setattr(gui_mod, "CLINE_PROVIDERS", prov_file)
    monkeypatch.setattr(gui_mod, "CLINE_GLOBAL", tmp_path / "none.json")
    monkeypatch.setattr(gui_mod, "find_accounts_dir", lambda: discovered)

    result = gui_mod.import_from_cline_config(output_dir=configured)

    assert result["written"] == 1
    assert list(configured.glob("*.txt"))
    assert not list(discovered.glob("*.txt"))


def test_import_retries_torn_providers_read(tmp_path, monkeypatch):
    from cline_gateway import gui as gui_mod

    prov_file = tmp_path / "providers.json"
    good = json.dumps({"providers": {"cline": _provider(_jwt(2_000_000_000))}})
    prov_file.write_text(good[:-8], encoding="utf-8")     # torn write

    accounts = tmp_path / "accounts"
    accounts.mkdir()
    monkeypatch.setattr(gui_mod, "CLINE_PROVIDERS", prov_file)
    monkeypatch.setattr(gui_mod, "CLINE_GLOBAL", tmp_path / "none.json")
    monkeypatch.setattr(gui_mod, "find_accounts_dir", lambda: accounts)

    # heal the file right after the first failed parse
    orig_read = Path.read_text

    def healing_read(self, *a, **kw):
        text = orig_read(self, *a, **kw)
        if self == prov_file and not text.endswith("}"):
            prov_file.write_text(good, encoding="utf-8")
            return good
        return text

    monkeypatch.setattr(Path, "read_text", healing_read)
    result = gui_mod.import_from_cline_config()
    assert result["written"] == 1


# --------------------------------------------------------------------------- #
# first-launch bootstrap
# --------------------------------------------------------------------------- #


def test_fresh_config_gets_local_accounts_dir(tmp_path, monkeypatch):
    from cline_gateway import gui as gui_mod

    # nothing discoverable anywhere: fresh-folder install -> local ./accounts
    monkeypatch.setattr(gui_mod, "base_dir", lambda: tmp_path)
    monkeypatch.setattr(gui_mod, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(gui_mod, "find_accounts_dir", lambda: tmp_path / "accounts")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(gui_mod.DEFAULT_CONFIG_YAML, encoding="utf-8")

    gui_mod._normalize_config(cfg)

    import yaml
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["accounts"]["dir"] == str(tmp_path / "accounts")


def test_normalize_prefers_populated_upstream_dir(tmp_path, monkeypatch):
    """Repo layout: exe in app/, real snapshots in ../accounts (parent)."""
    from cline_gateway import gui as gui_mod

    app = tmp_path / "app"
    app.mkdir()
    populated = tmp_path / "accounts"
    populated.mkdir()
    (populated / "a.txt").write_text("account_id: usr-1\naccess_token: t\n",
                                     encoding="utf-8")

    monkeypatch.setattr(gui_mod, "base_dir", lambda: app)
    monkeypatch.setattr(gui_mod, "app_dir", lambda: app)
    monkeypatch.setattr(gui_mod, "find_accounts_dir", lambda: populated)

    cfg = app / "config.yaml"
    cfg.write_text(gui_mod.DEFAULT_CONFIG_YAML, encoding="utf-8")  # ./accounts
    gui_mod._normalize_config(cfg)

    import yaml
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    # the populated up-tree folder wins over creating an empty local one
    assert data["accounts"]["dir"] == str(populated)
    assert not (app / "accounts").exists()      # no empty sibling created


def test_normalize_leaves_populated_existing_dir_alone(tmp_path, monkeypatch):
    from cline_gateway import gui as gui_mod

    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.setattr(gui_mod, "base_dir", lambda: app)
    cfg = app / "config.yaml"
    cfg.write_text("accounts:\n  dir: \"../accounts\"\n", encoding="utf-8")

    # ../accounts from app/ exists and holds a snapshot: leave it alone
    populated = tmp_path / "accounts"
    populated.mkdir()
    (populated / "a.txt").write_text("account_id: usr-1\naccess_token: t\n",
                                     encoding="utf-8")

    gui_mod._normalize_config(cfg)
    text = cfg.read_text(encoding="utf-8")
    assert '"../accounts"' in text          # untouched


def test_find_accounts_dir_prefers_populated(tmp_path, monkeypatch):
    from cline_gateway import gui as gui_mod

    app = tmp_path / "app"
    app.mkdir()
    (app / "accounts").mkdir()              # exists but empty
    populated = tmp_path / "accounts"       # parent level, holds snapshots
    populated.mkdir()
    (populated / "a.txt").write_text("account_id: usr-1\naccess_token: t\n",
                                     encoding="utf-8")

    monkeypatch.setattr(gui_mod, "base_dir", lambda: app)
    monkeypatch.setattr(gui_mod, "app_dir", lambda: app)

    found = gui_mod.find_accounts_dir()
    assert found == populated


def test_default_config_yaml_is_valid_and_local():
    import yaml
    data = yaml.safe_load(__import__("cline_gateway.gui",
                                     fromlist=["DEFAULT_CONFIG_YAML"])
                          .DEFAULT_CONFIG_YAML)
    assert data["accounts"]["dir"] == "./accounts"
    assert data["accounts"]["source"] == "accounts_dir"
    assert data["server"]["port"] == 8787
    assert data["models"]["auto_free_fallback"]["enabled"] is False
    assert data["models"]["auto_free_fallback"]["chain"]


def test_fallback_chain_editor_parses_order_and_rejects_invalid_values():
    from cline_gateway.gui import App

    assert App._fallback_chain_from_text(" model-a\n\nmodel-b ") == ["model-a", "model-b"]
    import pytest
    with pytest.raises(ValueError, match="at least one"):
        App._fallback_chain_from_text("\n")
    with pytest.raises(ValueError, match="spaces"):
        App._fallback_chain_from_text("model a")
    with pytest.raises(ValueError, match="duplicate"):
        App._fallback_chain_from_text("model-a\nmodel-a")


def test_render_preserves_selection_across_rebuilds():
    """Poll re-renders every 2.5 s; the user's selected row (and its detail
    strip) must survive instead of flashing away after 1-2 seconds."""
    from cline_gateway.gui import App
    try:
        app = App(autostart=False)
    except Exception as exc:
        import pytest
        pytest.skip(f"no display: {exc}")

    try:
        app.update_idletasks()

        def entry(i):
            return {"email": f"u{i}@x.com", "id": f"usr-{i}",
                    "state": "ready", "paid_exhausted": (i % 2 == 0),
                    "expires_at": 9999999999999, "in_flight": 0,
                    "notes": {"file": f"u{i}.txt"}}

        rows = [entry(i) for i in range(3)]
        app._render_accounts({"accounts": 3, "detail": rows})

        # user clicks the second row
        app.tree.selection_set(app.tree.get_children()[1])
        app._show_detail()
        assert app.selected_id() == "usr-1"
        assert "usr-1" in app.detail_lbl.cget("text")

        # next poll rebuild (with a balance update changing nothing structural)
        rows[0]["balance_micro"] = 100
        app._render({"accounts": 3, "detail": rows}, {}, None)

        assert app.selected_id() == "usr-1"          # selection survives
        assert "usr-1" in app.detail_lbl.cget("text")  # detail stays too

        # typing a filter that hides the selection legitimately clears it
        app.acct_filter.set("usr-2")
        assert app.selected_id() is None
    finally:
        app.destroy()

"""Configuration: YAML file + environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


class ClientKey(BaseModel):
    key: str
    name: str = "client"
    rpm: int = 0                 # requests-per-minute limit; 0 = unlimited


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    client_keys: list[ClientKey] = Field(default_factory=list)
    require_client_key: bool = True
    admin_key: str = "gw-admin-change-me"


class Fingerprint(BaseModel):
    user_agent: str = (
        "Cline/0.0.28 ai-sdk/openai-compatible/3.0.37 "
        "ai-sdk/provider-utils/5.0.30 runtime/bun/1.3.13"
    )
    client_version: str = "0.0.28"
    core_version: str = "0.0.83"
    platform: str = "Cline Desktop"


class UpstreamConfig(BaseModel):
    base_url: str = "https://api.cline.bot/api/v1"
    chat_path: str = "/chat/completions"
    timeout_connect: float = 15.0
    timeout_read: float = 600.0
    max_attempts: int = 3
    # Whether the anthropic body variant carries `cache_control`. The capture
    # always included it; flip to false to test whether it is optional.
    anthropic_cache_control: bool = True
    fingerprint: Fingerprint = Field(default_factory=Fingerprint)

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + self.chat_path


class PoolConfig(BaseModel):
    strategy: str = "least_in_flight"
    min_balance_micro: int = 0
    refresh_lead_seconds: int = 600
    cooldown_seconds: int = 30
    max_in_flight_per_account: int = 4
    # how long acquire() waits for an in-flight slot when every eligible
    # account is at max_in_flight_per_account (0 = fail immediately)
    acquire_wait_seconds: float = 5.0
    balance_poll_seconds: int = 300


class AccountsConfig(BaseModel):
    source: str = "providers_json"          # providers_json | pool_file | accounts_dir
    providers_json: str = "~/.cline/data/settings/providers.json"
    pool_file: str = "./pool.json"
    dir: str = "../accounts"                # for source=accounts_dir"


class AutoFreeFallbackConfig(BaseModel):
    enabled: bool = False
    chain: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    aliases: dict[str, str] = Field(default_factory=dict)
    default: str = "cline-free/deepseek-v4.1-flash"
    default_anthropic: str = "anthropic/claude-opus-5"
    default_max_tokens: int = 32000
    probe_unknown: bool = True
    auto_free_fallback: AutoFreeFallbackConfig = Field(
        default_factory=AutoFreeFallbackConfig)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    capture: bool = True
    capture_dir: str = "./logs"


class StoreConfig(BaseModel):
    sqlite_path: str = "./gateway.db"


class UpdateConfig(BaseModel):
    enabled: bool = True
    repo: str = "B3hnamR/ClineGate"     # GitHub releases feed
    interval_hours: float = 6.0


class Config(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    accounts: AccountsConfig = Field(default_factory=AccountsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    update: UpdateConfig = Field(default_factory=UpdateConfig)

    # resolved absolute paths (filled by load())
    _root: Path | None = None
    _config_file: Path | None = None

    def resolved(self, base: Path | None = None) -> "Config":
        """Expand ~ and resolve relative paths against the config file's folder.

        Without the base, a relative path like ../accounts would resolve against
        the process cwd - which breaks a copied .exe launched from elsewhere.
        """
        def fix(value: str) -> str:
            expanded = _expand(value)
            path = Path(expanded)
            if path.is_absolute() or base is None:
                return expanded
            return str((base / path).resolve())

        self.accounts.providers_json = fix(self.accounts.providers_json)
        self.accounts.dir = fix(self.accounts.dir)
        self.accounts.pool_file = fix(self.accounts.pool_file)
        self.logging.capture_dir = fix(self.logging.capture_dir)
        self.store.sqlite_path = fix(self.store.sqlite_path)
        # the folder is the base for relative paths; `_config_file` (set by
        # load_config) is the exact file the dashboard Settings tab writes back
        # to — cwd-relative fallbacks edited the wrong config.yaml when the exe
        # was launched from elsewhere
        self._root = base
        return self

    def ensure_dirs(self) -> None:
        Path(self.logging.capture_dir).mkdir(parents=True, exist_ok=True)
        Path(self.store.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        if self.accounts.source == "accounts_dir":
            # a standalone install must work with an empty folder that exists,
            # rather than failing pool loads until the first import
            Path(self.accounts.dir).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config.yaml (if present) merged over defaults, then env overrides."""
    data: dict[str, Any] = {}

    candidates = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("CLINE_GATEWAY_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(Path(__file__).resolve().parent.parent / "config.yaml")

    base: Path | None = None
    config_file: Path | None = None
    for c in candidates:
        if c and c.is_file():
            with open(c, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            data = _deep_merge(data, loaded)
            config_file = Path(c).resolve()
            base = config_file.parent
            break

    cfg = Config(**data)
    cfg._config_file = config_file

    # environment overrides (CLINE_GATEWAY_<SECTION>__<FIELD>)
    env = {k: v for k, v in os.environ.items() if k.startswith("CLINE_GATEWAY_")}
    for key, val in env.items():
        parts = key[len("CLINE_GATEWAY_"):].lower().split("__")
        if len(parts) != 2:
            continue
        section, field = parts
        if hasattr(cfg, section) and hasattr(getattr(cfg, section), field):
            target = getattr(cfg, section)
            current = getattr(target, field)
            try:
                if isinstance(current, bool):
                    setattr(target, field, val.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(target, field, int(val))
                elif isinstance(current, float):
                    setattr(target, field, float(val))
                else:
                    setattr(target, field, val)
            except ValueError:
                pass

    return cfg.resolved(base)

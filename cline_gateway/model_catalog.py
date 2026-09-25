"""Fetch Cline's current curated model list without blocking app startup."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .registry import Registry

log = logging.getLogger("cline_gateway.model_catalog")

CATALOG_PATH = "/ai/cline/recommended-models"
REFRESH_SECONDS = 300.0
RETRY_SECONDS = 30.0


class ModelCatalog:
    """Cached public feed; a failed refresh keeps the last usable catalogue."""

    def __init__(self, base_url: str, registry: Registry, *,
                 client: httpx.AsyncClient | None = None,
                 refresh_seconds: float = REFRESH_SECONDS,
                 retry_seconds: float = RETRY_SECONDS) -> None:
        self.url = base_url.rstrip("/") + CATALOG_PATH
        self.registry = registry
        self.refresh_seconds = refresh_seconds
        self.retry_seconds = retry_seconds
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=3.0),
            headers={"accept": "application/json", "user-agent": "ClineGate-models"},
        )
        self._owns_client = client is None
        self._lock = asyncio.Lock()
        self._next_check = 0.0
        self._last_attempt_at: float | None = None
        self._updated_at: float | None = None
        self._error: str | None = None

    @property
    def status(self) -> dict:
        return {
            "source": "live" if self._updated_at is not None else "snapshot",
            "updated_at": self._updated_at,
            "last_attempt_at": self._last_attempt_at,
            "error": self._error,
        }

    async def refresh_if_due(self, *, force: bool = False) -> dict:
        if not force and time.monotonic() < self._next_check:
            return self.status
        async with self._lock:
            if not force and time.monotonic() < self._next_check:
                return self.status
            self._last_attempt_at = time.time()
            try:
                response = await self._client.get(self.url)
                response.raise_for_status()
                self.registry.use_live_catalogue(response.json())
            except Exception as exc:
                # Do not replace the cache with an empty or partial feed.
                self._error = exc.__class__.__name__
                self._next_check = time.monotonic() + self.retry_seconds
                log.warning("model catalogue refresh failed: %s", self._error)
            else:
                self._updated_at = time.time()
                self._error = None
                self._next_check = time.monotonic() + self.refresh_seconds
            return self.status

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

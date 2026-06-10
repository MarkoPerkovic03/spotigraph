"""Deezer API client for SpotiGraph.

Provides a free audio-feature proxy to replace Spotify's deactivated
audio-features endpoint. Deezer's public API needs no API key.

Signals
-------
bpm       → tempo (often 0 / missing → treated as None)
gain      → loudness in dB (~ -20..0; more reliably present than bpm)

Endpoints used
--------------
GET /search?q=...      → resolve (artist, track) → Deezer track id
GET /track/{id}        → bpm + gain (NOT returned by /search)

This is an *energy* proxy (tempo + loudness), not full valence/mood.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

DEEZER_BASE = "https://api.deezer.com"


class DeezerSettings(BaseSettings):
    deezer_enabled: bool = True
    deezer_timeout: float = 10.0

    model_config = {"env_file": ".env", "extra": "ignore"}


def _normalize(text: str) -> str:
    """Lowercase, strip bracketed suffixes (feat./remix/edit) and punctuation."""
    text = text.lower()
    text = re.sub(r"[\(\[].*?[\)\]]", " ", text)   # drop "(feat. ...)", "[remix]"
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


class DeezerClient:
    def __init__(self) -> None:
        settings = DeezerSettings()
        self._enabled = settings.deezer_enabled
        self._timeout = settings.deezer_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "DeezerClient":
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("DeezerClient must be used as async context manager")
        return self._client

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    async def _get(self, path: str, **params) -> Optional[dict]:
        if not self._enabled:
            return None
        try:
            resp = await self.client.get(f"{DEEZER_BASE}{path}", params=params)
            if resp.status_code != 200:
                logger.warning("Deezer %s returned %d", path, resp.status_code)
                return None
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                logger.warning("Deezer error on %s: %s", path, data.get("error"))
                return None
            return data
        except Exception as exc:
            logger.warning("Deezer request failed (%s): %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Audio proxy: bpm + loudness via search → track lookup
    # ------------------------------------------------------------------

    async def get_audio_proxy(
        self, artist_name: str, track_name: str
    ) -> Optional[dict]:
        """
        Resolve a track on Deezer and return an audio proxy.

        Returns {"bpm": float|None, "loudness": float|None, "rank": int|None}
        or None if no confident match is found.
        """
        if not artist_name or not track_name:
            return None

        # 1. Search → first hit's Deezer track id
        search = await self._get("/search", q=f"{artist_name} {track_name}", limit=1)
        hits = (search or {}).get("data") or []
        if not hits:
            logger.info("Deezer: no match for '%s — %s'", artist_name, track_name)
            return None
        hit = hits[0]

        # Sanity check: title should roughly match (guards against wrong song)
        want = _normalize(track_name)
        got = _normalize(str(hit.get("title", "")))
        if want and got and want not in got and got not in want:
            logger.info(
                "Deezer: low-confidence match for '%s' → '%s' (skipped)",
                track_name, hit.get("title"),
            )
            return None

        track_id = hit.get("id")
        if not track_id:
            return None

        # 2. Track detail → bpm + gain (not present in search results)
        detail = await self._get(f"/track/{track_id}")
        if not detail:
            return None

        bpm = detail.get("bpm")
        bpm = float(bpm) if bpm else None          # 0 / missing → None
        gain = detail.get("gain")
        loudness = float(gain) if gain is not None else None

        return {"bpm": bpm, "loudness": loudness, "rank": detail.get("rank")}

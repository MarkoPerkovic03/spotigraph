"""Last.fm API client for SpotiGraph.

Replaces Spotify's restricted endpoints (audio features, genres, related artists)
with free Last.fm data.

Endpoints used
--------------
artist.getTopTags   → genre + mood tags for an artist
artist.getSimilar   → similar artists (replaces Spotify related-artists)
track.getTopTags    → track-specific tags (more precise than artist tags)

Tag classification
------------------
Last.fm tags are free-form strings contributed by users. We split them into:
  - genre_tags  : music genre descriptors (e.g. "balkan trap", "hip hop")
  - mood_tags   : emotional/atmospheric descriptors (e.g. "melancholic", "energetic")
  - era_tags    : decade/era descriptors (e.g. "90s", "old school")

Everything else is kept as a genre tag (most tags are genre-like).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

# Tags that are clearly mood/emotional descriptors
_MOOD_TAGS = {
    "melancholic", "melancholy", "sad", "happy", "energetic", "chill",
    "relaxing", "relaxed", "dark", "upbeat", "aggressive", "peaceful",
    "romantic", "emotional", "intense", "dreamy", "nostalgic", "angry",
    "euphoric", "somber", "uplifting", "mellow", "introspective",
    "cathartic", "tense", "calm", "epic", "haunting", "beautiful",
    "fun", "party", "party music", "feel good", "feelgood",
}

# Tags that indicate era/decade
_ERA_TAGS = {
    "60s", "70s", "80s", "90s", "2000s", "2010s", "2020s",
    "old school", "oldschool", "classic", "retro", "vintage", "throwback",
}

# Noise tags to skip entirely
_SKIP_TAGS = {
    "seen live", "under 2000 listeners", "all", "favorites", "favourite",
    "favorites", "love", "awesome", "good", "great", "best", "cool",
    "music", "songs", "tracks", "",
}

# Generic genre tags — only kept if no specific tags exist
_GENERIC_GENRE_TAGS = {
    "rap", "hip-hop", "hip hop", "hiphop", "pop", "rock", "electronic",
    "r&b", "rnb", "soul", "jazz", "indie", "alternative", "metal",
    "dance", "house", "techno", "classical", "country", "folk",
}


class LastFmSettings(BaseSettings):
    lastfm_api_key: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


class LastFmClient:
    def __init__(self) -> None:
        settings = LastFmSettings()
        self._api_key = settings.lastfm_api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "LastFmClient":
        self._client = httpx.AsyncClient(timeout=10.0)
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LastFmClient must be used as async context manager")
        return self._client

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    async def _get(self, method: str, **params) -> Optional[dict]:
        if not self._api_key:
            logger.warning("LASTFM_API_KEY not set")
            return None
        try:
            resp = await self.client.get(
                LASTFM_BASE,
                params={"method": method, "api_key": self._api_key, "format": "json", **params},
            )
            if resp.status_code != 200:
                logger.warning("Last.fm %s returned %d", method, resp.status_code)
                return None
            data = resp.json()
            if "error" in data:
                logger.warning("Last.fm error %s: %s", data.get("error"), data.get("message"))
                return None
            return data
        except Exception as exc:
            logger.warning("Last.fm request failed (%s): %s", method, exc)
            return None

    # ------------------------------------------------------------------
    # Artist tags  →  genre / mood / era labels
    # ------------------------------------------------------------------

    async def get_artist_tags(self, artist_name: str) -> dict[str, list[str]]:
        """
        Return classified tags for an artist.

        Returns: {"genre": [...], "mood": [...], "era": [...]}
        """
        data = await self._get("artist.getTopTags", artist=artist_name)
        if not data:
            return {"genre": [], "mood": [], "era": []}

        raw_tags = data.get("toptags", {}).get("tag", [])
        if isinstance(raw_tags, dict):   # single tag comes as dict, not list
            raw_tags = [raw_tags]

        return _classify_tags(raw_tags, limit=10)

    # ------------------------------------------------------------------
    # Track tags  →  more specific than artist tags
    # ------------------------------------------------------------------

    async def get_track_tags(self, artist_name: str, track_name: str) -> dict[str, list[str]]:
        """Return classified tags for a specific track."""
        data = await self._get("track.getTopTags", artist=artist_name, track=track_name)
        if not data:
            return {"genre": [], "mood": [], "era": []}

        raw_tags = data.get("toptags", {}).get("tag", [])
        if isinstance(raw_tags, dict):
            raw_tags = [raw_tags]

        return _classify_tags(raw_tags, limit=10)

    # ------------------------------------------------------------------
    # Similar artists  →  SIMILAR_TO edges in the graph
    # ------------------------------------------------------------------

    async def get_similar_artists(
        self, artist_name: str, limit: int = 5
    ) -> list[dict[str, str | float]]:
        """
        Return up to `limit` similar artists.

        Each entry: {"name": str, "match": float (0–1)}
        """
        data = await self._get("artist.getSimilar", artist=artist_name, limit=limit)
        if not data:
            return []

        artists = data.get("similarartists", {}).get("artist", [])
        if isinstance(artists, dict):
            artists = [artists]

        return [
            {"name": a["name"], "match": float(a.get("match", 0.5))}
            for a in artists
            if a.get("name")
        ]


# ---------------------------------------------------------------------------
# Tag classification helper
# ---------------------------------------------------------------------------

def _classify_tags(raw_tags: list[dict], limit: int = 10) -> dict[str, list[str]]:
    genre: list[str] = []
    mood:  list[str] = []
    era:   list[str] = []

    for tag in raw_tags[:limit * 2]:
        name = str(tag.get("name", "")).lower().strip()
        if not name or name in _SKIP_TAGS:
            continue
        if name in _ERA_TAGS or any(e in name for e in ("s music", " era")):
            era.append(name)
        elif name in _MOOD_TAGS:
            mood.append(name)
        else:
            genre.append(name)

    # If specific genre tags exist, drop the generic ones
    # e.g. if "deutschrap" exists, skip "rap" and "hip-hop"
    specific_genres = [g for g in genre if g not in _GENERIC_GENRE_TAGS]
    if specific_genres:
        final_genres = specific_genres[:limit]
    else:
        final_genres = genre[:limit]   # keep generic if nothing specific found

    return {
        "genre": final_genres,
        "mood":  mood[:5],
        "era":   era[:3],
    }

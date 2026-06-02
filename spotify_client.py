"""Spotify OAuth 2.0 Authorization Code Flow + API wrapper with rate-limit handling."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

import httpx
from pydantic_settings import BaseSettings

from models import AudioFeatures, CurrentlyPlaying, TokenData, TrackInfo

logger = logging.getLogger(__name__)

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

REQUIRED_SCOPES = " ".join([
    "user-read-currently-playing",
    "user-modify-playback-state",
    "user-read-playback-state",
    "user-library-read",
    "user-read-recently-played",
    "user-top-read",
])


# ---------------------------------------------------------------------------
# Settings (loaded from .env)
# ---------------------------------------------------------------------------

class SpotifySettings(BaseSettings):
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    spotify_redirect_uri: str = "http://localhost:8000/callback"

    model_config = {"env_file": ".env", "extra": "ignore"}


# ---------------------------------------------------------------------------
# Token storage (in-memory; swap for Redis / DB in production)
# ---------------------------------------------------------------------------

_token_store: dict[str, TokenData] = {}
_state_store: dict[str, str] = {}   # state -> code_verifier (for PKCE)


def _save_token(token: TokenData) -> None:
    _token_store["default"] = token
    # Persist to a local file as well so the token survives restarts
    try:
        with open(".spotify_token.json", "w") as f:
            f.write(token.model_dump_json())
    except OSError:
        pass


def _load_token() -> Optional[TokenData]:
    if "default" in _token_store:
        return _token_store["default"]
    try:
        with open(".spotify_token.json") as f:
            data = json.load(f)
            token = TokenData(**data)
            _token_store["default"] = token
            return token
    except (OSError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge)."""
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


# ---------------------------------------------------------------------------
# HTTP helper with exponential back-off on 429
# ---------------------------------------------------------------------------

async def _spotify_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = 5,
    **kwargs,
) -> Optional[httpx.Response]:
    delay = 1.0
    for attempt in range(max_retries):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", delay))
            logger.warning("Spotify rate limit hit — waiting %.1fs (attempt %d)", retry_after, attempt + 1)
            await asyncio.sleep(retry_after)
            delay = min(delay * 2, 60)
            continue
        if resp.status_code == 401:
            return resp
        if resp.status_code == 403:
            logger.warning("Spotify 403 Forbidden: %s %s (endpoint may require extended quota)", method, url)
            return resp
        resp.raise_for_status()
        return resp
    logger.error("Spotify request failed after %d retries: %s %s", max_retries, method, url)
    return None


# ---------------------------------------------------------------------------
# SpotifyClient
# ---------------------------------------------------------------------------

class SpotifyClient:
    def __init__(self) -> None:
        self.settings = SpotifySettings()
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SpotifyClient":
        self._client = httpx.AsyncClient(timeout=15.0)
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("SpotifyClient must be used as an async context manager")
        return self._client

    # ------------------------------------------------------------------
    # OAuth: Authorization Code + PKCE
    # ------------------------------------------------------------------

    def build_auth_url(self) -> tuple[str, str]:
        """Return (auth_url, state) for redirecting the user."""
        state = secrets.token_urlsafe(16)
        code_verifier, code_challenge = _generate_pkce_pair()
        _state_store[state] = code_verifier

        params = {
            "client_id": self.settings.spotify_client_id,
            "response_type": "code",
            "redirect_uri": self.settings.spotify_redirect_uri,
            "state": state,
            "scope": REQUIRED_SCOPES,
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
        }
        return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}", state

    async def exchange_code(self, code: str, state: str) -> TokenData:
        """Exchange auth code for access + refresh tokens."""
        code_verifier = _state_store.pop(state, None)
        if code_verifier is None:
            raise ValueError("Unknown or expired OAuth state")

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.settings.spotify_redirect_uri,
            "client_id": self.settings.spotify_client_id,
            "code_verifier": code_verifier,
        }
        resp = await self.client.post(SPOTIFY_TOKEN_URL, data=data)
        resp.raise_for_status()
        payload = resp.json()
        token = TokenData(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token", ""),
            expires_at=time.time() + payload.get("expires_in", 3600) - 60,
            scope=payload.get("scope", ""),
        )
        _save_token(token)
        return token

    async def refresh_token(self, token: TokenData) -> TokenData:
        """Use the refresh token to get a new access token."""
        # Spotify requires Basic auth for non-PKCE refresh; for PKCE, client_id only.
        data = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": self.settings.spotify_client_id,
        }
        resp = await self.client.post(SPOTIFY_TOKEN_URL, data=data)
        resp.raise_for_status()
        payload = resp.json()
        updated = TokenData(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token") or token.refresh_token,
            expires_at=time.time() + payload.get("expires_in", 3600) - 60,
            scope=payload.get("scope", token.scope),
        )
        _save_token(updated)
        return updated

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    async def get_valid_token(self) -> Optional[str]:
        """Return a valid access token, refreshing if needed."""
        token = _load_token()
        if token is None:
            return None
        if time.time() >= token.expires_at:
            token = await self.refresh_token(token)
        return token.access_token

    def is_authenticated(self) -> bool:
        return _load_token() is not None

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, **params) -> Optional[dict]:
        access_token = await self.get_valid_token()
        if not access_token:
            return None
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{SPOTIFY_API_BASE}{path}"
        resp = await _spotify_request(self.client, "GET", url, headers=headers, params=params or None)
        if resp is None or resp.status_code in (204, 403):
            return None
        if resp.status_code == 401:
            # Token may have expired mid-request; refresh once and retry
            token = _load_token()
            if token:
                token = await self.refresh_token(token)
                headers["Authorization"] = f"Bearer {token.access_token}"
                resp = await _spotify_request(self.client, "GET", url, headers=headers, params=params or None)
        if resp is None or resp.status_code in (204, 403):
            return None
        return resp.json()

    async def _post(self, path: str, json_body: dict | None = None) -> Optional[dict]:
        access_token = await self.get_valid_token()
        if not access_token:
            return None
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        url = f"{SPOTIFY_API_BASE}{path}"
        resp = await _spotify_request(
            self.client, "POST", url, headers=headers,
            json=json_body or {},
        )
        if resp is None or resp.status_code in (200, 204):
            return {}
        return resp.json() if resp.content else {}

    # ------------------------------------------------------------------
    # Spotify API: currently playing
    # ------------------------------------------------------------------

    async def get_currently_playing(self) -> Optional[CurrentlyPlaying]:
        data = await self._get("/me/player/currently-playing", additional_types="track")
        if data is None:
            return None
        item = data.get("item")
        if item is None or item.get("type") != "track":
            return None

        track = _parse_track(item)
        # Enrich with genres from artist
        track.genres = await self._get_artist_genres(track.artist_ids)

        context = data.get("context") or {}
        return CurrentlyPlaying(
            track=track,
            progress_ms=data.get("progress_ms", 0),
            is_playing=data.get("is_playing", False),
            context_type=context.get("type"),
            context_uri=context.get("uri"),
        )

    # ------------------------------------------------------------------
    # Spotify API: audio features
    # ------------------------------------------------------------------

    async def get_audio_features(self, track_id: str) -> Optional[AudioFeatures]:
        data = await self._get(f"/audio-features/{track_id}")
        if not data:
            return None
        return AudioFeatures(
            danceability=data.get("danceability", 0.0),
            energy=data.get("energy", 0.0),
            valence=data.get("valence", 0.0),
            tempo=data.get("tempo", 0.0),
            acousticness=data.get("acousticness", 0.0),
            instrumentalness=data.get("instrumentalness", 0.0),
            liveness=data.get("liveness", 0.0),
            speechiness=data.get("speechiness", 0.0),
            loudness=data.get("loudness", 0.0),
            key=data.get("key", 0),
            mode=data.get("mode", 0),
            time_signature=data.get("time_signature", 4),
        )

    # ------------------------------------------------------------------
    # Spotify API: track details
    # ------------------------------------------------------------------

    async def get_track(self, track_id: str) -> Optional[TrackInfo]:
        data = await self._get(f"/tracks/{track_id}")
        if not data:
            return None
        track = _parse_track(data)
        track.genres = await self._get_artist_genres(track.artist_ids)
        track.audio_features = await self.get_audio_features(track_id)
        return track

    async def get_tracks_batch(self, track_ids: list[str]) -> list[TrackInfo]:
        """Fetch up to 50 tracks at once."""
        if not track_ids:
            return []
        ids = ",".join(track_ids[:50])
        data = await self._get("/tracks", ids=ids)
        if not data:
            return []
        tracks = [_parse_track(t) for t in data.get("tracks", []) if t]
        return tracks

    async def get_audio_features_batch(self, track_ids: list[str]) -> dict[str, AudioFeatures]:
        """Fetch audio features for up to 100 tracks. Returns {track_id: AudioFeatures}."""
        if not track_ids:
            return {}
        ids = ",".join(track_ids[:100])
        data = await self._get("/audio-features", ids=ids)
        if not data:
            return {}
        result: dict[str, AudioFeatures] = {}
        for item in data.get("audio_features", []):
            if not item:
                continue
            result[item["id"]] = AudioFeatures(
                danceability=item.get("danceability", 0.0),
                energy=item.get("energy", 0.0),
                valence=item.get("valence", 0.0),
                tempo=item.get("tempo", 0.0),
                acousticness=item.get("acousticness", 0.0),
                instrumentalness=item.get("instrumentalness", 0.0),
                liveness=item.get("liveness", 0.0),
                speechiness=item.get("speechiness", 0.0),
                loudness=item.get("loudness", 0.0),
                key=item.get("key", 0),
                mode=item.get("mode", 0),
                time_signature=item.get("time_signature", 4),
            )
        return result

    # ------------------------------------------------------------------
    # Spotify API: related artists + their top tracks (graph bootstrap)
    # ------------------------------------------------------------------

    async def get_related_artists(self, artist_id: str) -> list[dict]:
        """Return up to 20 related artists as list of {id, name}."""
        data = await self._get(f"/artists/{artist_id}/related-artists")
        if not data:
            return []
        return [
            {"id": a["id"], "name": a["name"]}
            for a in data.get("artists", [])
            if a
        ]

    async def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> list[TrackInfo]:
        """Return up to 10 top tracks for an artist."""
        data = await self._get(f"/artists/{artist_id}/top-tracks", market=market)
        if not data:
            return []
        return [_parse_track(t) for t in data.get("tracks", []) if t]

    # ------------------------------------------------------------------
    # Spotify API: artist genres
    # ------------------------------------------------------------------

    async def _get_artist_genres(self, artist_ids: list[str]) -> list[str]:
        """Fetch genres by calling each artist individually (batch endpoint is 403 for new apps)."""
        genres: list[str] = []
        seen: set[str] = set()
        for artist_id in artist_ids[:3]:
            try:
                data = await self._get(f"/artists/{artist_id}")
                if not data:
                    continue
                for g in data.get("genres", []):
                    if g not in seen:
                        seen.add(g)
                        genres.append(g)
            except Exception as exc:
                logger.warning("Could not fetch genres for artist %s: %s", artist_id, exc)
        return genres

    # ------------------------------------------------------------------
    # Spotify API: saved tracks + search (bootstrap sources)
    # ------------------------------------------------------------------

    async def get_saved_tracks(self, limit: int = 50) -> list[TrackInfo]:
        """Return up to 50 of the user's liked/saved tracks."""
        data = await self._get("/me/tracks", limit=min(limit, 50))
        if not data:
            return []
        tracks = []
        for item in data.get("items", []):
            t = item.get("track") if item else None
            if t and t.get("type") == "track":
                tracks.append(_parse_track(t))
        return tracks

    async def search_tracks(self, query: str, limit: int = 20) -> list[TrackInfo]:
        """Search for tracks by query string."""
        data = await self._get("/search", q=query, type="track", limit=min(limit, 50))
        if not data:
            return []
        items = data.get("tracks", {}).get("items", [])
        return [_parse_track(t) for t in items if t]

    # ------------------------------------------------------------------
    # Spotify API: queue
    # ------------------------------------------------------------------

    async def get_queue(self) -> list[str]:
        """Return track IDs currently in the user's queue."""
        data = await self._get("/me/player/queue")
        if not data:
            return []
        queued: list[str] = []
        for item in data.get("queue", []):
            if item and item.get("type") == "track":
                queued.append(item["id"])
        return queued

    async def add_to_queue(self, track_id: str) -> bool:
        """Add a track URI to the user's Spotify queue."""
        uri = f"spotify:track:{track_id}"
        result = await self._post(f"/me/player/queue?uri={uri}")
        return result is not None

    async def add_many_to_queue(self, track_ids: list[str]) -> int:
        """Add multiple tracks to queue; return count successfully added."""
        added = 0
        for tid in track_ids:
            ok = await self.add_to_queue(tid)
            if ok:
                added += 1
            await asyncio.sleep(0.2)   # be gentle with the API
        return added

    # ------------------------------------------------------------------
    # Spotify API: recently played
    # ------------------------------------------------------------------

    async def get_recently_played(self, limit: int = 20) -> list[str]:
        """Return track IDs of recently played tracks."""
        data = await self._get("/me/player/recently-played", limit=min(limit, 50))
        if not data:
            return []
        return [
            item["track"]["id"]
            for item in data.get("items", [])
            if item.get("track")
        ]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_track(data: dict) -> TrackInfo:
    artists = data.get("artists") or []
    album = data.get("album") or {}
    images = album.get("images") or []
    image_url = images[0]["url"] if images else None
    external_urls = data.get("external_urls") or {}

    # Extract release year from album release_date (format: YYYY, YYYY-MM, or YYYY-MM-DD)
    release_year: int | None = None
    release_date = album.get("release_date", "")
    if release_date and len(release_date) >= 4:
        try:
            release_year = int(release_date[:4])
        except ValueError:
            pass

    return TrackInfo(
        spotify_id=data["id"],
        name=data["name"],
        artist_names=[a["name"] for a in artists],
        artist_ids=[a["id"] for a in artists],
        album_name=album.get("name", ""),
        album_id=album.get("id", ""),
        duration_ms=data.get("duration_ms", 0),
        popularity=data.get("popularity", 0),
        preview_url=data.get("preview_url"),
        external_url=external_urls.get("spotify", ""),
        image_url=image_url,
        release_year=release_year,
    )

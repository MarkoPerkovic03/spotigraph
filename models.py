"""Pydantic models for SpotiGraph."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Spotify domain models
# ---------------------------------------------------------------------------

class AudioFeatures(BaseModel):
    danceability: float = 0.0
    energy: float = 0.0
    valence: float = 0.0
    tempo: float = 0.0
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    liveness: float = 0.0
    speechiness: float = 0.0
    loudness: float = 0.0
    key: int = 0
    mode: int = 0
    time_signature: int = 4


class TrackInfo(BaseModel):
    spotify_id: str
    name: str
    artist_names: list[str]
    artist_ids: list[str]
    album_name: str
    album_id: str
    duration_ms: int
    popularity: int = 0
    preview_url: Optional[str] = None
    external_url: str = ""
    image_url: Optional[str] = None
    genres: list[str] = Field(default_factory=list)
    release_year: Optional[int] = None
    audio_features: Optional[AudioFeatures] = None


class CurrentlyPlaying(BaseModel):
    track: TrackInfo
    progress_ms: int
    is_playing: bool
    context_type: Optional[str] = None
    context_uri: Optional[str] = None


# ---------------------------------------------------------------------------
# Enrichment — rule-based, derived from audio features + release year
# ---------------------------------------------------------------------------

class EnrichmentResult(BaseModel):
    """
    Four semantic labels derived deterministically from Spotify audio features.

    mood_label    : energy × valence quadrant  (e.g. "euphoric", "melancholic")
    tempo_label   : tempo × danceability bucket (e.g. "groovy", "fast-paced")
    texture_label : acousticness × instrumentalness (e.g. "acoustic-vocal")
    era_label     : decade from release year    (e.g. "1990s", "contemporary")
    """
    mood_label: str = ""
    tempo_label: str = ""
    texture_label: str = ""
    era_label: str = ""


# ---------------------------------------------------------------------------
# Graph node representation
# ---------------------------------------------------------------------------

class TrackNode(BaseModel):
    spotify_id: str
    name: str
    artist_names: list[str]
    genres: list[str]
    popularity: int
    enriched: bool = False
    danceability: float = 0.0
    energy: float = 0.0
    valence: float = 0.0
    tempo: float = 0.0
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    liveness: float = 0.0
    speechiness: float = 0.0
    loudness: float = 0.0
    key: int = 0
    mode: int = 0


# ---------------------------------------------------------------------------
# Recommendation output
# ---------------------------------------------------------------------------

class RecommendationReason(BaseModel):
    shared_moods: list[str] = Field(default_factory=list)
    shared_tempos: list[str] = Field(default_factory=list)
    shared_textures: list[str] = Field(default_factory=list)
    shared_genres: list[str] = Field(default_factory=list)
    shared_eras: list[str] = Field(default_factory=list)
    via_related_artist: bool = False
    audio_distance: float = 0.0
    score: float = 0.0


class Recommendation(BaseModel):
    track: TrackInfo
    reason: RecommendationReason


class RecommendationResponse(BaseModel):
    source_track: TrackInfo
    recommendations: list[Recommendation]
    added_to_queue: bool = False


# ---------------------------------------------------------------------------
# Auth / session
# ---------------------------------------------------------------------------

class TokenData(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: float
    scope: str = ""


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    spotify_authenticated: bool
    version: str = "0.1.0"

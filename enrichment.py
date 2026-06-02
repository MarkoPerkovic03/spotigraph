"""Track enrichment via Last.fm tags + release year.

Fetches genre and mood tags from Last.fm for each track's artist,
then derives the era label from the album release year.

This replaces Spotify's restricted audio-features and genre endpoints.
"""

from __future__ import annotations

from lastfm_client import LastFmClient
from models import EnrichmentResult, TrackInfo


# ---------------------------------------------------------------------------
# Era: release year → musical decade (own logic, no external API)
# ---------------------------------------------------------------------------

def _era_label(release_year: int | None) -> str:
    if release_year is None:  return "unknown-era"
    if release_year < 1970:   return "pre-1970s"
    if release_year < 1980:   return "1970s"
    if release_year < 1990:   return "1980s"
    if release_year < 2000:   return "1990s"
    if release_year < 2010:   return "2000s"
    if release_year < 2020:   return "2010s"
    return "contemporary"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def enrich(track: TrackInfo, lastfm: LastFmClient) -> EnrichmentResult:
    """
    Enrich a track with Last.fm semantic tags + release year era label.

    Strategy:
    1. Try track-level tags first (most specific).
    2. Fall back to artist-level tags if track has no tags.
    3. Always compute era from release year.
    """
    artist_name = track.artist_names[0] if track.artist_names else ""
    era = _era_label(track.release_year)

    if not artist_name:
        return EnrichmentResult(era_label=era)

    # Track-level tags (most precise)
    track_tags = await lastfm.get_track_tags(artist_name, track.name)

    genre_tags = track_tags["genre"]
    mood_tags  = track_tags["mood"]

    # Fall back to artist-level tags if track has no useful tags
    if not genre_tags and not mood_tags:
        artist_tags = await lastfm.get_artist_tags(artist_name)
        genre_tags  = artist_tags["genre"]
        mood_tags   = artist_tags["mood"]

    return EnrichmentResult(
        genre_tags=genre_tags,
        mood_tags=mood_tags,
        era_label=era,
    )

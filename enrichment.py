"""Rule-based track enrichment for SpotiGraph.

Derives four semantic labels purely from Spotify Audio Features and the
track's release year.  No external AI API is used — the bucketing logic
is the project's own contribution.

Bucketing rationale
-------------------
mood_label    — 3×3 grid on the Energy–Valence plane (Spotify's own
                "arousal–valence" model, a standard musicology framework).
tempo_label   — Tempo × Danceability: captures both rhythmic speed and
                groove feel.
texture_label — Acousticness × Instrumentalness: four quadrants that
                describe the sonic production style.
era_label     — Album release year bucketed into musical decades.
"""

from __future__ import annotations

from models import EnrichmentResult, TrackInfo


# ---------------------------------------------------------------------------
# Mood: Energy × Valence  (3 × 3 grid)
# ---------------------------------------------------------------------------
#
#               valence < 0.35    0.35–0.65    > 0.65
#  energy > 0.65  aggressive      intense      euphoric
#  energy 0.35–0.65  tense        neutral      upbeat
#  energy < 0.35  somber          reflective   peaceful
#
def _mood_label(energy: float, valence: float) -> str:
    if energy > 0.65:
        if valence > 0.65:   return "euphoric"
        if valence > 0.35:   return "intense"
        return "aggressive"
    if energy > 0.35:
        if valence > 0.65:   return "upbeat"
        if valence > 0.35:   return "neutral"
        return "tense"
    # energy <= 0.35
    if valence > 0.65:       return "peaceful"
    if valence > 0.35:       return "reflective"
    return "somber"


# ---------------------------------------------------------------------------
# Tempo: BPM × Danceability
# ---------------------------------------------------------------------------
#
#                  danceability < 0.5    danceability >= 0.5
#  tempo > 140        fast-paced            high-tempo-dance
#  tempo 100–140      mid-tempo             groovy
#  tempo < 100        slow                  slow-groove
#
def _tempo_label(tempo: float, danceability: float) -> str:
    danceable = danceability >= 0.5
    if tempo > 140:
        return "high-tempo-dance" if danceable else "fast-paced"
    if tempo >= 100:
        return "groovy"          if danceable else "mid-tempo"
    return "slow-groove"         if danceable else "slow"


# ---------------------------------------------------------------------------
# Texture: Acousticness × Instrumentalness
# ---------------------------------------------------------------------------
#
#                   instrumentalness < 0.5    >= 0.5
#  acousticness > 0.5   acoustic-vocal       acoustic-instrumental
#  acousticness <= 0.5  electronic-vocal     electronic-instrumental
#
def _texture_label(acousticness: float, instrumentalness: float) -> str:
    acoustic = acousticness > 0.5
    instrumental = instrumentalness >= 0.5
    if acoustic and instrumental:     return "acoustic-instrumental"
    if acoustic and not instrumental: return "acoustic-vocal"
    if not acoustic and instrumental: return "electronic-instrumental"
    return "electronic-vocal"


# ---------------------------------------------------------------------------
# Era: release year → musical decade
# ---------------------------------------------------------------------------

def _era_label(release_year: int | None) -> str:
    if release_year is None:
        return "unknown-era"
    if release_year < 1970:  return "pre-1970s"
    if release_year < 1980:  return "1970s"
    if release_year < 1990:  return "1980s"
    if release_year < 2000:  return "1990s"
    if release_year < 2010:  return "2000s"
    if release_year < 2020:  return "2010s"
    return "contemporary"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich(track: TrackInfo) -> EnrichmentResult:
    """
    Derive semantic labels from a track's audio features and release year.

    If audio features are unavailable (e.g. Spotify 403), only the era label
    is set — the track is still usable for graph traversal via Genre + Era.
    """
    era = _era_label(track.release_year)

    if track.audio_features is None:
        return EnrichmentResult(era_label=era)

    af = track.audio_features
    return EnrichmentResult(
        mood_label=_mood_label(af.energy, af.valence),
        tempo_label=_tempo_label(af.tempo, af.danceability),
        texture_label=_texture_label(af.acousticness, af.instrumentalness),
        era_label=era,
    )

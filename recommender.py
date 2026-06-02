"""Graph traversal recommendation engine for SpotiGraph.

Pipeline
--------
1. Get currently playing track from Spotify.
2. Upsert track into Neo4j (audio features as node properties).
3. Rule-based enrichment → Mood / Tempo / Texture / Era nodes attached.
4. Graph bootstrap (runs once per track):
     - Fetch Related Artists from Spotify → SIMILAR_TO edges between Artist nodes.
     - Fetch top tracks for each related artist → upsert + enrich those tracks too.
   This populates the graph so traversal can find real candidates.
5. Traverse graph: find tracks sharing Mood / Tempo / Texture / Era / Genre
   nodes, or reachable via related Artist edges.
6. Score candidates:
     score = semantic_overlap × W_OVERLAP
           + artist_bonus    × W_ARTIST
           - audio_distance  × W_AUDIO
7. Return top N; optionally add to Spotify queue.
"""

from __future__ import annotations

import logging
from typing import Optional

from enrichment import enrich
from graph_client import GraphClient, audio_distance
from models import (
    AudioFeatures,
    Recommendation,
    RecommendationReason,
    RecommendationResponse,
    TrackInfo,
)
from spotify_client import SpotifyClient

logger = logging.getLogger(__name__)

# Scoring weights
W_OVERLAP = 2.0   # reward shared semantic nodes (Mood/Tempo/Texture/Era/Genre)
W_ARTIST  = 3.0   # reward reachability via related-artist edges
W_AUDIO   = 1.5   # penalise sonic dissimilarity

# Bootstrap: how many related artists and their top tracks to ingest
MAX_RELATED_ARTISTS = 5
MAX_TOP_TRACKS_PER_ARTIST = 5


class Recommender:
    def __init__(self, graph: GraphClient, spotify: SpotifyClient) -> None:
        self._graph = graph
        self._spotify = spotify

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def recommend(
        self,
        limit: int = 5,
        add_to_queue: bool = False,
    ) -> Optional[RecommendationResponse]:
        now_playing = await self._spotify.get_currently_playing()
        if now_playing is None:
            logger.info("No track currently playing")
            return None

        source = now_playing.track

        # Ensure track is in graph with audio features and semantic nodes
        await self._ensure_track_in_graph(source)

        # Build exclusion list (queue + recently played + current)
        exclude = set(await self._spotify.get_recently_played(limit=20))
        exclude |= set(await self._spotify.get_queue())
        exclude.add(source.spotify_id)

        # Graph traversal
        candidates_raw = await self._graph.find_candidates(
            track_id=source.spotify_id,
            exclude_ids=list(exclude),
            candidate_limit=max(limit * 10, 50),
        )

        if not candidates_raw:
            logger.info("No graph candidates — graph is still sparse; bootstrapping helped but may need more tracks")
            return RecommendationResponse(
                source_track=source,
                recommendations=[],
                added_to_queue=False,
            )

        # Source audio features for distance calculation
        source_af = await self._graph.get_track_audio_features(source.spotify_id)
        if source_af is None:
            af = source.audio_features or AudioFeatures()
            source_af = {
                "danceability": af.danceability, "energy": af.energy,
                "valence": af.valence,           "tempo": af.tempo,
                "acousticness": af.acousticness, "instrumentalness": af.instrumentalness,
                "liveness": af.liveness,         "speechiness": af.speechiness,
            }

        scored = _score_candidates(candidates_raw, source_af)
        top = scored[:limit]

        # Fetch full TrackInfo for top candidates
        track_map = {t.spotify_id: t for t in await self._spotify.get_tracks_batch(
            [c["spotify_id"] for c in top]
        )}

        recommendations: list[Recommendation] = []
        for c in top:
            tid = c["spotify_id"]
            track_info = track_map.get(tid) or TrackInfo(
                spotify_id=tid,
                name=c.get("name", "Unknown"),
                artist_names=list(c.get("artist_names") or []),
                artist_ids=[],
                album_name="", album_id="", duration_ms=0,
                genres=list(c.get("genres") or []),
            )
            recommendations.append(Recommendation(
                track=track_info,
                reason=RecommendationReason(
                    shared_moods=list(c.get("shared_moods") or []),
                    shared_tempos=list(c.get("shared_tempos") or []),
                    shared_textures=list(c.get("shared_textures") or []),
                    shared_genres=list(c.get("shared_genres") or []),
                    shared_eras=list(c.get("shared_eras") or []),
                    via_related_artist=bool(c.get("via_related_artist")),
                    audio_distance=round(c["audio_dist"], 3),
                    score=round(c["score"], 3),
                ),
            ))

        added = False
        if add_to_queue and recommendations:
            n = await self._spotify.add_many_to_queue(
                [r.track.spotify_id for r in recommendations]
            )
            added = n > 0
            logger.info("Added %d/%d tracks to queue", n, len(recommendations))

        return RecommendationResponse(
            source_track=source,
            recommendations=recommendations,
            added_to_queue=added,
        )

    # ------------------------------------------------------------------
    # Ingest a single track (used by /ingest endpoint + bootstrap)
    # ------------------------------------------------------------------

    async def ingest_track_by_id(self, spotify_id: str) -> Optional[TrackInfo]:
        track = await self._spotify.get_track(spotify_id)
        if not track:
            return None
        await self._ensure_track_in_graph(track)
        return track

    # ------------------------------------------------------------------
    # Internal: ensure track is fully in graph
    # ------------------------------------------------------------------

    async def _ensure_track_in_graph(self, track: TrackInfo) -> None:
        # Fetch audio features if not yet loaded
        if track.audio_features is None:
            track.audio_features = await self._spotify.get_audio_features(track.spotify_id)

        await self._graph.upsert_track(track)

        if not await self._graph.is_enriched(track.spotify_id):
            result = enrich(track)   # pure rule-based, no API call
            await self._graph.apply_enrichment(track.spotify_id, result)
            logger.debug("Enriched '%s' → mood=%s tempo=%s texture=%s era=%s",
                         track.name, result.mood_label, result.tempo_label,
                         result.texture_label, result.era_label)

            # Bootstrap graph edges for this track (runs once per track)
            await self._bootstrap_graph(track)

    # ------------------------------------------------------------------
    # Bootstrap: Related Artists → SIMILAR_TO edges + their top tracks
    # ------------------------------------------------------------------

    async def _bootstrap_graph(self, track: TrackInfo) -> None:
        """
        Populate the graph with genre- and artist-appropriate tracks.

        Priority order:
        1. Search by genre  — most semantically relevant candidates
        2. Search by artist — more tracks from the same artist
        3. Saved tracks     — user's personal library as fallback
        """
        ingested = 0

        async def ingest(t: TrackInfo) -> None:
            nonlocal ingested
            # Fetch genres if missing (single artist endpoint)
            if not t.genres and t.artist_ids:
                t.genres = await self._spotify._get_artist_genres(t.artist_ids)
            await self._graph.upsert_track(t)
            if not await self._graph.is_enriched(t.spotify_id):
                await self._graph.apply_enrichment(t.spotify_id, enrich(t))
            ingested += 1

        # Source 1: search by genre (best match — genre-homogeneous candidates)
        for genre in track.genres[:3]:
            results = await self._spotify.search_tracks(f'genre:"{genre}"', limit=20)
            for t in results:
                await ingest(t)

        # Source 2: search by artist name
        for artist_name in track.artist_names[:2]:
            results = await self._spotify.search_tracks(
                f'artist:"{artist_name}"', limit=20
            )
            for t in results:
                await ingest(t)

        # Source 3: saved tracks (personal library fallback)
        saved = await self._spotify.get_saved_tracks(limit=50)
        for t in saved:
            await ingest(t)

        logger.info("Bootstrap complete — %d tracks ingested into graph", ingested)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_candidates(
    candidates: list[dict],
    source_af: dict[str, float],
) -> list[dict]:
    for c in candidates:
        overlap = (
            len(c.get("shared_moods")    or [])
            + len(c.get("shared_tempos")   or [])
            + len(c.get("shared_textures") or [])
            + len(c.get("shared_genres")   or [])
            + len(c.get("shared_eras")     or [])
        )
        artist_bonus = 1 if c.get("via_related_artist") else 0
        candidate_af = {
            "danceability":    c.get("danceability", 0.0),
            "energy":          c.get("energy", 0.0),
            "valence":         c.get("valence", 0.0),
            "tempo":           c.get("tempo", 0.0),
            "acousticness":    c.get("acousticness", 0.0),
            "instrumentalness":c.get("instrumentalness", 0.0),
            "liveness":        c.get("liveness", 0.0),
            "speechiness":     c.get("speechiness", 0.0),
        }
        adist = audio_distance(source_af, candidate_af)
        c["audio_dist"] = adist
        c["score"] = overlap * W_OVERLAP + artist_bonus * W_ARTIST - adist * W_AUDIO

    return sorted(candidates, key=lambda x: x["score"], reverse=True)

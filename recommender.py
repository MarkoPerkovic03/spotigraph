"""Graph traversal recommendation engine for SpotiGraph.

Pipeline
--------
1. Get currently playing track from Spotify.
2. Enrich track via Last.fm (genre + mood tags → graph nodes).
3. Bootstrap graph:
     a. Last.fm similar artists → search on Spotify → ingest tracks
     b. User's saved tracks → ingest
4. Traverse graph: find tracks sharing Genre / Mood / Era nodes
   or connected via SIMILAR_TO artist edges.
5. Score candidates using three signals:
     a. TF-IDF Genre Score  — rare shared genres score higher than common ones
     b. Multi-Signal Bonus  — bonus when genre + mood + artist all match
     c. Popularity Penalty  — penalize large popularity gap (niche vs mainstream)
6. Return top N (max 1 per artist); optionally add to Spotify queue.

Scoring formula
---------------
    genre_score    = Σ  log((N+1)/(freq_i+1)) + 1   for each shared genre i
    mood_score     = shared_moods × 2.0
    era_score      = shared_eras  × 0.5
    multi_bonus    = (active_signals - 1) × 1.5      if >1 signal active
    pop_penalty    = |seed.popularity - cand.popularity| / 100 × 0.8
    ──────────────────────────────────────────────────────────────────
    final_score    = genre_score + mood_score + era_score
                   + multi_bonus - pop_penalty
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from enrichment import enrich
from graph_client import GraphClient
from lastfm_client import LastFmClient
from models import (
    Recommendation,
    RecommendationReason,
    RecommendationResponse,
    TrackInfo,
)
from spotify_client import SpotifyClient

logger = logging.getLogger(__name__)

MAX_SIMILAR_ARTISTS = 5
MAX_TRACKS_PER_ARTIST = 10


class Recommender:
    def __init__(
        self,
        graph: GraphClient,
        spotify: SpotifyClient,
        lastfm: LastFmClient,
    ) -> None:
        self._graph   = graph
        self._spotify = spotify
        self._lastfm  = lastfm

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
        await self._ensure_track_in_graph(source)

        exclude = set(await self._spotify.get_recently_played(limit=20))
        exclude |= set(await self._spotify.get_queue())
        exclude.add(source.spotify_id)

        candidates_raw = await self._graph.find_candidates(
            track_id=source.spotify_id,
            exclude_ids=list(exclude),
            candidate_limit=max(limit * 15, 75),
        )

        if not candidates_raw:
            logger.info("No candidates found — graph may need more tracks")
            return RecommendationResponse(
                source_track=source,
                recommendations=[],
                added_to_queue=False,
            )

        # Genre frequencies for TF-IDF weighting — N = actual track count in graph
        genre_freqs = await self._graph.get_genre_frequencies()
        stats = await self._graph.get_stats()
        total_tracks = max(int(stats.get("tracks", 1)), max(genre_freqs.values(), default=1))
        seed_popularity = source.popularity or 50

        # Score with TF-IDF + multi-signal + popularity
        scored = _score_candidates(candidates_raw, genre_freqs, total_tracks, seed_popularity)

        # Diversity: max 1 track per artist
        top: list[dict] = []
        seen_artists: set[str] = set()
        for c in scored:
            artist_names = [a.lower() for a in (c.get("artist_names") or [])]
            if any(a in seen_artists for a in artist_names):
                continue
            top.append(c)
            seen_artists.update(artist_names)
            if len(top) >= limit:
                break
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
                artist_ids=[], album_name="", album_id="", duration_ms=0,
            )
            recommendations.append(Recommendation(
                track=track_info,
                reason=RecommendationReason(
                    shared_genres=list(c.get("shared_genres") or []),
                    shared_moods=list(c.get("shared_moods") or []),
                    shared_eras=list(c.get("shared_eras") or []),
                    via_related_artist=bool(c.get("via_related_artist")),
                    score=round(float(c.get("score", 0)), 2),
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
    # Manual ingestion (used by /ingest endpoint)
    # ------------------------------------------------------------------

    async def ingest_track_by_id(self, spotify_id: str) -> Optional[TrackInfo]:
        track = await self._spotify.get_track(spotify_id)
        if not track:
            return None
        await self._ensure_track_in_graph(track)
        return track

    # ------------------------------------------------------------------
    # Internal: ensure track is enriched in graph
    # ------------------------------------------------------------------

    async def _ensure_track_in_graph(self, track: TrackInfo) -> None:
        await self._graph.upsert_track(track)

        if not await self._graph.is_enriched(track.spotify_id):
            result = await enrich(track, self._lastfm)
            await self._graph.apply_enrichment(track.spotify_id, result)
            logger.info("Last.fm enrichment result: %s", result.model_dump())
            logger.info(
                "Enriched '%s' → genres=%s moods=%s era=%s",
                track.name,
                result.genre_tags[:3],
                result.mood_tags[:2],
                result.era_label,
            )
            await self._bootstrap_graph(track)

    # ------------------------------------------------------------------
    # Bootstrap: Last.fm similar artists → Spotify search → ingest
    # ------------------------------------------------------------------

    async def _bootstrap_graph(self, track: TrackInfo) -> None:
        """
        Build graph edges by:
        1. Asking Last.fm for artists similar to this track's artist.
        2. Searching Spotify for those artists' tracks.
        3. Ingesting + enriching those tracks.
        4. Creating SIMILAR_TO edges between artists in the graph.

        Also ingests user's saved tracks as personal context.
        """
        ingested = 0

        for artist_name in track.artist_names[:2]:
            similar = await self._lastfm.get_similar_artists(
                artist_name, limit=MAX_SIMILAR_ARTISTS
            )
            logger.info("Last.fm similar artists for '%s': %s", artist_name, [s['name'] for s in similar])
            for sim in similar:
                sim_name = sim["name"]
                weight   = float(sim["match"])

                # Search Spotify for this similar artist's tracks
                results = await self._spotify.search_tracks(
                    f'artist:"{sim_name}"', limit=MAX_TRACKS_PER_ARTIST
                )
                for t in results:
                    await self._graph.upsert_track(t)
                    if not await self._graph.is_enriched(t.spotify_id):
                        r = await enrich(t, self._lastfm)
                        await self._graph.apply_enrichment(t.spotify_id, r)
                    # Direct edge: seed track → similar artist's track
                    await self._graph.create_sonically_similar(
                        track.spotify_id, t.spotify_id, score=weight
                    )
                    ingested += 1

                # Create SIMILAR_TO edge (artist_name → sim_name)
                # We use artist names as IDs here since we may not have Spotify IDs
                await self._graph.create_artist_similar_by_name(
                    artist_name, sim_name, weight
                )

        # Also ingest user's saved tracks as personal context
        saved = await self._spotify.get_saved_tracks(limit=30)
        for t in saved:
            await self._graph.upsert_track(t)
            if not await self._graph.is_enriched(t.spotify_id):
                r = await enrich(t, self._lastfm)
                await self._graph.apply_enrichment(t.spotify_id, r)
            ingested += 1

        logger.info("Bootstrap complete — %d tracks ingested into graph", ingested)


# ---------------------------------------------------------------------------
# Scoring: TF-IDF genres + multi-signal bonus + popularity penalty
# ---------------------------------------------------------------------------

def _score_candidates(
    candidates: list[dict],
    genre_freqs: dict[str, int],
    total_tracks: int,
    seed_popularity: int,
) -> list[dict]:
    """
    Score each candidate track using three signals:

    1. TF-IDF Genre Score
       Rare shared genres score higher than common ones.
       score_i = log((N+1) / (freq_i+1)) + 1   for each shared genre i
       → "balkanski trap" shared by 3 tracks scores ~3× higher than
         "pop" shared by 150 tracks.

    2. Multi-Signal Bonus
       Bonus when more than one signal type is active simultaneously
       (genre + mood, genre + artist, all three).
       bonus = (active_signals - 1) × 1.5
       → Rewards tracks with multiple independent reasons to be recommended.

    3. Popularity Penalty
       Penalizes large popularity gaps between seed and candidate.
       penalty = |seed_pop - cand_pop| / 100 × 0.8
       → Playing underground music → mainstream hits are penalized.
    """
    for c in candidates:
        shared_genres   = c.get("shared_genres")   or []
        shared_moods    = c.get("shared_moods")    or []
        shared_eras     = c.get("shared_eras")     or []
        via_artist      = bool(c.get("via_related_artist"))
        direct_similar  = bool(c.get("direct_similar"))
        cand_popularity = int(c.get("popularity") or 50)

        # 1. TF-IDF genre score
        genre_score = 0.0
        for genre in shared_genres:
            freq = genre_freqs.get(genre, 1)
            idf = math.log((total_tracks + 1) / (freq + 1)) + 1
            genre_score += idf

        # 2. Mood score
        mood_score = len(shared_moods) * 2.0

        # 3. Era score (minor signal)
        era_score = len(shared_eras) * 0.5

        # 4. Direct similarity bonus (SONICALLY_SIMILAR edge = strong direct connection)
        direct_bonus = 3.0 if direct_similar else 0.0

        # 5. Multi-signal bonus
        active_signals = sum([genre_score > 0, mood_score > 0, via_artist, direct_similar])
        multi_bonus = (active_signals - 1) * 1.5 if active_signals > 1 else 0.0

        # 5. Popularity penalty (gentle — only penalizes extreme mismatches)
        pop_penalty = abs(seed_popularity - cand_popularity) / 100.0 * 0.3

        c["score"] = genre_score + mood_score + era_score + direct_bonus + multi_bonus - pop_penalty

    return sorted(candidates, key=lambda x: x["score"], reverse=True)

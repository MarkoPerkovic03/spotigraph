"""Graph traversal recommendation engine for SpotiGraph.

Pipeline
--------
1. Get currently playing track from Spotify.
2. Enrich track via Last.fm (genres) + Deezer (loudness/bpm → Energy bucket).
3. Bootstrap graph:
     a. Last.fm similar artists → search on Spotify → ingest tracks
     b. User's saved tracks → ingest (excluded from being recommended)
4. Weighted spreading activation from the seed over the graph: candidates
   accumulate activation along shared Genre / Energy / Era nodes and along
   weighted SIMILAR_TO (artist) and SONICALLY_SIMILAR (sonic) edges.
5. Refine by vibe (fine Deezer energy distance), gate, diversify.
6. Return top N (max 1 per artist); optionally add to Spotify queue.

Ranking formula
---------------
    relevance = w_g·Σidf(genre) + w_e·Σidf(energy) + w_a·artist_score
              + w_s·sonic_score + w_era·#eras − pop_penalty
    vibe_factor = 0.5 + 0.5·(1 − min(vibe_distance, 1))   (0.5 if no audio)
    ──────────────────────────────────────────────────────────────────
    score       = max(relevance, 0) × vibe_factor

    artist_score = Σ over SIMILAR_TO paths of (edge weight × hop-decay), max path
    sonic_score  = SONICALLY_SIMILAR.score

    Gates (applied in recommend(), in order):
      1. relevance — keep only real graph links (genre / artist / sonic);
         energy- or era-only matches are dropped.
      2. vibe — among those, drop energy outliers (vibe_distance > 0.35).
    Plus: the user's saved tracks are excluded from candidates.

Graph relevance (multiplicative) drives the order — a no-link track can't win
on coincidental loudness; vibe only refines among genuinely related tracks.
Signal weights live in the _W_* constants below.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from deezer_client import DeezerClient
from enrichment import enrich
from gnn_recommender import GnnScorer
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

# Normalization for the Deezer energy proxy (so bpm and loudness are comparable)
_BPM_NORM = 200.0        # typical bpm span 60–200
_LOUDNESS_NORM = 20.0    # Deezer gain in dB, typically -20..0

# Ranking: weighted spreading activation (graph relevance) refined by vibe.
# Signal weights — retune here.
_W_GENRE  = 1.0    # shared genres (already IDF-weighted)
_W_ENERGY = 0.8    # shared Deezer energy buckets (already IDF-weighted)
_W_ARTIST = 4.0    # artist similarity (SIMILAR_TO weight × hop-decay, ~0..1)
_W_SONIC  = 3.0    # direct sonic edge (SONICALLY_SIMILAR score, ~0..1)
_W_ERA    = 0.5    # shared era (weak, per shared era)

_VIBE_NONE_DEFAULT = 0.5   # neutral vibe for candidates without Deezer audio data
VIBE_GATE = 0.35           # drop relevant candidates whose energy is this far from the seed


class Recommender:
    def __init__(
        self,
        graph: GraphClient,
        spotify: SpotifyClient,
        lastfm: LastFmClient,
        deezer: Optional[DeezerClient] = None,
        gnn: Optional["GnnScorer"] = None,
    ) -> None:
        self._graph   = graph
        self._spotify = spotify
        self._lastfm  = lastfm
        self._deezer  = deezer
        self._gnn     = gnn

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
        # Exclude the user's own library — they want graph discoveries related to
        # the seed, not the songs they already listen to ("nicht meine Lieder").
        saved = await self._spotify.get_saved_tracks(limit=50)
        exclude |= {t.spotify_id for t in saved}
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

        # Node frequencies for IDF weighting — N = actual track count in graph
        genre_freqs = await self._graph.get_genre_frequencies()
        energy_freqs = await self._graph.get_energy_frequencies()
        stats = await self._graph.get_stats()
        total_tracks = max(int(stats.get("tracks", 1)), max(genre_freqs.values(), default=1))
        seed_popularity = source.popularity or 50

        # Heuristic scoring (spreading activation) — always computed: it fills
        # the explainability fields (genre/energy/vibe) and the gate flags.
        scored = _score_candidates(
            candidates_raw, genre_freqs, energy_freqs, total_tracks, seed_popularity
        )

        # GNN re-ranking: if a trained model exists, replace the ranking score
        # with the learned link-prediction probability seed↔candidate.
        # Falls back silently to the heuristic order when unavailable.
        if self._gnn is not None and self._gnn.is_ready():
            try:
                export = await self._graph.export_graph()
                probs = self._gnn.score(
                    export, source.spotify_id, [c["spotify_id"] for c in scored]
                )
                if probs:
                    for c in scored:
                        c["gnn_prob"] = probs.get(c["spotify_id"])
                        if c["gnn_prob"] is not None:
                            c["score"] = round(float(c["gnn_prob"]), 4)
                            c["via_gnn"] = True
                    scored.sort(key=lambda x: x["score"], reverse=True)
                    logger.info("GNN link-prediction applied to %d candidates", len(probs))
            except Exception as exc:
                logger.warning("GNN scoring failed, using heuristic order: %s", exc)

        # Measurement: log the activation/score breakdown
        _log_vibe_comparison(source, scored)

        # 1) Relevance gate: only recommend tracks with a REAL graph connection
        #    to the seed (shared genre / related artist / sonic edge). This drops
        #    tracks that merely share an era or a coincidental loudness — e.g.
        #    saved songs from an unrelated genre. "Schlage Lieder anhand Graph."
        before = len(scored)
        relevant = [c for c in scored if c.get("has_graph_link")]
        logger.info(
            "Relevance gate: kept %d/%d (dropped %d with no real graph link)",
            len(relevant), before, before - len(relevant),
        )
        if relevant:
            scored = relevant

        # 2) Vibe gate: among relevant tracks, drop energy outliers. Tracks
        #    without Deezer audio pass (we can't judge their energy).
        before = len(scored)
        gated = [c for c in scored
                 if c.get("vibe_distance") is None or c["vibe_distance"] <= VIBE_GATE]
        logger.info(
            "Vibe gate (<= %.2f): kept %d/%d, dropped %d energy outliers",
            VIBE_GATE, len(gated), before, before - len(gated),
        )
        if gated:
            scored = gated

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
                    shared_energy=list(c.get("shared_energy") or []),
                    shared_eras=list(c.get("shared_eras") or []),
                    via_related_artist=bool(c.get("via_related_artist")),
                    via_gnn=bool(c.get("via_gnn")),
                    score=round(float(c.get("score", 0)), 3),
                    genre_score=round(float(c.get("genre_score", 0)), 2),
                    vibe_distance=c.get("vibe_distance"),
                    seed_bpm=c.get("seed_bpm") or None,
                    cand_bpm=c.get("cand_bpm") or None,
                    seed_loudness=c.get("seed_loudness") or None,
                    cand_loudness=c.get("cand_loudness") or None,
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
            result = await enrich(track, self._lastfm, self._deezer)
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
                        r = await enrich(t, self._lastfm, self._deezer)
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
                r = await enrich(t, self._lastfm, self._deezer)
                await self._graph.apply_enrichment(t.spotify_id, r)
            ingested += 1

        logger.info("Bootstrap complete — %d tracks ingested into graph", ingested)


# ---------------------------------------------------------------------------
# Scoring: TF-IDF genres + multi-signal bonus + popularity penalty
# ---------------------------------------------------------------------------

def _idf(freq: int, total: int) -> float:
    """Inverse document frequency — rare shared nodes carry more activation."""
    return math.log((total + 1) / (freq + 1)) + 1


def _score_candidates(
    candidates: list[dict],
    genre_freqs: dict[str, int],
    energy_freqs: dict[str, int],
    total_tracks: int,
    seed_popularity: int,
) -> list[dict]:
    """
    Weighted spreading activation: each candidate's relevance is the weighted
    sum of the graph paths connecting it to the seed, then refined by vibe.

        genre   = Σ idf(g)           over shared genres        (rare > common)
        energy  = Σ idf(e)           over shared energy buckets (Deezer vibe)
        artist  = artist_score       (Σ SIMILAR_TO weight × hop-decay, from query)
        sonic   = sonic_score        (SONICALLY_SIMILAR.score, from query)
        era     = #shared_eras × small constant
        ───────────────────────────────────────────────────────────────────
        relevance   = w_g·genre + w_e·energy + w_a·artist + w_s·sonic + w_era·era
                      − popularity_penalty
        vibe_factor = 0.5 + 0.5·(1 − min(vibe_distance, 1))   # ±50% from energy
        score       = relevance × vibe_factor

    Multiplicative refinement keeps relevance primary: a track with no real
    graph link cannot win on a coincidental loudness match.
    """
    for c in candidates:
        shared_genres = c.get("shared_genres") or []
        shared_energy = c.get("shared_energy") or []
        shared_eras   = c.get("shared_eras")   or []
        artist_score  = float(c.get("artist_score") or 0.0)
        sonic_score   = float(c.get("sonic_score") or 0.0)
        cand_pop      = int(c.get("popularity") or 50)

        genre_act  = sum(_idf(genre_freqs.get(g, 1), total_tracks) for g in shared_genres)
        energy_act = sum(_idf(energy_freqs.get(e, 1), total_tracks) for e in shared_energy)
        era_act    = len(shared_eras) * _W_ERA
        pop_penalty = abs(seed_popularity - cand_pop) / 100.0 * 0.3

        relevance = (
            _W_GENRE  * genre_act
            + _W_ENERGY * energy_act
            + _W_ARTIST * artist_score
            + _W_SONIC  * sonic_score
            + era_act
            - pop_penalty
        )

        # Vibe (fine-grained energy distance) refines the order ±50%.
        vibe_distance = _vibe_distance(
            c.get("seed_bpm"), c.get("seed_loudness"),
            c.get("cand_bpm"), c.get("cand_loudness"),
        )
        vibe_component = (_VIBE_NONE_DEFAULT if vibe_distance is None
                          else 1.0 - min(vibe_distance, 1.0))

        c["score"] = max(relevance, 0.0) * (0.5 + 0.5 * vibe_component)
        # Real graph link = genre / artist / sonic (NOT energy or era alone).
        c["has_graph_link"] = (genre_act > 0 or artist_score > 0 or sonic_score > 0)
        # Component breakdown (for the measurement log + reason transparency).
        c["genre_score"]  = round(genre_act, 3)
        c["energy_score"] = round(energy_act, 3)
        c["relevance"]    = round(relevance, 3)
        c["vibe_distance"] = vibe_distance

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def _vibe_distance(
    seed_bpm: Optional[float],
    seed_loudness: Optional[float],
    cand_bpm: Optional[float],
    cand_loudness: Optional[float],
) -> Optional[float]:
    """
    Normalized RMS distance in the Deezer energy proxy (tempo + loudness).

    0.0 = identical energy, higher = more different. Returns None when no
    audio dimension is shared (missing Deezer data). 0.0 values are treated
    as missing: bpm 0 means "unknown", and loudness 0.0 is the node default
    (real Deezer gain is virtually always negative).
    """
    dims: list[float] = []
    if seed_bpm and cand_bpm:
        dims.append(((seed_bpm - cand_bpm) / _BPM_NORM) ** 2)
    if seed_loudness and cand_loudness:
        dims.append(((seed_loudness - cand_loudness) / _LOUDNESS_NORM) ** 2)
    if not dims:
        return None
    return round(math.sqrt(sum(dims) / len(dims)), 3)


def _log_vibe_comparison(source: TrackInfo, scored: list[dict], top_n: int = 15) -> None:
    """
    Log the spreading-activation breakdown per candidate (genre / energy /
    artist / sonic → relevance, then vibe) so the ranking is inspectable.
    Pure measurement — does not change what gets recommended.
    """
    top = scored[:top_n]
    with_vibe = [c for c in top if c.get("vibe_distance") is not None]

    logger.info(
        "── Activation for '%s' (seed loud=%s) — %d/%d have Deezer audio ──",
        source.name,
        _fmt(top[0].get("seed_loudness")) if top else "?",
        len(with_vibe), len(top),
    )
    logger.info("  rank | %-28s | genre energy artist sonic | relev | vibeΔ | score", "track")
    for i, c in enumerate(top, 1):
        vd = c.get("vibe_distance")
        logger.info(
            "  %-4d | %-28s | %5.2f %5.2f %5.2f %5.2f | %5.2f | %5s | %5.2f",
            i, (c.get("name") or "")[:28],
            float(c.get("genre_score", 0.0)), float(c.get("energy_score", 0.0)),
            float(c.get("artist_score", 0.0)), float(c.get("sonic_score", 0.0)),
            float(c.get("relevance", 0.0)),
            f"{vd:.2f}" if vd is not None else "n/a",
            float(c.get("score", 0.0)),
        )


def _fmt(v: Optional[float]) -> str:
    return f"{v:.1f}" if v else "n/a"

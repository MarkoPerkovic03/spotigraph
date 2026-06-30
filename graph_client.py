"""Neo4j Knowledge Graph client for SpotiGraph.

Schema
------
Nodes:
  Track    {spotify_id, name, artist_names, genres, popularity, enriched,
            release_year, danceability, energy, valence, tempo, acousticness,
            instrumentalness, liveness, speechiness, loudness, key, mode}
  Artist   {spotify_id, name}
  Genre    {name}
  Mood     {name}   -- Last.fm mood tag (e.g. "euphoric", "somber")
  Energy   {name}   -- Deezer loudness/bpm bucket (e.g. "calm", "energetic")
  Era      {label}  -- release decade (e.g. "1990s", "contemporary")

Relationships:
  (Track)-[:BY]->(Artist)
  (Track)-[:HAS_GENRE]->(Genre)
  (Track)-[:EVOKES]->(Mood)
  (Track)-[:HAS_ENERGY]->(Energy)
  (Track)-[:BELONGS_TO_ERA]->(Era)
  (Artist)-[:SIMILAR_TO {weight: float}]->(Artist)
  (Track)-[:SONICALLY_SIMILAR {score: float}]->(Track)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic_settings import BaseSettings

from models import AudioFeatures, EnrichmentResult, TrackInfo

logger = logging.getLogger(__name__)


def _energy_bucket(loudness: Optional[float], bpm: Optional[float] = None) -> Optional[str]:
    """
    Map the Deezer loudness (gain, dB) to a coarse energy bucket node.

    Deezer gain is ~ -20..0 dB. We turn it into a graph-traversable "vibe"
    dimension (replacing the dead Mood signal). bpm is accepted for future
    refinement but is missing for most tracks, so loudness drives the bucket.
    Returns None when there is no usable loudness (0.0 / None = unknown).
    """
    if not loudness:           # 0.0 (node default) or None → unknown
        return None
    if loudness >= -6:   return "intense"
    if loudness >= -9:   return "energetic"
    if loudness >= -12:  return "moderate"
    if loudness >= -16:  return "calm"
    return "very-calm"


class Neo4jSettings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "spotigraph123"

    model_config = {"env_file": ".env", "extra": "ignore"}


class GraphClient:
    def __init__(self) -> None:
        settings = Neo4jSettings()
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def verify_connectivity(self) -> bool:
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.warning("Neo4j connectivity check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        constraints = [
            "CREATE CONSTRAINT track_id   IF NOT EXISTS FOR (t:Track)   REQUIRE t.spotify_id IS UNIQUE",
            "CREATE CONSTRAINT artist_name IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre)   REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT mood_name  IF NOT EXISTS FOR (m:Mood)    REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT era_label  IF NOT EXISTS FOR (e:Era)     REQUIRE e.label IS UNIQUE",
            "CREATE CONSTRAINT energy_name IF NOT EXISTS FOR (en:Energy) REQUIRE en.name IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX track_name    IF NOT EXISTS FOR (t:Track) ON (t.name)",
            "CREATE INDEX track_enriched IF NOT EXISTS FOR (t:Track) ON (t.enriched)",
        ]
        async with self._driver.session() as session:
            for stmt in constraints + indexes:
                try:
                    await session.run(stmt)
                except Exception as exc:
                    logger.debug("Schema stmt skipped (%s): %s", exc, stmt[:60])

    # ------------------------------------------------------------------
    # Track upsert
    # ------------------------------------------------------------------

    async def upsert_track(self, track: TrackInfo) -> None:
        af = track.audio_features or AudioFeatures()
        props = {
            "spotify_id": track.spotify_id,
            "name": track.name,
            "artist_names": track.artist_names,
            "genres": track.genres,
            "popularity": track.popularity,
            "release_year": track.release_year,
            "danceability": af.danceability,
            "energy": af.energy,
            "valence": af.valence,
            "tempo": af.tempo,
            "acousticness": af.acousticness,
            "instrumentalness": af.instrumentalness,
            "liveness": af.liveness,
            "speechiness": af.speechiness,
            "loudness": af.loudness,
            "key": af.key,
            "mode": af.mode,
        }

        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (t:Track {spotify_id: $spotify_id})
                ON CREATE SET t += $props, t.enriched = false
                ON MATCH  SET
                    t.name          = $props.name,
                    t.artist_names  = $props.artist_names,
                    t.genres        = $props.genres,
                    t.popularity    = $props.popularity,
                    t.release_year  = $props.release_year
                    // NOTE: audio features (tempo/loudness/energy/...) are
                    // deliberately NOT updated on match. Spotify no longer
                    // supplies them, so $props carries only zero defaults —
                    // overwriting here would clobber Deezer-enriched values
                    // on every re-upsert (and is_enriched then skips refetch).
                """,
                spotify_id=track.spotify_id,
                props=props,
            )

            for artist_id, artist_name in zip(track.artist_ids, track.artist_names):
                await session.run(
                    """
                    MERGE (a:Artist {name: $artist_name})
                    ON CREATE SET a.spotify_id = $artist_id
                    ON MATCH  SET a.spotify_id = $artist_id
                    WITH a
                    MATCH (t:Track {spotify_id: $track_id})
                    MERGE (t)-[:BY]->(a)
                    """,
                    artist_id=artist_id,
                    artist_name=artist_name,
                    track_id=track.spotify_id,
                )

            for genre in track.genres:
                await session.run(
                    """
                    MERGE (g:Genre {name: $genre})
                    WITH g
                    MATCH (t:Track {spotify_id: $track_id})
                    MERGE (t)-[:HAS_GENRE]->(g)
                    """,
                    genre=genre,
                    track_id=track.spotify_id,
                )

    # ------------------------------------------------------------------
    # Enrichment: attach semantic bucket nodes
    # ------------------------------------------------------------------

    async def apply_enrichment(self, track_id: str, result: EnrichmentResult) -> None:
        async with self._driver.session() as session:
            # Genre nodes (multiple, from Last.fm)
            for genre in result.genre_tags:
                await session.run(
                    """
                    MERGE (g:Genre {name: $name})
                    WITH g MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:HAS_GENRE]->(g)
                    """,
                    name=genre, tid=track_id,
                )

            # Mood nodes (multiple, from Last.fm)
            for mood in result.mood_tags:
                await session.run(
                    """
                    MERGE (m:Mood {name: $name})
                    WITH m MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:EVOKES]->(m)
                    """,
                    name=mood, tid=track_id,
                )

            # Era node (single, from release year)
            if result.era_label:
                await session.run(
                    """
                    MERGE (e:Era {label: $label})
                    WITH e MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:BELONGS_TO_ERA]->(e)
                    """,
                    label=result.era_label, tid=track_id,
                )

            # Deezer audio proxy (energy signal) — reuse existing node props.
            # Only overwrite when we actually got a value, so a failed Deezer
            # lookup never clobbers a previously stored proxy.
            if result.bpm is not None:
                await session.run(
                    "MATCH (t:Track {spotify_id: $tid}) SET t.tempo = $bpm",
                    tid=track_id, bpm=result.bpm,
                )
            if result.loudness is not None:
                await session.run(
                    "MATCH (t:Track {spotify_id: $tid}) SET t.loudness = $loudness",
                    tid=track_id, loudness=result.loudness,
                )

            # Energy bucket node (graph-native vibe dimension, from Deezer).
            bucket = _energy_bucket(result.loudness, result.bpm)
            if bucket:
                await session.run(
                    """
                    MERGE (en:Energy {name: $name})
                    WITH en MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:HAS_ENERGY]->(en)
                    """,
                    name=bucket, tid=track_id,
                )

            await session.run(
                "MATCH (t:Track {spotify_id: $tid}) SET t.enriched = true",
                tid=track_id,
            )

    # ------------------------------------------------------------------
    # Artist similarity edges (from Spotify Related Artists)
    # ------------------------------------------------------------------

    async def create_artist_similar_by_name(
        self, name1: str, name2: str, weight: float = 1.0
    ) -> None:
        """Create SIMILAR_TO edge between artists identified by name (from Last.fm)."""
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (a1:Artist {name: $name1})
                MERGE (a2:Artist {name: $name2})
                MERGE (a1)-[r:SIMILAR_TO]->(a2)
                ON CREATE SET r.weight = $weight
                """,
                name1=name1, name2=name2, weight=weight,
            )

    async def create_artist_similar(
        self,
        artist1_id: str,
        artist1_name: str,
        artist2_id: str,
        artist2_name: str,
        weight: float = 1.0,
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (a1:Artist {spotify_id: $id1}) ON CREATE SET a1.name = $name1
                MERGE (a2:Artist {spotify_id: $id2}) ON CREATE SET a2.name = $name2
                MERGE (a1)-[r:SIMILAR_TO]->(a2)
                ON CREATE SET r.weight = $weight
                """,
                id1=artist1_id, name1=artist1_name,
                id2=artist2_id, name2=artist2_name,
                weight=weight,
            )

    # ------------------------------------------------------------------
    # Sonic similarity edges (bootstrap from Spotify artist top tracks)
    # ------------------------------------------------------------------

    async def create_sonically_similar(
        self,
        source_id: str,
        target_id: str,
        score: float,
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (s:Track {spotify_id: $src})
                MATCH (t:Track {spotify_id: $tgt})
                MERGE (s)-[r:SONICALLY_SIMILAR]->(t)
                ON CREATE SET r.score = $score
                ON MATCH  SET r.score = CASE WHEN r.score < $score THEN $score ELSE r.score END
                """,
                src=source_id, tgt=target_id, score=score,
            )

    # ------------------------------------------------------------------
    # Enrichment status
    # ------------------------------------------------------------------

    async def is_enriched(self, track_id: str) -> bool:
        """True only if enriched AND has at least one Genre or Mood node connected."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Track {spotify_id: $id})
                OPTIONAL MATCH (t)-[:HAS_GENRE]->(g:Genre)
                OPTIONAL MATCH (t)-[:EVOKES]->(m:Mood)
                OPTIONAL MATCH (t)-[:BY]->(a:Artist)-[:SIMILAR_TO]-(:Artist)
                RETURN
                    t.enriched AS enriched,
                    count(DISTINCT g) + count(DISTINCT m) + count(DISTINCT a) AS semantic_count
                LIMIT 1
                """,
                id=track_id,
            )
            record = await result.single()
            if not record:
                return False
            return bool(record["enriched"]) and int(record["semantic_count"] or 0) > 0

    async def reset_all_enrichment(self) -> int:
        """Reset enriched flag on all tracks — forces re-enrichment next Recommend call."""
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (t:Track) SET t.enriched = false RETURN count(t) AS n"
            )
            record = await result.single()
            return int(record["n"]) if record else 0

    # ------------------------------------------------------------------
    # Candidate discovery via graph traversal
    # ------------------------------------------------------------------

    async def find_candidates(
        self,
        track_id: str,
        exclude_ids: list[str],
        candidate_limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Weighted spreading activation from the seed track.

        Each candidate accumulates activation over weighted graph paths:
          - shared Genre nodes        (Python weights by IDF)
          - shared Energy buckets      (Python weights by IDF) — Deezer vibe
          - shared Era                 (weak)
          - artist similarity          (SIMILAR_TO weight × hop decay, max path)
          - sonic edge                 (SONICALLY_SIMILAR.score)

        Edge weights (not just booleans) are returned so the scorer can combine
        them. artist_score / sonic_score use max() so the cross-product of the
        OPTIONAL MATCHes never inflates them.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (seed:Track {spotify_id: $track_id})

                // Seed's semantic labels (1-hop)
                OPTIONAL MATCH (seed)-[:HAS_GENRE]->(sg:Genre)
                OPTIONAL MATCH (seed)-[:HAS_ENERGY]->(sen:Energy)
                OPTIONAL MATCH (seed)-[:BELONGS_TO_ERA]->(se:Era)

                WITH seed,
                     seed.tempo                 AS seedBpm,
                     seed.loudness              AS seedLoudness,
                     collect(DISTINCT sg.name)  AS seedGenres,
                     collect(DISTINCT sen.name) AS seedEnergy,
                     collect(DISTINCT se.label) AS seedEras

                MATCH (candidate:Track)
                WHERE candidate.spotify_id <> $track_id
                  AND NOT candidate.spotify_id IN $exclude_ids

                OPTIONAL MATCH (candidate)-[:HAS_GENRE]->(cg:Genre)
                  WHERE cg.name IN seedGenres
                OPTIONAL MATCH (candidate)-[:HAS_ENERGY]->(cen:Energy)
                  WHERE cen.name IN seedEnergy
                OPTIONAL MATCH (candidate)-[:BELONGS_TO_ERA]->(ce:Era)
                  WHERE ce.label IN seedEras
                OPTIONAL MATCH (seed)-[ss:SONICALLY_SIMILAR]->(candidate)
                // Weighted artist-similarity path: product of SIMILAR_TO weights
                // divided by hop count (1 hop = full, 2 hops = halved).
                OPTIONAL MATCH (seed)-[:BY]->(:Artist)-[rels:SIMILAR_TO*1..2]-(:Artist)<-[:BY]-(candidate)

                WITH candidate, seedBpm, seedLoudness,
                     collect(DISTINCT cg.name)  AS sharedGenres,
                     collect(DISTINCT cen.name) AS sharedEnergy,
                     collect(DISTINCT ce.label) AS sharedEras,
                     max(coalesce(ss.score, 0.0)) AS sonicScore,
                     max(coalesce(
                         reduce(w = 1.0, r IN rels | w * coalesce(r.weight, 0.5)) / size(rels),
                         0.0)) AS artistScore

                WHERE size(sharedGenres) + size(sharedEnergy) + size(sharedEras) > 0
                   OR sonicScore > 0 OR artistScore > 0

                RETURN
                    candidate.spotify_id   AS spotify_id,
                    candidate.name         AS name,
                    candidate.artist_names AS artist_names,
                    candidate.genres       AS genres,
                    candidate.popularity   AS popularity,
                    candidate.tempo        AS cand_bpm,
                    candidate.loudness     AS cand_loudness,
                    seedBpm                AS seed_bpm,
                    seedLoudness           AS seed_loudness,
                    sharedGenres           AS shared_genres,
                    sharedEnergy           AS shared_energy,
                    sharedEras             AS shared_eras,
                    sonicScore             AS sonic_score,
                    artistScore            AS artist_score,
                    artistScore > 0        AS via_related_artist,
                    sonicScore > 0         AS direct_similar,
                    size(sharedGenres) + size(sharedEnergy) AS overlap_count
                ORDER BY overlap_count DESC, artistScore DESC, sonicScore DESC
                LIMIT $limit
                """,
                track_id=track_id,
                exclude_ids=list(exclude_ids),
                limit=candidate_limit,
            )
            return [dict(r) async for r in result]

    # ------------------------------------------------------------------
    # Audio features for scoring
    # ------------------------------------------------------------------

    async def get_genre_frequencies(self) -> dict[str, int]:
        """Return {genre_name: track_count} — used for TF-IDF genre weighting."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (g:Genre)<-[:HAS_GENRE]-(t:Track)
                RETURN g.name AS genre, count(t) AS freq
                """
            )
            return {r["genre"]: r["freq"] async for r in result}

    async def get_energy_frequencies(self) -> dict[str, int]:
        """Return {energy_bucket: track_count} — used for IDF weighting of energy."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (en:Energy)<-[:HAS_ENERGY]-(t:Track)
                RETURN en.name AS energy, count(t) AS freq
                """
            )
            return {r["energy"]: r["freq"] async for r in result}

    async def get_track_audio_features(self, track_id: str) -> Optional[dict[str, float]]:
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Track {spotify_id: $id})
                RETURN t.danceability AS danceability, t.energy AS energy,
                       t.valence AS valence, t.tempo AS tempo,
                       t.acousticness AS acousticness,
                       t.instrumentalness AS instrumentalness,
                       t.liveness AS liveness, t.speechiness AS speechiness,
                       t.loudness AS loudness
                LIMIT 1
                """,
                id=track_id,
            )
            record = await result.single()
            return dict(record) if record else None

    # ------------------------------------------------------------------
    # Stats + neighborhood (for dashboard)
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, int]:
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Track)  WITH count(t) AS tracks
                MATCH (a:Artist) WITH tracks, count(a) AS artists
                MATCH (g:Genre)  WITH tracks, artists, count(g) AS genres
                MATCH (m:Mood)   WITH tracks, artists, genres, count(m) AS moods
                RETURN tracks, artists, genres, moods
                """
            )
            record = await result.single()
            if not record:
                return {"tracks": 0, "artists": 0, "genres": 0, "moods": 0}
            return dict(record)

    async def export_graph(self) -> dict[str, Any]:
        """
        Export the whole graph for the GNN: every node with a stable string id
        (`t:`/`a:`/`g:`/`en:`/`er:` prefix) + node type + Track audio features,
        and every relevant edge as (source, target) id pairs.
        """
        def id_expr(v: str) -> str:
            return (
                f"CASE WHEN {v}:Track THEN 't:'+{v}.spotify_id "
                f"WHEN {v}:Artist THEN 'a:'+{v}.name "
                f"WHEN {v}:Genre THEN 'g:'+{v}.name "
                f"WHEN {v}:Energy THEN 'en:'+{v}.name "
                f"WHEN {v}:Era THEN 'er:'+{v}.label END"
            )
        async with self._driver.session() as session:
            node_res = await session.run(
                f"""
                MATCH (n)
                WHERE n:Track OR n:Artist OR n:Genre OR n:Energy OR n:Era
                RETURN {id_expr('n')} AS id, labels(n)[0] AS type,
                       n.spotify_id AS track_id,
                       coalesce(n.loudness, 0.0)  AS loudness,
                       coalesce(n.tempo, 0.0)     AS bpm,
                       coalesce(n.popularity, 0)  AS popularity
                """
            )
            nodes = [dict(r) async for r in node_res]

            edge_res = await session.run(
                f"""
                MATCH (a)-[r]->(b)
                WHERE (a:Track OR a:Artist OR a:Genre OR a:Energy OR a:Era)
                  AND (b:Track OR b:Artist OR b:Genre OR b:Energy OR b:Era)
                RETURN {id_expr('a')} AS source, {id_expr('b')} AS target
                """
            )
            edges = [dict(r) async for r in edge_res]

        return {"nodes": nodes, "edges": edges}

    async def get_neighborhood(self, track_id: str) -> dict[str, Any]:
        """
        Return nodes and edges for vis.js visualization.
        Shows the seed track + all semantic nodes + connected tracks.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (seed:Track {spotify_id: $track_id})

                OPTIONAL MATCH (seed)-[:BY]->(a:Artist)
                OPTIONAL MATCH (seed)-[:HAS_GENRE]->(g:Genre)
                OPTIONAL MATCH (seed)-[:EVOKES]->(m:Mood)
                OPTIONAL MATCH (seed)-[:BELONGS_TO_ERA]->(e:Era)
                OPTIONAL MATCH (a)-[:SIMILAR_TO]-(simA:Artist)
                OPTIONAL MATCH (simA)<-[:BY]-(t3:Track)
                  WHERE t3.spotify_id <> $track_id
                OPTIONAL MATCH (g)<-[:HAS_GENRE]-(t2:Track)
                  WHERE t2.spotify_id <> $track_id

                RETURN
                    seed,
                    collect(DISTINCT a)[0..5]    AS artists,
                    collect(DISTINCT g)[0..8]    AS genres,
                    collect(DISTINCT m)[0..5]    AS moods,
                    collect(DISTINCT e)[0..3]    AS eras,
                    collect(DISTINCT t2)[0..10]  AS genreTracks,
                    collect(DISTINCT simA)[0..8] AS simArtists,
                    collect(DISTINCT t3)[0..10]  AS simArtistTracks
                """,
                track_id=track_id,
            )
            record = await result.single()

        if not record:
            return {"nodes": [], "edges": []}

        nodes: list[dict] = []
        edges: list[dict] = []
        seen: set[str] = set()

        def node(nid: str, label: str, title: str, spotify_id: str = "") -> None:
            if nid and nid not in seen:
                seen.add(nid)
                nodes.append({"node_id": nid, "label": label, "title": title, "spotify_id": spotify_id})

        def edge(src: str, tgt: str, rel: str) -> None:
            if src and tgt:
                edges.append({"source": src, "target": tgt, "rel_type": rel})

        # Seed track
        node(track_id, "Track", record["seed"].get("name", ""), track_id)

        for a in record["artists"] or []:
            if a:
                aid = f"artist:{a.get('name','')}"
                node(aid, "Artist", a.get("name", ""))
                edge(track_id, aid, "BY")

        for g in record["genres"] or []:
            if g:
                gid = f"genre:{g.get('name','')}"
                node(gid, "Genre", g.get("name", ""))
                edge(track_id, gid, "HAS_GENRE")

                for t2 in record["genreTracks"] or []:
                    if t2:
                        t2id = t2.get("spotify_id", "")
                        node(t2id, "Track", t2.get("name", ""), t2id)
                        edge(t2id, gid, "HAS_GENRE")

        for m in record["moods"] or []:
            if m:
                mid = f"mood:{m.get('name','')}"
                node(mid, "Mood", m.get("name", ""))
                edge(track_id, mid, "EVOKES")

        for e in record["eras"] or []:
            if e:
                eid = f"era:{e.get('label','')}"
                node(eid, "Era", e.get("label", ""))
                edge(track_id, eid, "BELONGS_TO_ERA")

        for sa in record["simArtists"] or []:
            if sa:
                said = f"artist:{sa.get('name','')}"
                node(said, "Artist", sa.get("name", ""))
                for a in record["artists"] or []:
                    if a:
                        edge(f"artist:{a.get('name','')}", said, "SIMILAR_TO")

                for t3 in record["simArtistTracks"] or []:
                    if t3:
                        t3id = t3.get("spotify_id", "")
                        node(t3id, "Track", t3.get("name", ""), t3id)
                        edge(t3id, said, "BY")

        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Audio feature distance (Euclidean, normalized)
# ---------------------------------------------------------------------------

_FEATURE_KEYS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness", "speechiness",
]
_TEMPO_NORM = 200.0


def audio_distance(a: dict[str, float], b: dict[str, float]) -> float:
    diffs: list[float] = []
    for key in _FEATURE_KEYS:
        va = a.get(key, 0.0)
        vb = b.get(key, 0.0)
        if key == "tempo":
            va /= _TEMPO_NORM
            vb /= _TEMPO_NORM
        diffs.append((va - vb) ** 2)
    return math.sqrt(sum(diffs))

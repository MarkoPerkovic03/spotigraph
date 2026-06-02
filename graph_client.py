"""Neo4j Knowledge Graph client for SpotiGraph.

Schema
------
Nodes:
  Track    {spotify_id, name, artist_names, genres, popularity, enriched,
            release_year, danceability, energy, valence, tempo, acousticness,
            instrumentalness, liveness, speechiness, loudness, key, mode}
  Artist   {spotify_id, name}
  Genre    {name}
  Mood     {name}   -- energy × valence bucket  (e.g. "euphoric", "somber")
  Tempo    {name}   -- tempo × danceability bucket (e.g. "groovy", "fast-paced")
  Texture  {name}   -- acousticness × instrumentalness (e.g. "acoustic-vocal")
  Era      {label}  -- release decade (e.g. "1990s", "contemporary")

Relationships:
  (Track)-[:BY]->(Artist)
  (Track)-[:HAS_GENRE]->(Genre)
  (Track)-[:EVOKES]->(Mood)
  (Track)-[:HAS_TEMPO]->(Tempo)
  (Track)-[:HAS_TEXTURE]->(Texture)
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
            "CREATE CONSTRAINT artist_id  IF NOT EXISTS FOR (a:Artist)  REQUIRE a.spotify_id IS UNIQUE",
            "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre)   REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT mood_name  IF NOT EXISTS FOR (m:Mood)    REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT tempo_name IF NOT EXISTS FOR (t:Tempo)   REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT texture_name IF NOT EXISTS FOR (x:Texture) REQUIRE x.name IS UNIQUE",
            "CREATE CONSTRAINT era_label  IF NOT EXISTS FOR (e:Era)     REQUIRE e.label IS UNIQUE",
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
                    t.release_year  = $props.release_year,
                    t.danceability  = $props.danceability,
                    t.energy        = $props.energy,
                    t.valence       = $props.valence,
                    t.tempo         = $props.tempo,
                    t.acousticness  = $props.acousticness,
                    t.instrumentalness = $props.instrumentalness,
                    t.liveness      = $props.liveness,
                    t.speechiness   = $props.speechiness,
                    t.loudness      = $props.loudness,
                    t.key           = $props.key,
                    t.mode          = $props.mode
                """,
                spotify_id=track.spotify_id,
                props=props,
            )

            for artist_id, artist_name in zip(track.artist_ids, track.artist_names):
                await session.run(
                    """
                    MERGE (a:Artist {spotify_id: $artist_id})
                    ON CREATE SET a.name = $artist_name
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
            if result.mood_label:
                await session.run(
                    """
                    MERGE (m:Mood {name: $name})
                    WITH m MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:EVOKES]->(m)
                    """,
                    name=result.mood_label, tid=track_id,
                )

            if result.tempo_label:
                await session.run(
                    """
                    MERGE (tp:Tempo {name: $name})
                    WITH tp MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:HAS_TEMPO]->(tp)
                    """,
                    name=result.tempo_label, tid=track_id,
                )

            if result.texture_label:
                await session.run(
                    """
                    MERGE (tx:Texture {name: $name})
                    WITH tx MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:HAS_TEXTURE]->(tx)
                    """,
                    name=result.texture_label, tid=track_id,
                )

            if result.era_label:
                await session.run(
                    """
                    MERGE (e:Era {label: $label})
                    WITH e MATCH (t:Track {spotify_id: $tid})
                    MERGE (t)-[:BELONGS_TO_ERA]->(e)
                    """,
                    label=result.era_label, tid=track_id,
                )

            await session.run(
                "MATCH (t:Track {spotify_id: $tid}) SET t.enriched = true",
                tid=track_id,
            )

    # ------------------------------------------------------------------
    # Artist similarity edges (from Spotify Related Artists)
    # ------------------------------------------------------------------

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
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (t:Track {spotify_id: $id}) RETURN t.enriched AS e LIMIT 1",
                id=track_id,
            )
            record = await result.single()
            return bool(record and record["e"])

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
        Find candidate tracks by traversing shared semantic nodes
        (Mood, Tempo, Texture, Era, Genre) and artist similarity edges.

        Scoring signals returned per candidate:
          shared_moods, shared_tempos, shared_textures,
          shared_genres, shared_eras, via_related_artist
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (seed:Track {spotify_id: $track_id})

                // Collect seed's semantic labels
                OPTIONAL MATCH (seed)-[:EVOKES]->(sm:Mood)
                OPTIONAL MATCH (seed)-[:HAS_TEMPO]->(st:Tempo)
                OPTIONAL MATCH (seed)-[:HAS_TEXTURE]->(sx:Texture)
                OPTIONAL MATCH (seed)-[:BELONGS_TO_ERA]->(se:Era)
                OPTIONAL MATCH (seed)-[:HAS_GENRE]->(sg:Genre)
                OPTIONAL MATCH (seed)-[:BY]->(sa:Artist)-[:SIMILAR_TO]-(ra:Artist)

                WITH seed,
                     collect(DISTINCT sm.name)  AS seedMoods,
                     collect(DISTINCT st.name)  AS seedTempos,
                     collect(DISTINCT sx.name)  AS seedTextures,
                     collect(DISTINCT se.label) AS seedEras,
                     collect(DISTINCT sg.name)  AS seedGenres,
                     collect(DISTINCT ra.spotify_id) AS relArtistIds

                // Find all candidate tracks (not excluded)
                MATCH (candidate:Track)
                WHERE candidate.spotify_id <> $track_id
                  AND NOT candidate.spotify_id IN $exclude_ids

                // What does the candidate share with the seed?
                OPTIONAL MATCH (candidate)-[:EVOKES]->(cm:Mood)
                  WHERE cm.name IN seedMoods
                OPTIONAL MATCH (candidate)-[:HAS_TEMPO]->(ct:Tempo)
                  WHERE ct.name IN seedTempos
                OPTIONAL MATCH (candidate)-[:HAS_TEXTURE]->(cx:Texture)
                  WHERE cx.name IN seedTextures
                OPTIONAL MATCH (candidate)-[:BELONGS_TO_ERA]->(ce:Era)
                  WHERE ce.label IN seedEras
                OPTIONAL MATCH (candidate)-[:HAS_GENRE]->(cg:Genre)
                  WHERE cg.name IN seedGenres
                OPTIONAL MATCH (candidate)-[:BY]->(ca:Artist)
                  WHERE ca.spotify_id IN relArtistIds

                WITH candidate,
                     collect(DISTINCT cm.name)  AS sharedMoods,
                     collect(DISTINCT ct.name)  AS sharedTempos,
                     collect(DISTINCT cx.name)  AS sharedTextures,
                     collect(DISTINCT ce.label) AS sharedEras,
                     collect(DISTINCT cg.name)  AS sharedGenres,
                     count(DISTINCT ca)         AS relatedArtistHits

                WHERE size(sharedMoods) + size(sharedTempos) + size(sharedTextures)
                    + size(sharedGenres) + size(sharedEras) + relatedArtistHits > 0

                RETURN DISTINCT
                    candidate.spotify_id       AS spotify_id,
                    candidate.name             AS name,
                    candidate.artist_names     AS artist_names,
                    candidate.genres           AS genres,
                    candidate.danceability     AS danceability,
                    candidate.energy           AS energy,
                    candidate.valence          AS valence,
                    candidate.tempo            AS tempo,
                    candidate.acousticness     AS acousticness,
                    candidate.instrumentalness AS instrumentalness,
                    candidate.liveness         AS liveness,
                    candidate.speechiness      AS speechiness,
                    candidate.loudness         AS loudness,
                    sharedMoods                AS shared_moods,
                    sharedTempos               AS shared_tempos,
                    sharedTextures             AS shared_textures,
                    sharedEras                 AS shared_eras,
                    sharedGenres               AS shared_genres,
                    relatedArtistHits > 0      AS via_related_artist,
                    size(sharedMoods) + size(sharedTempos) + size(sharedTextures)
                        + size(sharedGenres) + size(sharedEras)
                        + relatedArtistHits    AS overlap_count
                ORDER BY overlap_count DESC
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
                MATCH (t:Track)   WITH count(t) AS tracks
                MATCH (a:Artist)  WITH tracks, count(a) AS artists
                MATCH (g:Genre)   WITH tracks, artists, count(g) AS genres
                MATCH (m:Mood)    WITH tracks, artists, genres, count(m) AS moods
                MATCH (tp:Tempo)  WITH tracks, artists, genres, moods, count(tp) AS tempos
                MATCH (tx:Texture) WITH tracks, artists, genres, moods, tempos, count(tx) AS textures
                RETURN tracks, artists, genres, moods, tempos, textures
                """
            )
            record = await result.single()
            if not record:
                return {"tracks": 0, "artists": 0, "genres": 0, "moods": 0, "tempos": 0, "textures": 0}
            return dict(record)

    async def get_neighborhood(self, track_id: str) -> dict[str, Any]:
        async with self._driver.session() as session:
            # Collect all nodes within 2 hops of the track
            result = await session.run(
                """
                MATCH (t:Track {spotify_id: $track_id})
                OPTIONAL MATCH (t)-[:BY]->(a:Artist)
                OPTIONAL MATCH (t)-[:HAS_GENRE]->(g:Genre)
                OPTIONAL MATCH (t)-[:EVOKES]->(m:Mood)
                OPTIONAL MATCH (t)-[:HAS_TEMPO]->(tp:Tempo)
                OPTIONAL MATCH (t)-[:HAS_TEXTURE]->(tx:Texture)
                OPTIONAL MATCH (t)-[:BELONGS_TO_ERA]->(e:Era)
                OPTIONAL MATCH (t2:Track)-[:BELONGS_TO_ERA]->(e)
                OPTIONAL MATCH (t2)-[:BY]->(a2:Artist)
                WITH t,
                     collect(DISTINCT a)  AS artists,
                     collect(DISTINCT g)  AS genres,
                     collect(DISTINCT m)  AS moods,
                     collect(DISTINCT tp) AS tempos,
                     collect(DISTINCT tx) AS textures,
                     collect(DISTINCT e)  AS eras,
                     collect(DISTINCT t2)[..10] AS relTracks,
                     collect(DISTINCT a2)[..10] AS relArtists
                RETURN t, artists, genres, moods, tempos, textures, eras, relTracks, relArtists
                """,
                track_id=track_id,
            )
            record = await result.single()
            if not record:
                return {"nodes": [], "edges": []}

        nodes: list[dict] = []
        edges: list[dict] = []
        seen_ids: set[str] = set()

        def add_node(node_id: str, label: str, title: str, spotify_id: str = "") -> None:
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                nodes.append({"node_id": node_id, "label": label, "title": title, "spotify_id": spotify_id})

        def add_edge(src: str, tgt: str, rel_type: str) -> None:
            edges.append({"source": src, "target": tgt, "rel_type": rel_type})

        t = record["t"]
        t_id = track_id
        add_node(t_id, "Track", t.get("name", ""), t_id)

        for a in record["artists"] or []:
            if a:
                aid = a.get("spotify_id") or a.get("name", "")
                add_node(aid, "Artist", a.get("name", ""), aid)
                add_edge(t_id, aid, "BY")

        for g in record["genres"] or []:
            if g:
                gid = f"genre:{g.get('name','')}"
                add_node(gid, "Genre", g.get("name", ""))
                add_edge(t_id, gid, "HAS_GENRE")

        for m in record["moods"] or []:
            if m:
                mid = f"mood:{m.get('name','')}"
                add_node(mid, "Mood", m.get("name", ""))
                add_edge(t_id, mid, "EVOKES")

        for tp in record["tempos"] or []:
            if tp:
                tpid = f"tempo:{tp.get('name','')}"
                add_node(tpid, "Tempo", tp.get("name", ""))
                add_edge(t_id, tpid, "HAS_TEMPO")

        for tx in record["textures"] or []:
            if tx:
                txid = f"texture:{tx.get('name','')}"
                add_node(txid, "Texture", tx.get("name", ""))
                add_edge(t_id, txid, "HAS_TEXTURE")

        for e in record["eras"] or []:
            if e:
                eid = f"era:{e.get('label','')}"
                add_node(eid, "Era", e.get("label", ""))
                add_edge(t_id, eid, "BELONGS_TO_ERA")

                for t2 in record["relTracks"] or []:
                    if t2 and t2.get("spotify_id") != track_id:
                        t2id = t2.get("spotify_id", "")
                        add_node(t2id, "Track", t2.get("name", ""), t2id)
                        add_edge(t2id, eid, "BELONGS_TO_ERA")

        for a2 in record["relArtists"] or []:
            if a2:
                a2id = a2.get("spotify_id") or a2.get("name", "")
                add_node(a2id, "Artist", a2.get("name", ""), a2id)

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

# SpotiGraph

A Knowledge-Graph-powered Spotify queue filler. When you're playing a song, SpotiGraph traverses a semantic graph of musical relationships to find the best tracks to queue next — no black-box embeddings, just traversable, inspectable graph paths.

---

## Architecture

```
Spotify API ──► spotify_client.py ──► recommender.py ──► Spotify Queue
                      │                     │
                      ▼                     ▼
             AudioFeatures          Neo4j (graph_client.py)
                      │                     ▲
                      └──► enrichment.py ───┘
                           (Claude API)
```

### Graph Schema

| Node type  | Key property | Description |
|------------|-------------|-------------|
| `Track`    | `spotify_id` | A Spotify track with audio features as node properties |
| `Artist`   | `spotify_id` | Artist; connected to tracks via `BY` |
| `Genre`    | `name`       | Genre string from Spotify artist data |
| `Mood`     | `name`       | Claude-generated mood descriptor (e.g. "melancholic") |
| `Concept`  | `name`       | Abstract musical concept (e.g. "driving rhythm") |
| `Era`      | `label`      | Era/decade feel (e.g. "1990s alternative") |

| Relationship | Description |
|---|---|
| `(Track)-[:BY]->(Artist)` | Authorship |
| `(Track)-[:HAS_GENRE]->(Genre)` | Genre membership |
| `(Track)-[:EVOKES]->(Mood)` | Emotional character |
| `(Track)-[:SHARES_CONCEPT]->(Concept)` | Abstract musical concepts |
| `(Track)-[:BELONGS_TO_ERA]->(Era)` | Era/decade feel |
| `(Artist)-[:SIMILAR_TO {weight}]->(Artist)` | *(reserved for future expansion)* |
| `(Track)-[:SONICALLY_SIMILAR {score}]->(Track)` | *(reserved for future expansion)* |

### AI Enrichment Step

For each new track, `enrichment.py` calls Claude (`claude-sonnet-4-6`) with:
- Track name, artist, album, genres
- All Spotify Audio Features (danceability, energy, valence, tempo, etc.)

Claude returns structured JSON:
```json
{
  "mood_tags": ["melancholic", "introspective", "tense"],
  "concept_tags": ["driving rhythm", "nocturnal atmosphere", "cathartic release"],
  "era_feel": "1990s alternative",
  "similar_artist_archetypes": ["brooding post-punk vocalist"]
}
```

These become `Mood`, `Concept`, and `Era` nodes in Neo4j linked to the track. The `enriched: true` flag on the Track node prevents duplicate API calls.

### Recommendation Scoring

Candidates are found via 2-6 hop traversal through semantic nodes, then scored:

```
score = shared_overlap × 2.0
      − hop_distance  × 0.5
      − audio_distance × 1.5
```

where `audio_distance` is Euclidean distance over normalized audio features (danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness).

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker Desktop
- Spotify Developer App — [create one here](https://developer.spotify.com/dashboard)
- Anthropic API key

### 2. Environment

```bash
cd spotigraph
cp .env.example .env
# Edit .env with your keys:
# SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, ANTHROPIC_API_KEY
```

In your Spotify App settings, add `http://localhost:8000/callback` as a Redirect URI.

### 3. Start Neo4j

```bash
docker compose up -d
# Wait ~20s for Neo4j to initialize
# Browser UI: http://localhost:7474  (neo4j / spotigraph123)
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run SpotiGraph

```bash
python main.py
```

Open **http://localhost:8000** → click **Connect Spotify** → authorize → you're live.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health (Neo4j + Spotify auth status) |
| `GET` | `/login` | Start Spotify OAuth flow |
| `GET` | `/callback` | OAuth callback (Spotify redirects here) |
| `GET` | `/currently-playing` | Currently playing track |
| `GET` | `/recommend?limit=N` | Run recommender, return JSON |
| `POST` | `/recommend/queue?limit=N` | Recommend + add to Spotify queue |
| `POST` | `/ingest/{track_id}` | Manually ingest a track into the graph |
| `GET` | `/graph/neighborhood/{id}` | Subgraph JSON for vis.js |
| `GET` | `/graph/stats` | Node/edge counts |
| `GET` | `/dashboard` | Web UI |

### Example `/recommend` response

```json
{
  "source_track": { "spotify_id": "...", "name": "Creep", "artist_names": ["Radiohead"], ... },
  "recommendations": [
    {
      "track": { "spotify_id": "...", "name": "Karma Police", "artist_names": ["Radiohead"] },
      "reason": {
        "hop_distance": 1,
        "shared_moods": ["melancholic", "introspective"],
        "shared_concepts": ["driving rhythm"],
        "shared_genres": ["alternative rock"],
        "audio_distance": 0.142,
        "score": 5.787
      }
    }
  ],
  "added_to_queue": false
}
```

---

## Continuous Mode

After OAuth, a background asyncio task polls every 30 seconds. When the track changes and the queue drops below 3 tracks, it automatically runs the recommender and adds 3 tracks to the queue. No action required from you.

---

## Neo4j Browser

Open **http://localhost:7474** (credentials: `neo4j` / `spotigraph123`).

Useful Cypher queries:

```cypher
// See all tracks and their moods
MATCH (t:Track)-[:EVOKES]->(m:Mood)
RETURN t.name, collect(m.name) AS moods
LIMIT 20

// Find tracks that share concepts
MATCH (t1:Track)-[:SHARES_CONCEPT]->(c:Concept)<-[:SHARES_CONCEPT]-(t2:Track)
WHERE t1 <> t2
RETURN t1.name, c.name, t2.name
LIMIT 30

// Full neighborhood of a track
MATCH path = (:Track {name: "Creep"})-[*1..3]-()
RETURN path LIMIT 50
```

---

## File Structure

```
spotigraph/
├── main.py              FastAPI app, routes, background polling loop
├── spotify_client.py    Spotify OAuth (PKCE) + API wrapper + rate-limit backoff
├── graph_client.py      Neo4j driver, schema setup, Cypher queries
├── enrichment.py        Claude API calls → structured mood/concept JSON
├── recommender.py       Graph traversal + scoring pipeline
├── models.py            Pydantic models (TrackInfo, Recommendation, etc.)
├── docker-compose.yml   Neo4j 5.x + APOC plugin
├── requirements.txt
├── .env.example
└── frontend/
    └── index.html       vis.js Knowledge Graph dashboard
```

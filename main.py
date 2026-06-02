"""SpotiGraph — FastAPI application entry point.

Routes
------
GET  /                       Redirect to dashboard
GET  /health                 Health check (Neo4j + Spotify auth status)
GET  /login                  Redirect to Spotify OAuth
GET  /callback               Spotify OAuth callback
GET  /currently-playing      Currently playing track
GET  /recommend?limit=N      Run recommender, return JSON
POST /recommend/queue?limit=N Run recommender + add to Spotify queue
POST /ingest/{track_id}      Manually ingest a track into the graph
GET  /graph/neighborhood/{id} Subgraph for vis.js dashboard
GET  /graph/stats            Node/edge counts
GET  /dashboard              Serve frontend/index.html
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings

from graph_client import GraphClient
from lastfm_client import LastFmClient
from models import HealthResponse, RecommendationResponse
from recommender import Recommender
from spotify_client import SpotifyClient

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("spotigraph")


# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    app_secret_key: str = "dev_secret_change_in_prod"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    polling_interval_seconds: int = 30
    queue_low_threshold: int = 3

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = AppSettings()

# ---------------------------------------------------------------------------
# Shared service instances (created once at startup)
# ---------------------------------------------------------------------------

graph: Optional[GraphClient] = None
spotify: Optional[SpotifyClient] = None
lastfm: Optional[LastFmClient] = None
recommender_svc: Optional[Recommender] = None
_http_client_ctx = None       # keeps the SpotifyClient context open
_polling_task: Optional[asyncio.Task] = None
_last_track_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, spotify, lastfm, recommender_svc, _http_client_ctx, _polling_task

    # Boot services
    graph = GraphClient()
    connected = await graph.verify_connectivity()
    if connected:
        await graph.ensure_schema()
        logger.info("Neo4j connected and schema ready")
    else:
        logger.warning("Neo4j not reachable — graph features will be unavailable")

    # SpotifyClient lives as an async context manager
    spotify = SpotifyClient()
    _http_client_ctx = spotify
    await spotify.__aenter__()

    lastfm = LastFmClient()
    await lastfm.__aenter__()
    recommender_svc = Recommender(graph=graph, spotify=spotify, lastfm=lastfm)

    logger.info("SpotiGraph started on http://%s:%d", settings.app_host, settings.app_port)

    yield   # <-- app runs here

    # Shutdown
    if _polling_task and not _polling_task.done():
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass

    if spotify:
        await spotify.__aexit__(None, None, None)
    if lastfm:
        await lastfm.__aexit__(None, None, None)
    if graph:
        await graph.close()
    logger.info("SpotiGraph shut down cleanly")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpotiGraph",
    description="Knowledge-Graph-powered Spotify queue filler",
    version="0.1.0",
    lifespan=lifespan,
)

# Serve static files (frontend) if the directory exists
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    neo4j_status = "ok" if (graph and await graph.verify_connectivity()) else "unreachable"
    return HealthResponse(
        status="ok",
        neo4j=neo4j_status,
        spotify_authenticated=spotify.is_authenticated() if spotify else False,
    )


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------

@app.get("/login", tags=["Auth"])
async def login():
    """Redirect the user to Spotify's authorization page."""
    if spotify is None:
        raise HTTPException(503, "Spotify client not initialized")
    auth_url, _state = spotify.build_auth_url()
    return RedirectResponse(auth_url)


@app.get("/callback", tags=["Auth"])
async def callback(
    code: str = Query(...),
    state: str = Query(...),
    error: Optional[str] = Query(None),
):
    """Spotify OAuth callback — exchanges code for tokens and starts polling."""
    global _polling_task

    if error:
        raise HTTPException(400, f"Spotify authorization error: {error}")
    if spotify is None:
        raise HTTPException(503, "Spotify client not initialized")

    token = await spotify.exchange_code(code=code, state=state)
    logger.info("OAuth complete — scopes: %s", token.scope)

    # Start background polling if not already running
    if _polling_task is None or _polling_task.done():
        _polling_task = asyncio.create_task(_polling_loop())
        logger.info("Background polling loop started (interval=%ds)", settings.polling_interval_seconds)

    return RedirectResponse("/dashboard")


# ---------------------------------------------------------------------------
# Currently playing
# ---------------------------------------------------------------------------

@app.get("/currently-playing", tags=["Spotify"])
async def currently_playing():
    if spotify is None:
        raise HTTPException(503, "Spotify client not initialized")
    if not spotify.is_authenticated():
        raise HTTPException(401, "Not authenticated — visit /login first")

    result = await spotify.get_currently_playing()
    if result is None:
        return {"playing": False, "track": None}
    return {"playing": result.is_playing, "track": result.track}


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@app.get("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(limit: int = Query(5, ge=1, le=20)):
    """Run the recommendation engine and return top N tracks."""
    _check_ready()
    result = await recommender_svc.recommend(limit=limit, add_to_queue=False)
    if result is None:
        raise HTTPException(404, "No track currently playing on Spotify")
    return result


@app.post("/recommend/queue", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_and_queue(limit: int = Query(5, ge=1, le=20)):
    """Run the recommendation engine and add top N tracks to the Spotify queue."""
    _check_ready()
    result = await recommender_svc.recommend(limit=limit, add_to_queue=True)
    if result is None:
        raise HTTPException(404, "No track currently playing on Spotify")
    return result


# ---------------------------------------------------------------------------
# Track ingestion
# ---------------------------------------------------------------------------

@app.post("/ingest/{track_id}", tags=["Graph"])
async def ingest_track(track_id: str):
    """Manually ingest and enrich a Spotify track into the Knowledge Graph."""
    _check_ready()
    track = await recommender_svc.ingest_track_by_id(track_id)
    if track is None:
        raise HTTPException(404, f"Track {track_id} not found on Spotify")
    return {"status": "ingested", "track": track}


# ---------------------------------------------------------------------------
# Graph endpoints (for dashboard)
# ---------------------------------------------------------------------------

@app.get("/graph/neighborhood/{track_id}", tags=["Graph"])
async def graph_neighborhood(track_id: str):
    """Return the local subgraph around a track for vis.js visualization."""
    _check_ready()
    return await graph.get_neighborhood(track_id)


@app.get("/graph/stats", tags=["Graph"])
async def graph_stats():
    """Return node/edge counts for the dashboard."""
    _check_ready()
    return await graph.get_stats()


# ---------------------------------------------------------------------------
# Admin / debug endpoints
# ---------------------------------------------------------------------------

@app.get("/admin/reset-enrichment", tags=["Admin"])
async def reset_enrichment():
    """Reset enriched flag on all tracks so they get re-enriched on next Recommend."""
    _check_ready()
    n = await graph.reset_all_enrichment()
    return {"reset": n, "message": f"{n} tracks will be re-enriched on next Recommend"}


@app.get("/admin/test-lastfm/{artist_name}", tags=["Admin"])
async def test_lastfm(artist_name: str):
    """Test what Last.fm returns for an artist name."""
    if lastfm is None:
        raise HTTPException(503, "Last.fm client not initialized")
    tags = await lastfm.get_artist_tags(artist_name)
    similar = await lastfm.get_similar_artists(artist_name, limit=5)
    return {"artist": artist_name, "tags": tags, "similar_artists": similar}


# ---------------------------------------------------------------------------
# Dashboard (frontend)
# ---------------------------------------------------------------------------

@app.get("/", tags=["Frontend"])
async def root():
    return RedirectResponse("/dashboard")


@app.get("/dashboard", tags=["Frontend"], response_class=HTMLResponse)
async def dashboard():
    index_path = _frontend_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>SpotiGraph</h1><p>Frontend not found. Check frontend/index.html.</p>")
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

async def _polling_loop() -> None:
    """
    Every POLLING_INTERVAL_SECONDS:
    - Check what's currently playing.
    - If track changed, trigger recommendation + queue fill.
    - If queue is running low (< QUEUE_LOW_THRESHOLD), add more.
    """
    global _last_track_id

    logger.info("Polling loop active")
    while True:
        try:
            await asyncio.sleep(settings.polling_interval_seconds)

            if spotify is None or not spotify.is_authenticated():
                continue

            now_playing = await spotify.get_currently_playing()
            if now_playing is None or not now_playing.is_playing:
                continue

            current_id = now_playing.track.spotify_id
            track_changed = current_id != _last_track_id

            if track_changed:
                logger.info(
                    "Track changed: '%s' — running recommender",
                    now_playing.track.name,
                )
                _last_track_id = current_id

                queue = await spotify.get_queue()
                if len(queue) < settings.queue_low_threshold or track_changed:
                    result = await recommender_svc.recommend(
                        limit=3,
                        add_to_queue=True,
                    )
                    if result and result.recommendations:
                        names = [r.track.name for r in result.recommendations]
                        logger.info("Auto-queued: %s", ", ".join(names))

        except asyncio.CancelledError:
            logger.info("Polling loop cancelled")
            return
        except Exception as exc:
            logger.exception("Polling loop error (will retry): %s", exc)


# ---------------------------------------------------------------------------
# Guard helper
# ---------------------------------------------------------------------------

def _check_ready() -> None:
    if spotify is None or graph is None or recommender_svc is None:
        raise HTTPException(503, "Services not initialized")
    if not spotify.is_authenticated():
        raise HTTPException(401, "Not authenticated with Spotify — visit /login")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cert = Path(__file__).parent / "cert.pem"
    key  = Path(__file__).parent / "key.pem"
    ssl_kwargs = {}
    if cert.exists() and key.exists():
        ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        logger.info("HTTPS enabled — using %s", cert)
    else:
        logger.warning("cert.pem / key.pem not found — running HTTP. Run gen_certs.py first.")

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,   # reload=True is incompatible with ssl in uvicorn
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        **ssl_kwargs,
    )

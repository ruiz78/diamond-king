"""
Diamond King — FastAPI Backend (Windows-compatible)
Run from inside the Diamond King folder:
    python -m uvicorn api.server:app --reload --port 8000
"""

import json
import os
import sys
from datetime import date, datetime
from typing import Optional

# Fix paths so imports work on Windows regardless of where you run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Diamond King API",
    description="Baseball prediction engine — projections, edges, and prop finder",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_slate(game_date: str) -> dict:
    path   = os.path.join(DATA_DIR, f"projections_{game_date}.json")
    latest = os.path.join(DATA_DIR, "projections_latest.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    elif os.path.exists(latest):
        with open(latest) as f:
            return json.load(f)
    return None


@app.get("/")
def health():
    return {
        "app":     "Diamond King",
        "status":  "running 👑",
        "version": "1.0.0",
        "date":    date.today().isoformat(),
    }


@app.get("/slate")
def get_todays_slate():
    today = date.today().isoformat()
    data  = load_slate(today)
    if not data:
        return {
            "message": "No projections yet for today. Run the midnight engine first.",
            "hint":    "python pipeline/midnight_engine.py",
            "date":    today,
        }
    return data


@app.get("/slate/{game_date}")
def get_slate(game_date: str):
    data = load_slate(game_date)
    if not data:
        raise HTTPException(404, f"No data for {game_date}. Run: python pipeline/midnight_engine.py")
    return data


@app.get("/games")
def get_games(game_date: Optional[str] = Query(None)):
    d    = game_date or date.today().isoformat()
    data = load_slate(d)
    if not data:
        raise HTTPException(404, "No game data. Run midnight engine first.")
    return {"date": d, "count": len(data["games"]), "games": data["games"]}


@app.get("/projections/pitchers")
def get_pitcher_projections(game_date: Optional[str] = Query(None)):
    d    = game_date or date.today().isoformat()
    data = load_slate(d)
    if not data:
        raise HTTPException(404, "No pitcher data. Run midnight engine first.")
    pitchers = sorted(data.get("pitcher_projections", []),
                      key=lambda x: x.get("proj_k", 0), reverse=True)
    return {"date": d, "count": len(pitchers), "pitchers": pitchers}


@app.get("/projections/batters")
def get_batter_projections(
    game_date: Optional[str] = Query(None),
    stat: Optional[str] = Query("hits"),
):
    d    = game_date or date.today().isoformat()
    data = load_slate(d)
    if not data:
        raise HTTPException(404, "No batter data. Run midnight engine first.")
    sort_map = {"hr": "proj_hr", "hits": "proj_hits", "runs": "proj_runs", "rbi": "proj_rbi"}
    key      = sort_map.get(stat, "proj_hits")
    batters  = sorted(data.get("batter_projections", []),
                      key=lambda x: x.get(key, 0), reverse=True)
    return {"date": d, "sort_by": stat, "count": len(batters), "batters": batters}


@app.get("/edges")
def get_edges(
    game_date: Optional[str] = Query(None),
    top_n:     int = Query(20),
):
    d    = game_date or date.today().isoformat()
    data = load_slate(d)
    if not data:
        raise HTTPException(404, "No edge data. Run midnight engine first.")
    edges = data.get("edges", [])
    return {"date": d, "total": len(edges), "top_edges": edges[:top_n]}


@app.get("/edges/royal")
def get_royal_picks(game_date: Optional[str] = Query(None)):
    d    = game_date or date.today().isoformat()
    data = load_slate(d)
    if not data:
        raise HTTPException(404, "No data. Run midnight engine first.")
    royal = [e for e in data.get("edges", [])
             if "ROYAL" in e.get("rating", "") or "HIGH" in e.get("rating", "")]
    return {"date": d, "royal_picks": royal, "count": len(royal)}


@app.post("/engine/run")
def trigger_engine(background_tasks: BackgroundTasks,
                   game_date: Optional[str] = Query(None)):
    from pipeline.midnight_engine import run_midnight_engine
    d = game_date or date.today().isoformat()
    background_tasks.add_task(run_midnight_engine, d, True)
    return {"status": "Engine triggered", "date": d}


@app.get("/player/{player_name}")
def get_player(player_name: str, game_date: Optional[str] = Query(None)):
    d    = game_date or date.today().isoformat()
    data = load_slate(d)
    if not data:
        raise HTTPException(404, "No data. Run midnight engine first.")
    name    = player_name.lower()
    pitcher = next((p for p in data.get("pitcher_projections", [])
                    if name in p.get("player_name", "").lower()), None)
    batter  = next((b for b in data.get("batter_projections", [])
                    if name in b.get("player_name", "").lower()), None)
    edges   = [e for e in data.get("edges", [])
               if name in e.get("player_name", "").lower()]
    if not pitcher and not batter:
        raise HTTPException(404, f"'{player_name}' not found in today's slate.")
    return {"date": d, "pitcher_proj": pitcher, "batter_proj": batter, "edges": edges}

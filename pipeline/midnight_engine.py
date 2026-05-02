"""
Diamond King — Midnight Cron Engine (Windows-compatible)
Run from the Diamond King folder:
    cd pipeline
    python midnight_engine.py
"""

import json
import os
import sys
import time
from datetime import date, datetime

# Fix imports — works on Windows from any directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.mlb_fetcher import build_todays_slate, get_umpire_k_modifier
from models.prediction_models import PitcherKModel, BatterProjectionModel
from pipeline.edge_calculator import fetch_mlb_props, find_edges

OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAYS_UMPIRES = {}

def get_ump_for_game(game_pk):
    return TODAYS_UMPIRES.get(game_pk, "Unknown")


def run_midnight_engine(game_date=None, verbose=True):
    start_time = time.time()
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    log = []
    def _log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        log.append(line)
        if verbose:
            print(line)

    _log(f"👑 Diamond King Midnight Engine starting for {game_date}")
    _log("=" * 55)

    # Step 1: Fetch Slate
    _log("📡 Step 1/6: Fetching today's games & lineups...")
    try:
        slate    = build_todays_slate(game_date)
        games    = slate["games"]
        pitchers = slate["pitchers"]
        batters  = slate["batters"]
        _log(f"   ✅ {len(games)} games | {len(pitchers)} pitchers | {len(batters)} batters")
    except Exception as e:
        _log(f"   ❌ Failed to build slate: {e}")
        return None

    if not games:
        _log("   ⚠️  No games found for today.")
        return {"date": game_date, "games": [], "pitchers": [], "batters": [], "edges": []}

    # Step 2: Load Models
    _log("🤖 Step 2/6: Loading ML models...")
    try:
        pitcher_model = PitcherKModel.load()
        batter_model  = BatterProjectionModel.load()
        _log("   ✅ Models loaded")
    except Exception as e:
        _log(f"   ⚠️  Training fresh models: {e}")
        pitcher_model = PitcherKModel()
        pitcher_model.train(verbose=False)
        batter_model = BatterProjectionModel()
        batter_model.train(verbose=False)

    # Step 3: Pitcher Projections
    _log("⚾ Step 3/6: Running pitcher K projections...")
    pitcher_projections = []
    for p in pitchers:
        try:
            ump_name = get_ump_for_game(p["game_pk"])
            ump_mod  = get_umpire_k_modifier(ump_name)
            proj     = pitcher_model.predict(p, umpire_k_mod=ump_mod)
            pitcher_projections.append({**p, **proj, "ump_name": ump_name, "ump_mod": ump_mod})
        except Exception as e:
            _log(f"   ⚠️  Failed: {p.get('player_name')}: {e}")
    _log(f"   ✅ {len(pitcher_projections)} pitcher projections complete")

    # Step 4: Batter Projections
    _log("🏏 Step 4/6: Running batter projections...")
    batter_projections = []
    for b in batters:
        try:
            proj = batter_model.predict(b)
            batter_projections.append({**b, **proj})
        except Exception as e:
            _log(f"   ⚠️  Failed: {b.get('player_name')}: {e}")
    _log(f"   ✅ {len(batter_projections)} batter projections complete")

    # Step 5: Fetch Props
    _log("💰 Step 5/6: Fetching sportsbook prop lines...")
    try:
        props = fetch_mlb_props()
        _log(f"   ✅ {len(props)} props fetched")
    except Exception as e:
        _log(f"   ⚠️  Props fetch failed: {e}")
        props = []

    # Step 6: Calculate Edges
    _log("🎯 Step 6/6: Calculating edges...")
    edges = []
    if props:
        try:
            edges = find_edges(pitcher_projections, batter_projections, props)
            royal = [e for e in edges if "ROYAL" in e["rating"] or "HIGH" in e["rating"]]
            _log(f"   ✅ {len(edges)} edges | 👑 {len(royal)} Royal/High")
        except Exception as e:
            _log(f"   ⚠️  Edge calc failed: {e}")

    elapsed = round(time.time() - start_time, 1)
    payload = {
        "meta": {
            "date": game_date,
            "generated": datetime.now().isoformat(),
            "elapsed_sec": elapsed,
            "game_count": len(games),
            "pitcher_count": len(pitcher_projections),
            "batter_count": len(batter_projections),
            "edge_count": len(edges),
        },
        "games": games,
        "pitcher_projections": pitcher_projections,
        "batter_projections": batter_projections,
        "edges": edges,
        "top_edges": edges[:10],
        "log": log,
    }

    out_path    = os.path.join(OUTPUT_DIR, f"projections_{game_date}.json")
    latest_path = os.path.join(OUTPUT_DIR, "projections_latest.json")
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        with open(latest_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        _log(f"💾 Saved to data/projections_{game_date}.json")
    except Exception as e:
        _log(f"⚠️  Save failed: {e}")

    _log("=" * 55)
    _log(f"✅ Done in {elapsed}s — {len(games)} games | {len(pitcher_projections)} pitchers | {len(batter_projections)} batters | {len(edges)} edges")

    if edges:
        top = edges[0]
        _log(f"\n👑 TOP PICK: {top['player_name']} — {top['bet_type']} {top['line']} {top['stat_label']}")
        _log(f"   Win Prob: {top['win_prob_pct']}  |  Edge: {top['edge_pct']}  |  {top['book']}")

    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    run_midnight_engine(game_date=args.date, verbose=True)

"""
Diamond King — Prop Edge Calculator
Compares model projections vs sportsbook lines to surface the highest-edge plays.

Edge = (Model Win Probability) - (Book Implied Probability)
Positive edge = our model thinks this bet is underpriced.
"""

import requests
import json
import os
from datetime import date
from typing import Optional
import numpy as np

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_KEY_HERE")
ODDS_BASE    = "https://api.the-odds-api.com/v4"

# Minimum edge threshold to surface as a "Royal Pick"
MIN_EDGE_THRESHOLD = 0.08  # 8% edge minimum


# ─────────────────────────────────────────
#  ODDS FETCHER
# ─────────────────────────────────────────

def fetch_mlb_props(markets: list[str] = None) -> list[dict]:
    """
    Fetches MLB player prop lines from The Odds API.
    markets: list of prop markets to fetch
      e.g. ["batter_hits", "batter_home_runs", "pitcher_strikeouts"]
    Returns raw odds data per player.
    """
    if markets is None:
        markets = [
            "batter_hits",
            "batter_home_runs",
            "batter_rbis",
            "batter_runs_scored",
            "pitcher_strikeouts",
        ]

    if ODDS_API_KEY == "YOUR_KEY_HERE":
        print("[edge_calc] ⚠️  No ODDS_API_KEY set — using mock data")
        return _mock_props()

    props = []
    for market in markets:
        url = f"{ODDS_BASE}/sports/baseball_mlb/events"
        try:
            # First get today's event IDs
            r = requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=10)
            r.raise_for_status()
            events = r.json()

            for event in events[:5]:  # limit to avoid burning quota
                event_id = event["id"]
                odds_url = f"{ODDS_BASE}/sports/baseball_mlb/events/{event_id}/odds"
                params = {
                    "apiKey":  ODDS_API_KEY,
                    "markets": market,
                    "regions": "us",
                    "oddsFormat": "american",
                }
                or_ = requests.get(odds_url, params=params, timeout=10)
                or_.raise_for_status()
                props.extend(_parse_odds_response(or_.json(), market))

        except Exception as e:
            print(f"[edge_calc] Error fetching {market}: {e}")

    return props


def _parse_odds_response(data: dict, market: str) -> list[dict]:
    """Parse Odds API response into normalized prop dicts."""
    props = []
    for book in data.get("bookmakers", []):
        book_name = book.get("title", "Unknown")
        for mkt in book.get("markets", []):
            if mkt.get("key") != market:
                continue
            for outcome in mkt.get("outcomes", []):
                props.append({
                    "player_name": outcome.get("description", ""),
                    "market":      market,
                    "book":        book_name,
                    "bet_type":    outcome.get("name", ""),   # "Over" or "Under"
                    "line":        float(outcome.get("point", 0)),
                    "odds":        int(outcome.get("price", -110)),
                })
    return props


def _mock_props() -> list[dict]:
    """Mock prop data for development without API key."""
    return [
        # Pitcher Ks
        {"player_name": "Gerrit Cole",    "market": "pitcher_strikeouts",
         "book": "DraftKings", "bet_type": "Over",  "line": 7.5,  "odds": -115},
        {"player_name": "Gerrit Cole",    "market": "pitcher_strikeouts",
         "book": "DraftKings", "bet_type": "Under", "line": 7.5,  "odds": -105},
        {"player_name": "Dylan Cease",    "market": "pitcher_strikeouts",
         "book": "FanDuel",    "bet_type": "Over",  "line": 6.5,  "odds": -120},
        {"player_name": "Dylan Cease",    "market": "pitcher_strikeouts",
         "book": "FanDuel",    "bet_type": "Under", "line": 6.5,  "odds": +100},
        {"player_name": "Logan Webb",     "market": "pitcher_strikeouts",
         "book": "BetMGM",     "bet_type": "Over",  "line": 5.5,  "odds": -110},
        # Batter Hits
        {"player_name": "Freddie Freeman","market": "batter_hits",
         "book": "DraftKings", "bet_type": "Over",  "line": 1.5,  "odds": -130},
        {"player_name": "Shohei Ohtani",  "market": "batter_hits",
         "book": "FanDuel",    "bet_type": "Over",  "line": 1.5,  "odds": -125},
        {"player_name": "Ronald Acuña Jr.","market": "batter_hits",
         "book": "BetMGM",    "bet_type": "Over",   "line": 1.5,  "odds": -118},
        # Batter HR
        {"player_name": "Aaron Judge",    "market": "batter_home_runs",
         "book": "DraftKings", "bet_type": "Over",  "line": 0.5,  "odds": +175},
        {"player_name": "Shohei Ohtani",  "market": "batter_home_runs",
         "book": "FanDuel",    "bet_type": "Over",  "line": 0.5,  "odds": +200},
        {"player_name": "Yordan Alvarez", "market": "batter_home_runs",
         "book": "BetMGM",     "bet_type": "Over",  "line": 0.5,  "odds": +190},
        # Batter RBI
        {"player_name": "Yordan Alvarez", "market": "batter_rbis",
         "book": "DraftKings", "bet_type": "Over",  "line": 1.5,  "odds": -108},
        {"player_name": "Aaron Judge",    "market": "batter_rbis",
         "book": "FanDuel",    "bet_type": "Over",  "line": 1.5,  "odds": -112},
    ]


# ─────────────────────────────────────────
#  PROBABILITY MATH
# ─────────────────────────────────────────

def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def proj_to_win_prob(proj_value: float, line: float,
                     bet_type: str, sigma: float = 1.2) -> float:
    """
    Convert a model projection to win probability for an over/under bet.
    Uses normal distribution around the projected value.

    proj_value: model's projected stat (e.g. 8.1 Ks)
    line:       sportsbook line (e.g. 7.5)
    bet_type:   "Over" or "Under"
    sigma:      estimated standard deviation of outcome
    """
    from scipy import stats as scipy_stats

    try:
        if bet_type == "Over":
            # P(outcome > line) with our projected mean
            prob = 1 - scipy_stats.norm.cdf(line, loc=proj_value, scale=sigma)
        else:
            prob = scipy_stats.norm.cdf(line, loc=proj_value, scale=sigma)
        return float(np.clip(prob, 0.01, 0.99))
    except ImportError:
        # Fallback without scipy — simple heuristic
        diff = (proj_value - line) / sigma
        if bet_type == "Over":
            return float(np.clip(0.5 + diff * 0.2, 0.05, 0.95))
        else:
            return float(np.clip(0.5 - diff * 0.2, 0.05, 0.95))


def calc_edge(model_prob: float, implied_prob: float) -> float:
    """Edge = model win probability minus book's implied probability."""
    return round(model_prob - implied_prob, 4)


# ─────────────────────────────────────────
#  EDGE FINDER
# ─────────────────────────────────────────

def find_edges(pitcher_projections: list[dict],
               batter_projections: list[dict],
               props: Optional[list[dict]] = None) -> list[dict]:
    """
    Master edge finder.
    Takes model projections and compares vs sportsbook lines.

    pitcher_projections: list of {player_name, proj_k, proj_ip, confidence, ...}
    batter_projections:  list of {player_name, proj_hr, proj_hits, proj_runs, proj_rbi, ...}

    Returns list of edge opportunities sorted by edge descending.
    """
    if props is None:
        props = fetch_mlb_props()

    # Build lookup maps
    pitcher_map = {p["player_name"]: p for p in pitcher_projections}
    batter_map  = {b["player_name"]: b for b in batter_projections}

    edges = []

    for prop in props:
        player = prop["player_name"]
        market = prop["market"]
        line   = prop["line"]
        bet    = prop["bet_type"]
        odds   = prop["odds"]
        book   = prop["book"]

        implied_prob = american_to_implied_prob(odds)
        model_proj   = None
        sigma        = 1.2

        # Match player to projection
        if market == "pitcher_strikeouts" and player in pitcher_map:
            p = pitcher_map[player]
            model_proj = p["proj_k"]
            sigma      = 1.8  # Ks have more variance
            stat_label = "K"

        elif market == "batter_hits" and player in batter_map:
            b = batter_map[player]
            model_proj = b["proj_hits"]
            sigma      = 0.9
            stat_label = "Hits"

        elif market == "batter_home_runs" and player in batter_map:
            b = batter_map[player]
            model_proj = b["proj_hr"]
            sigma      = 0.5
            stat_label = "HR"

        elif market == "batter_rbis" and player in batter_map:
            b = batter_map[player]
            model_proj = b["proj_rbi"]
            sigma      = 1.1
            stat_label = "RBI"

        elif market == "batter_runs_scored" and player in batter_map:
            b = batter_map[player]
            model_proj = b["proj_runs"]
            sigma      = 0.9
            stat_label = "Runs"

        if model_proj is None:
            continue

        model_prob = proj_to_win_prob(model_proj, line, bet, sigma)
        edge       = calc_edge(model_prob, implied_prob)

        if edge >= MIN_EDGE_THRESHOLD or edge <= -MIN_EDGE_THRESHOLD:
            edges.append({
                "player_name":   player,
                "market":        market,
                "stat_label":    stat_label,
                "book":          book,
                "bet_type":      bet,
                "line":          line,
                "odds":          odds,
                "model_proj":    round(model_proj, 2),
                "model_prob":    round(model_prob, 4),
                "implied_prob":  round(implied_prob, 4),
                "edge":          round(edge, 4),
                "edge_pct":      f"{edge*100:+.1f}%",
                "win_prob_pct":  f"{model_prob*100:.0f}%",
                "rating":        _edge_rating(edge),
            })

    # Sort: biggest positive edge first
    edges.sort(key=lambda x: x["edge"], reverse=True)
    return edges


def _edge_rating(edge: float) -> str:
    """Convert edge value to human-readable tier."""
    if edge >= 0.18:  return "👑 ROYAL"
    if edge >= 0.12:  return "🔥 HIGH"
    if edge >= 0.08:  return "✅ MEDIUM"
    if edge >= 0.04:  return "📊 LOW"
    return "❌ NO EDGE"


# ─────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────

def print_edges(edges: list[dict], top_n: int = 10):
    """Pretty-print the top edge opportunities."""
    print(f"\n👑 Diamond King — Top {top_n} Edges")
    print("=" * 60)

    royal_picks = [e for e in edges if "ROYAL" in e["rating"] or "HIGH" in e["rating"]]

    for i, e in enumerate(edges[:top_n], 1):
        print(f"\n{i:2}. {e['rating']}  {e['player_name']}")
        print(f"    {e['bet_type']} {e['line']} {e['stat_label']}  •  {e['book']}")
        print(f"    Odds: {e['odds']:+d}  •  Book: {e['implied_prob']*100:.0f}%  •  Model: {e['model_prob']*100:.0f}%")
        print(f"    Model Proj: {e['model_proj']}  •  Edge: {e['edge_pct']}  •  Win Prob: {e['win_prob_pct']}")

    print(f"\n📊 Total edges found: {len(edges)}")
    print(f"👑 Royal/High edges:  {len(royal_picks)}")


if __name__ == "__main__":
    # Demo with mock data
    mock_pitchers = [
        {"player_name": "Gerrit Cole",  "proj_k": 8.1, "proj_ip": 6.1, "confidence": 0.82},
        {"player_name": "Dylan Cease",  "proj_k": 7.3, "proj_ip": 5.8, "confidence": 0.74},
        {"player_name": "Logan Webb",   "proj_k": 5.2, "proj_ip": 6.0, "confidence": 0.70},
    ]
    mock_batters = [
        {"player_name": "Shohei Ohtani",   "proj_hr": 0.42, "proj_hits": 1.38, "proj_runs": 0.91, "proj_rbi": 1.12},
        {"player_name": "Freddie Freeman", "proj_hr": 0.21, "proj_hits": 1.55, "proj_runs": 0.88, "proj_rbi": 0.94},
        {"player_name": "Aaron Judge",     "proj_hr": 0.48, "proj_hits": 1.22, "proj_runs": 0.82, "proj_rbi": 1.28},
        {"player_name": "Yordan Alvarez",  "proj_hr": 0.38, "proj_hits": 1.31, "proj_runs": 0.79, "proj_rbi": 1.41},
        {"player_name": "Ronald Acuña Jr.","proj_hr": 0.29, "proj_hits": 1.62, "proj_runs": 1.08, "proj_rbi": 0.88},
    ]

    edges = find_edges(mock_pitchers, mock_batters)
    print_edges(edges)

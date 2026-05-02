"""
Diamond King — MLB Data Fetcher (Windows-compatible)
"""

import requests
import time
from datetime import date
from typing import Optional

BASE = "https://statsapi.mlb.com/api/v1"

PARK_HR_FACTORS = {
    "Coors Field": 1.38, "Great American Ball Park": 1.25,
    "Globe Life Field": 1.18, "Yankee Stadium": 1.20,
    "Chase Field": 1.12, "Fenway Park": 1.08,
    "Wrigley Field": 1.05, "Truist Park": 1.04,
    "American Family Field": 1.10, "Citizens Bank Park": 1.15,
    "loanDepot park": 0.88, "Oracle Park": 0.82,
    "Petco Park": 0.85, "T-Mobile Park": 0.87,
    "Dodger Stadium": 0.95, "Kauffman Stadium": 0.92,
    "Busch Stadium": 0.93,
}

UMPIRE_K_MODIFIERS = {
    "CB Bucknor": +1.6, "Angel Hernandez": +0.3,
    "Joe West": -0.8, "Dan Iassogna": +1.2,
    "Ted Barrett": +0.5, "Jim Joyce": -0.4,
    "Larry Vanover": +0.9, "Brian Gorman": -0.6,
    "Mark Carlson": +0.7,
}

def get_park_hr_factor(venue_name):
    for key, factor in PARK_HR_FACTORS.items():
        if key.lower() in venue_name.lower():
            return factor
    return 1.0

def get_umpire_k_modifier(ump_name):
    for key, mod in UMPIRE_K_MODIFIERS.items():
        if key.lower() in ump_name.lower():
            return mod
    return 0.0

def get_todays_games(game_date=None):
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")
    url = f"{BASE}/schedule"
    params = {"sportId": 1, "date": game_date, "hydrate": "team,linescore,probablePitcher,lineups"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[mlb_fetcher] Error fetching schedule: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            away_pitcher = away.get("probablePitcher", {})
            home_pitcher = home.get("probablePitcher", {})
            games.append({
                "game_pk": game["gamePk"],
                "game_date": game_date,
                "status": game.get("status", {}).get("detailedState", "Scheduled"),
                "venue": game.get("venue", {}).get("name", "Unknown"),
                "game_time": game.get("gameDate", ""),
                "away_team_id": away.get("team", {}).get("id"),
                "away_team_name": away.get("team", {}).get("name", ""),
                "away_team_abbr": away.get("team", {}).get("abbreviation", ""),
                "away_record_w": away.get("leagueRecord", {}).get("wins", 0),
                "away_record_l": away.get("leagueRecord", {}).get("losses", 0),
                "home_team_id": home.get("team", {}).get("id"),
                "home_team_name": home.get("team", {}).get("name", ""),
                "home_team_abbr": home.get("team", {}).get("abbreviation", ""),
                "home_record_w": home.get("leagueRecord", {}).get("wins", 0),
                "home_record_l": home.get("leagueRecord", {}).get("losses", 0),
                "away_pitcher_id": away_pitcher.get("id"),
                "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                "home_pitcher_id": home_pitcher.get("id"),
                "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
            })
    return games

def get_game_lineup(game_pk):
    url = f"{BASE}/game/{game_pk}/boxscore"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[mlb_fetcher] Error fetching lineup {game_pk}: {e}")
        return {"away": [], "home": []}

    def parse_side(side_data):
        players = []
        batting_order = side_data.get("battingOrder", [])
        all_players = side_data.get("players", {})
        for i, pid in enumerate(batting_order):
            key = f"ID{pid}"
            p = all_players.get(key, {})
            person = p.get("person", {})
            pos = p.get("position", {})
            players.append({
                "player_id": person.get("id"),
                "player_name": person.get("fullName", ""),
                "batting_order": i + 1,
                "position": pos.get("abbreviation", ""),
            })
        return players

    teams = data.get("teams", {})
    return {"away": parse_side(teams.get("away", {})), "home": parse_side(teams.get("home", {}))}

def get_batter_stats(player_id, season=None):
    if season is None:
        season = date.today().year
    url = f"{BASE}/people/{player_id}/stats"
    params = {"stats": "season", "group": "hitting", "season": season, "gameType": "R"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except:
        return {}
    result = {}
    for sg in data.get("stats", []):
        splits = sg.get("splits", [])
        if not splits:
            continue
        s = splits[0].get("stat", {})
        ab = max(int(s.get("atBats", 1) or 1), 1)
        result["season"] = {
            "avg": float(s.get("avg", 0) or 0),
            "obp": float(s.get("obp", 0) or 0),
            "slg": float(s.get("slg", 0) or 0),
            "ops": float(s.get("ops", 0) or 0),
            "hr": int(s.get("homeRuns", 0) or 0),
            "hits": int(s.get("hits", 0) or 0),
            "rbi": int(s.get("rbi", 0) or 0),
            "runs": int(s.get("runs", 0) or 0),
            "ab": ab,
            "strikeouts": int(s.get("strikeOuts", 0) or 0),
            "bb": int(s.get("baseOnBalls", 0) or 0),
        }
    return result

def get_pitcher_stats(player_id, season=None):
    if season is None:
        season = date.today().year
    url = f"{BASE}/people/{player_id}/stats"
    params = {"stats": "season", "group": "pitching", "season": season, "gameType": "R"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except:
        return {}
    result = {}
    for sg in data.get("stats", []):
        splits = sg.get("splits", [])
        if not splits:
            continue
        s = splits[0].get("stat", {})
        result["season"] = {
            "era": float(s.get("era", 4.5) or 4.5),
            "whip": float(s.get("whip", 1.3) or 1.3),
            "k9": float(s.get("strikeoutsPer9Inn", 8.0) or 8.0),
            "bb9": float(s.get("walksPer9Inn", 3.0) or 3.0),
            "strikeouts": int(s.get("strikeOuts", 0) or 0),
            "innings": float(s.get("inningsPitched", 0) or 0),
            "games": int(s.get("gamesStarted", 1) or 1),
            "hr_allowed": int(s.get("homeRuns", 0) or 0),
        }
    return result

def get_player_info(player_id):
    url = f"{BASE}/people/{player_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        p = r.json().get("people", [{}])[0]
        return {
            "player_id": p.get("id"),
            "name": p.get("fullName", ""),
            "bat_side": p.get("batSide", {}).get("code", "R"),
            "pitch_hand": p.get("pitchHand", {}).get("code", "R"),
        }
    except:
        return {}

def build_todays_slate(game_date=None):
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    print(f"\n🔄 Building slate for {game_date}...")
    games = get_todays_games(game_date)
    print(f"   Found {len(games)} games")

    all_pitchers, all_batters = [], []

    for game in games:
        venue = game["venue"]
        park_factor = get_park_hr_factor(venue)
        game_pk = game["game_pk"]

        for side in ["away", "home"]:
            pid = game[f"{side}_pitcher_id"]
            pname = game[f"{side}_pitcher_name"]
            if not pid:
                continue
            time.sleep(0.1)
            stats = get_pitcher_stats(pid)
            info = get_player_info(pid)
            season = stats.get("season", {})
            ip = max(season.get("innings", 18), 1)
            games_count = max(season.get("games", 3), 1)
            all_pitchers.append({
                "player_id": pid, "player_name": pname,
                "game_pk": game_pk, "game_date": game_date,
                "venue": venue, "park_hr_factor": park_factor,
                "side": side,
                "opp_team": game[f"{'home' if side=='away' else 'away'}_team_abbr"],
                "pitch_hand": info.get("pitch_hand", "R"),
                "era": season.get("era", 4.50),
                "whip": season.get("whip", 1.30),
                "k9": season.get("k9", 8.0),
                "bb9": season.get("bb9", 3.0),
                "hr9": round(season.get("hr_allowed", 0) / max(ip / 9, 1), 2),
                "season_k": season.get("strikeouts", 0),
                "season_ip": ip,
                "season_games": games_count,
                "last5_k9": season.get("k9", 8.0),
                "last5_era": season.get("era", 4.50),
                "avg_ip_per_start": round(ip / games_count, 2),
            })

        time.sleep(0.15)
        lineup = get_game_lineup(game_pk)

        for side in ["away", "home"]:
            opp_pitcher_hand = "R"
            opp_pid = game[f"{'home' if side=='away' else 'away'}_pitcher_id"]
            if opp_pid:
                opp_info = get_player_info(opp_pid)
                opp_pitcher_hand = opp_info.get("pitch_hand", "R")

            for batter in lineup[side]:
                bid = batter.get("player_id")
                if not bid:
                    continue
                time.sleep(0.1)
                bstats = get_batter_stats(bid)
                binfo = get_player_info(bid)
                season = bstats.get("season", {})
                ab = max(season.get("ab", 1), 1)
                all_batters.append({
                    "player_id": bid,
                    "player_name": batter["player_name"],
                    "game_pk": game_pk, "game_date": game_date,
                    "venue": venue, "park_hr_factor": park_factor,
                    "side": side,
                    "batting_order": batter["batting_order"],
                    "bat_side": binfo.get("bat_side", "R"),
                    "opp_pitcher_hand": opp_pitcher_hand,
                    "platoon_match": int(binfo.get("bat_side", "R") != opp_pitcher_hand),
                    "avg": season.get("avg", .250),
                    "obp": season.get("obp", .320),
                    "slg": season.get("slg", .400),
                    "ops": season.get("ops", .720),
                    "hr_rate": round(season.get("hr", 0) / ab, 4),
                    "hit_rate": round(season.get("hits", 0) / ab, 4),
                    "rbi_rate": round(season.get("rbi", 0) / ab, 4),
                    "run_rate": round(season.get("runs", 0) / ab, 4),
                    "k_rate": round(season.get("strikeouts", 0) / ab, 4),
                    "bb_rate": round(season.get("bb", 0) / ab, 4),
                    "last7_avg": season.get("avg", .250),
                })

    print(f"   ✅ {len(all_pitchers)} pitchers, {len(all_batters)} batters loaded")
    return {"date": game_date, "games": games, "pitchers": all_pitchers, "batters": all_batters}

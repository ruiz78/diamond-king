# 👑 Diamond King — Setup & Run Guide

## Project Structure
```
diamond_king/
├── models/
│   ├── prediction_models.py   ← ML models (Pitcher K, Batter HR/Hits/Runs/RBI)
│   ├── pitcher_k_model.pkl    ← Saved model (auto-generated on first run)
│   └── batter_model.pkl       ← Saved model (auto-generated on first run)
├── pipeline/
│   ├── mlb_fetcher.py         ← MLB Stats API data fetcher
│   ├── edge_calculator.py     ← Prop edge finder vs sportsbooks
│   └── midnight_engine.py     ← Master cron orchestrator
├── api/
│   └── server.py              ← FastAPI REST server
├── data/
│   └── projections_YYYY-MM-DD.json  ← Output files
└── README.md
```

## 1. Install Dependencies
```bash
pip install scikit-learn requests fastapi uvicorn
```

## 2. Set Your Odds API Key
```bash
export ODDS_API_KEY="your_key_here"
```
Get one free at: https://the-odds-api.com

## 3. Train the ML Models
```bash
python3 models/prediction_models.py
```
✅ Saves pitcher_k_model.pkl and batter_model.pkl

## 4. Run the Midnight Engine (generate today's projections)
```bash
python3 pipeline/midnight_engine.py
```
Output saved to: data/projections_YYYY-MM-DD.json

## 5. Start the API Server
```bash
pip install fastapi uvicorn
uvicorn api.server:app --reload --port 8000
```
Then open: http://localhost:8000/docs

## API Endpoints
| Endpoint | Description |
|---|---|
| GET /slate | Today's full projection slate |
| GET /projections/pitchers | Pitcher K projections |
| GET /projections/batters | Batter HR/Hits/Runs/RBI |
| GET /edges | All prop edges ranked |
| GET /edges/royal | Only Royal/High tier picks |
| GET /games | Today's game schedule |
| GET /player/{name} | Single player lookup |
| POST /engine/run | Trigger engine manually |

## 6. Schedule Midnight Auto-Refresh (Linux/Mac)
```bash
crontab -e
# Add this line:
0 0 * * * cd /path/to/diamond_king && python3 pipeline/midnight_engine.py
```

## What the Models Predict
- **Pitcher K Model** — strikeouts + innings pitched per start
  - Inputs: K/9, ERA, WHIP, SwStr%, recent form, umpire modifier, park factor
  - MAE: ~0.83 Ks per game
- **Batter Models** — per-game HR, Hits, Runs, RBI
  - Inputs: AVG, OBP, SLG, OPS, platoon matchup, lineup spot, park factor, hot/cold streak
  - MAE: HR 0.26 | Hits 0.77 | Runs 0.56 | RBI 0.56

## Next Steps (Upgrade Path)
1. **Real training data** — export Statcast CSVs from Baseball Savant and retrain
2. **Umpire scraper** — auto-pull daily assignments from umpire sources
3. **PostgreSQL** — swap JSON files for a real database
4. **Web frontend** — React app consuming the API endpoints
5. **Push notifications** — Firebase for mobile alerts

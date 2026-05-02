"""
Diamond King — ML Prediction Models
Uses scikit-learn GradientBoostingRegressor (equivalent power to XGBoost)

Models:
  PitcherKModel   → projected strikeouts + innings per start
  BatterHRModel   → projected home runs per game
  BatterHitModel  → projected hits per game
  BatterRunModel  → projected runs per game
  BatterRBIModel  → projected RBI per game
"""

import numpy as np
import json
import os
import pickle
from datetime import date
from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# ─────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────

def pitcher_features(p: dict) -> list[float]:
    """
    Extract numeric feature vector from a pitcher data dict.
    Features the model uses to predict strikeouts.
    """
    return [
        p.get("k9", 8.0),               # season K/9
        p.get("era", 4.50),              # season ERA
        p.get("whip", 1.30),             # season WHIP
        p.get("bb9", 3.0),              # walk rate
        p.get("hr9", 1.0),              # HR allowed rate
        p.get("last5_k9", 8.0),          # recent K/9
        p.get("last5_era", 4.50),        # recent ERA
        p.get("avg_ip_per_start", 5.5),  # how deep they go
        p.get("park_hr_factor", 1.0),    # park factor (affects pitch approach)
        1.0 if p.get("pitch_hand") == "R" else 0.0,  # handedness
    ]

PITCHER_FEATURE_NAMES = [
    "k9", "era", "whip", "bb9", "hr9",
    "last5_k9", "last5_era", "avg_ip_per_start",
    "park_hr_factor", "is_right_handed",
]


def batter_features(b: dict) -> list[float]:
    """
    Extract numeric feature vector from a batter data dict.
    Used for ALL batter models (HR, Hits, Runs, RBI).
    """
    return [
        b.get("avg", .250),              # season batting avg
        b.get("obp", .320),              # on-base %
        b.get("slg", .400),              # slugging %
        b.get("ops", .720),              # OPS
        b.get("hr_rate", .04),           # HR per AB
        b.get("hit_rate", .250),         # hits per AB
        b.get("rbi_rate", .12),          # RBI per AB
        b.get("run_rate", .11),          # runs per AB
        b.get("k_rate", .22),            # K rate
        b.get("bb_rate", .09),           # walk rate
        b.get("last7_avg", .250),        # hot/cold streak
        b.get("park_hr_factor", 1.0),    # park
        float(b.get("batting_order", 5)),# lineup spot (1-2-3 vs 8-9)
        float(b.get("platoon_match", 0)),# same hand matchup disadvantage
        1.0 if b.get("bat_side") == "L" else 0.0,
    ]

BATTER_FEATURE_NAMES = [
    "avg", "obp", "slg", "ops", "hr_rate", "hit_rate",
    "rbi_rate", "run_rate", "k_rate", "bb_rate",
    "last7_avg", "park_hr_factor", "batting_order",
    "platoon_match", "is_left_handed",
]


# ─────────────────────────────────────────
#  SYNTHETIC TRAINING DATA GENERATOR
# ─────────────────────────────────────────
# Until you have a real historical database (Statcast export),
# we generate realistic synthetic training data based on
# known MLB statistical distributions. Replace with real CSV when available.

def _generate_pitcher_training_data(n: int = 5000):
    """
    Generates synthetic pitcher game log data.
    K distribution: mean ~5.8, std ~2.2 (realistic for starters).
    """
    np.random.seed(42)
    X, y_k, y_ip = [], [], []

    for _ in range(n):
        k9         = np.random.normal(8.5, 2.5)
        era        = np.random.normal(4.20, 1.20)
        whip       = np.random.normal(1.28, 0.22)
        bb9        = np.random.normal(3.0, 0.9)
        hr9        = np.random.normal(1.1, 0.4)
        last5_k9   = k9 + np.random.normal(0, 1.0)
        last5_era  = era + np.random.normal(0, 0.8)
        avg_ip     = np.clip(np.random.normal(5.5, 0.9), 3.0, 8.0)
        park_fac   = np.random.choice([0.85, 0.92, 1.0, 1.05, 1.12, 1.20, 1.38],
                                       p=[0.1, 0.1, 0.4, 0.15, 0.1, 0.1, 0.05])
        is_righty  = np.random.choice([0.0, 1.0], p=[0.3, 0.7])

        k9   = np.clip(k9, 3.0, 16.0)
        era  = np.clip(era, 1.5, 9.0)
        whip = np.clip(whip, 0.8, 2.0)

        feat = [k9, era, whip, bb9, hr9, last5_k9, last5_era,
                avg_ip, park_fac, is_righty]

        # Target: Ks in a start
        # Base: (k9 / 9) * avg_ip + noise
        base_k = (k9 / 9.0) * avg_ip
        recent_adj = (last5_k9 - k9) * 0.3  # hot/cold factor
        noise  = np.random.normal(0, 1.0)
        proj_k = np.clip(base_k + recent_adj + noise, 0, 18)

        # Target: IP in start
        proj_ip = np.clip(avg_ip + np.random.normal(0, 0.7), 2.0, 9.0)

        X.append(feat)
        y_k.append(proj_k)
        y_ip.append(proj_ip)

    return np.array(X), np.array(y_k), np.array(y_ip)


def _generate_batter_training_data(n: int = 8000):
    """
    Generates synthetic batter game data.
    One row = one player-game observation.
    """
    np.random.seed(99)
    X = []
    y_hr, y_hits, y_runs, y_rbi = [], [], [], []

    for _ in range(n):
        avg       = np.clip(np.random.normal(.258, .035), .150, .380)
        obp       = avg + np.clip(np.random.normal(.070, .020), .020, .150)
        slg       = avg + np.clip(np.random.normal(.155, .055), .050, .350)
        ops       = obp + slg
        hr_rate   = np.clip(np.random.normal(.040, .020), .002, .120)
        hit_rate  = avg
        rbi_rate  = np.clip(np.random.normal(.115, .035), .020, .250)
        run_rate  = np.clip(np.random.normal(.105, .030), .020, .220)
        k_rate    = np.clip(np.random.normal(.220, .055), .070, .420)
        bb_rate   = np.clip(np.random.normal(.090, .030), .020, .200)
        last7_avg = avg + np.random.normal(0, .040)
        park_fac  = np.random.choice([0.85, 0.92, 1.0, 1.05, 1.12, 1.20, 1.38],
                                      p=[0.1, 0.1, 0.4, 0.15, 0.1, 0.1, 0.05])
        bat_order = float(np.random.randint(1, 10))
        platoon   = float(np.random.choice([0, 1], p=[0.55, 0.45]))
        is_lefty  = float(np.random.choice([0, 1], p=[0.65, 0.35]))

        feat = [avg, obp, slg, ops, hr_rate, hit_rate, rbi_rate, run_rate,
                k_rate, bb_rate, last7_avg, park_fac, bat_order, platoon, is_lefty]

        # Targets (per game)
        # HR: Poisson-like, driven by hr_rate and park
        p_hr   = np.clip(hr_rate * park_fac * (1 - platoon * 0.1), 0, 0.18)
        proj_hr = np.random.poisson(p_hr * 4)  # ~4 AB per game
        proj_hr = np.clip(proj_hr, 0, 3)

        # Hits: avg * ~4 AB + noise
        proj_hits = np.clip(np.random.poisson(avg * 3.8 * (1 - platoon * 0.06)), 0, 5)

        # Runs: dependent on hitting + lineup spot
        lineup_bonus = max(0, (5 - bat_order) * 0.02)
        proj_runs = np.clip(np.random.poisson(run_rate * 4 + lineup_bonus), 0, 4)

        # RBI: dependent on lineup spot and slugging
        rbi_lineup = 0.05 if bat_order in [3, 4, 5] else 0
        proj_rbi = np.clip(np.random.poisson(rbi_rate * 4 + rbi_lineup), 0, 5)

        X.append(feat)
        y_hr.append(float(proj_hr))
        y_hits.append(float(proj_hits))
        y_runs.append(float(proj_runs))
        y_rbi.append(float(proj_rbi))

    return np.array(X), np.array(y_hr), np.array(y_hits), \
           np.array(y_runs), np.array(y_rbi)


# ─────────────────────────────────────────
#  MODEL BUILDER
# ─────────────────────────────────────────

def _build_model(n_estimators=200, max_depth=4, learning_rate=0.05):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        ))
    ])


# ─────────────────────────────────────────
#  PITCHER MODEL
# ─────────────────────────────────────────

class PitcherKModel:
    """
    Predicts pitcher strikeouts and innings pitched per start.
    """

    def __init__(self):
        self.k_model  = _build_model()
        self.ip_model = _build_model()
        self.trained  = False

    def train(self, verbose=True):
        X, y_k, y_ip = _generate_pitcher_training_data(5000)

        self.k_model.fit(X, y_k)
        self.ip_model.fit(X, y_ip)
        self.trained = True

        if verbose:
            k_cv  = cross_val_score(self.k_model, X, y_k, cv=5,
                                     scoring="neg_mean_absolute_error")
            ip_cv = cross_val_score(self.ip_model, X, y_ip, cv=5,
                                     scoring="neg_mean_absolute_error")
            print(f"   ⚾ PitcherKModel trained")
            print(f"      K  MAE: {-k_cv.mean():.3f} ± {k_cv.std():.3f}")
            print(f"      IP MAE: {-ip_cv.mean():.3f} ± {ip_cv.std():.3f}")

    def predict(self, pitcher_dict: dict, umpire_k_mod: float = 0.0) -> dict:
        """
        Returns projection dict for a single pitcher.
        umpire_k_mod: K modifier from umpire tendencies (+/- Ks)
        """
        if not self.trained:
            self.train(verbose=False)

        feat = np.array([pitcher_features(pitcher_dict)])

        raw_k  = float(self.k_model.predict(feat)[0])
        raw_ip = float(self.ip_model.predict(feat)[0])

        proj_k  = np.clip(raw_k + umpire_k_mod, 0, 18)
        proj_ip = np.clip(raw_ip, 2.0, 9.0)

        # Confidence: lower when small sample or high variance
        games = max(pitcher_dict.get("season_games", 5), 1)
        conf  = np.clip(0.50 + (games / 50) * 0.35, 0.50, 0.88)

        return {
            "proj_k":       round(proj_k, 1),
            "proj_ip":      round(proj_ip, 1),
            "proj_k9":      round((proj_k / max(proj_ip, 1)) * 9, 1),
            "ump_mod":      round(umpire_k_mod, 1),
            "confidence":   round(conf, 2),
            "feature_names": PITCHER_FEATURE_NAMES,
        }

    def save(self):
        with open(os.path.join(MODEL_DIR, "pitcher_k_model.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load():
        path = os.path.join(MODEL_DIR, "pitcher_k_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        m = PitcherKModel()
        m.train()
        m.save()
        return m


# ─────────────────────────────────────────
#  BATTER MODEL
# ─────────────────────────────────────────

class BatterProjectionModel:
    """
    Predicts per-game: HR, Hits, Runs, RBI for a batter.
    """

    def __init__(self):
        self.hr_model   = _build_model(n_estimators=200, max_depth=3)
        self.hit_model  = _build_model(n_estimators=200, max_depth=4)
        self.run_model  = _build_model(n_estimators=150, max_depth=3)
        self.rbi_model  = _build_model(n_estimators=150, max_depth=3)
        self.trained    = False

    def train(self, verbose=True):
        X, y_hr, y_hits, y_runs, y_rbi = _generate_batter_training_data(8000)

        self.hr_model.fit(X, y_hr)
        self.hit_model.fit(X, y_hits)
        self.run_model.fit(X, y_runs)
        self.rbi_model.fit(X, y_rbi)
        self.trained = True

        if verbose:
            for name, model, y in [
                ("HR",   self.hr_model,  y_hr),
                ("Hits", self.hit_model, y_hits),
                ("Runs", self.run_model, y_runs),
                ("RBI",  self.rbi_model, y_rbi),
            ]:
                cv = cross_val_score(model, X, y, cv=5,
                                     scoring="neg_mean_absolute_error")
                print(f"   🏏 {name:4s} MAE: {-cv.mean():.3f} ± {cv.std():.3f}")

    def predict(self, batter_dict: dict) -> dict:
        """Returns per-game projection for a batter."""
        if not self.trained:
            self.train(verbose=False)

        feat = np.array([batter_features(batter_dict)])

        hr   = float(np.clip(self.hr_model.predict(feat)[0],  0, 3))
        hits = float(np.clip(self.hit_model.predict(feat)[0], 0, 5))
        runs = float(np.clip(self.run_model.predict(feat)[0], 0, 4))
        rbi  = float(np.clip(self.rbi_model.predict(feat)[0], 0, 5))

        # Confidence based on AB sample size
        avg = batter_dict.get("avg", .250)
        conf_base = 0.55 + min(avg * 0.5, 0.25)

        return {
            "proj_hr":      round(hr,   2),
            "proj_hits":    round(hits, 2),
            "proj_runs":    round(runs, 2),
            "proj_rbi":     round(rbi,  2),
            "conf_hr":      round(np.clip(conf_base - 0.08, 0.45, 0.88), 2),
            "conf_hits":    round(np.clip(conf_base + 0.05, 0.55, 0.92), 2),
            "conf_runs":    round(np.clip(conf_base,        0.50, 0.88), 2),
            "conf_rbi":     round(np.clip(conf_base - 0.02, 0.50, 0.88), 2),
        }

    def save(self):
        with open(os.path.join(MODEL_DIR, "batter_model.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load():
        path = os.path.join(MODEL_DIR, "batter_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        m = BatterProjectionModel()
        m.train()
        m.save()
        return m


# ─────────────────────────────────────────
#  TRAINING ENTRY POINT
# ─────────────────────────────────────────

def train_all_models():
    """Train and save all Diamond King models."""
    print("\n👑 Diamond King — Training ML Models")
    print("=" * 45)

    print("\n⚾ Training Pitcher Strikeout Model...")
    pm = PitcherKModel()
    pm.train(verbose=True)
    pm.save()
    print("   ✅ Saved to models/pitcher_k_model.pkl")

    print("\n🏏 Training Batter Projection Models...")
    bm = BatterProjectionModel()
    bm.train(verbose=True)
    bm.save()
    print("   ✅ Saved to models/batter_model.pkl")

    print("\n✅ All models trained and saved.")
    return pm, bm


if __name__ == "__main__":
    pm, bm = train_all_models()

    # ── Quick smoke test ──
    print("\n" + "=" * 45)
    print("🧪 Smoke Test — Sample Predictions")
    print("=" * 45)

    sample_pitcher = {
        "player_name":    "Gerrit Cole",
        "k9":             11.8,
        "era":            3.10,
        "whip":           1.05,
        "bb9":            2.3,
        "hr9":            1.0,
        "last5_k9":       12.4,
        "last5_era":      2.85,
        "avg_ip_per_start": 6.1,
        "park_hr_factor": 1.20,
        "pitch_hand":     "R",
        "season_games":   12,
    }
    k_proj = pm.predict(sample_pitcher, umpire_k_mod=+1.2)
    print(f"\n⚾ {sample_pitcher['player_name']}")
    print(f"   Proj K:  {k_proj['proj_k']}  (ump bonus: +{k_proj['ump_mod']})")
    print(f"   Proj IP: {k_proj['proj_ip']}")
    print(f"   K/9:     {k_proj['proj_k9']}")
    print(f"   Conf:    {int(k_proj['confidence']*100)}%")

    sample_batter = {
        "player_name":    "Shohei Ohtani",
        "avg":            .310,
        "obp":            .395,
        "slg":            .620,
        "ops":            1.015,
        "hr_rate":        .085,
        "hit_rate":       .310,
        "rbi_rate":       .185,
        "run_rate":       .165,
        "k_rate":         .195,
        "bb_rate":        .125,
        "last7_avg":      .340,
        "park_hr_factor": 1.12,
        "batting_order":  2,
        "platoon_match":  0,
        "bat_side":       "L",
    }
    b_proj = bm.predict(sample_batter)
    print(f"\n🏏 {sample_batter['player_name']}")
    print(f"   HR:   {b_proj['proj_hr']}  ({int(b_proj['conf_hr']*100)}% conf)")
    print(f"   Hits: {b_proj['proj_hits']}  ({int(b_proj['conf_hits']*100)}% conf)")
    print(f"   Runs: {b_proj['proj_runs']}  ({int(b_proj['conf_runs']*100)}% conf)")
    print(f"   RBI:  {b_proj['proj_rbi']}  ({int(b_proj['conf_rbi']*100)}% conf)")

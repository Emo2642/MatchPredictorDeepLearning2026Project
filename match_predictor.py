# =============================================================================
# Football Match Prediction System
# BIM447 - Deep Learning ESTÜ 2026 Group Project
#
# Predicts:
#   1. Match result (Home Win / Draw / Away Win) — multi-class classification
#   2. Goal counts (home goals, away goals)      — regression
#
# Compares:
#   - Model 1: Multi-Output Deep Neural Network (DNN)
#   - Model 2: Multi-Output LSTM
#
# Dataset: Kaggle "European Soccer Database"
#   https://www.kaggle.com/datasets/hugomathien/soccer
#   Place the downloaded database.sqlite file next to this script,
#   or set USE_FULL_DATASET = False to use generated synthetic data instead.
# =============================================================================

# ---------------------------------------------------------------------------
# Top-level flag: set to True to use the real Kaggle SQLite database,
#                 set to False to run on 100 synthetic samples (demo mode).
# ---------------------------------------------------------------------------
USE_FULL_DATASET = False
DATABASE_PATH = "database.sqlite"   # path to the Kaggle SQLite file

# =============================================================================
# SECTION 1 — Imports and Setup
# =============================================================================
import os
import warnings
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)

# =============================================================================
# SECTION 2 — Data Loading and Exploration
# =============================================================================

def load_real_data(db_path: str) -> pd.DataFrame:
    """
    Load the Kaggle European Soccer Database from a local SQLite file.
    Joins Match, Team_Attributes tables to retrieve per-match statistics.
    Returns a cleaned DataFrame with raw match and team-attribute columns.
    """
    conn = sqlite3.connect(db_path)

    match_query = """
        SELECT
            m.id            AS match_id,
            m.season,
            m.date,
            m.home_team_api_id,
            m.away_team_api_id,
            m.home_team_goal,
            m.away_team_goal,
            -- home team attributes (most recent before match)
            ht.buildUpPlaySpeed         AS h_speed,
            ht.buildUpPlayPassing       AS h_passing,
            ht.chanceCreationPassing    AS h_chance_pass,
            ht.chanceCreationCrossing   AS h_chance_cross,
            ht.chanceCreationShooting   AS h_chance_shoot,
            ht.defencePressure          AS h_def_pressure,
            ht.defenceAggression        AS h_def_aggression,
            ht.defenceTeamWidth         AS h_def_width,
            -- away team attributes
            at.buildUpPlaySpeed         AS a_speed,
            at.buildUpPlayPassing       AS a_passing,
            at.chanceCreationPassing    AS a_chance_pass,
            at.chanceCreationCrossing   AS a_chance_cross,
            at.chanceCreationShooting   AS a_chance_shoot,
            at.defencePressure          AS a_def_pressure,
            at.defenceAggression        AS a_def_aggression,
            at.defenceTeamWidth         AS a_def_width
        FROM Match m
        LEFT JOIN Team_Attributes ht
            ON m.home_team_api_id = ht.team_api_id
            AND ht.date = (
                SELECT MAX(date) FROM Team_Attributes
                WHERE team_api_id = m.home_team_api_id AND date <= m.date
            )
        LEFT JOIN Team_Attributes at
            ON m.away_team_api_id = at.team_api_id
            AND at.date = (
                SELECT MAX(date) FROM Team_Attributes
                WHERE team_api_id = m.away_team_api_id AND date <= m.date
            )
        WHERE m.home_team_goal IS NOT NULL
          AND m.away_team_goal IS NOT NULL
        ORDER BY m.date
    """
    df = pd.read_sql(match_query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True)
    print(f"Loaded {len(df):,} matches from {db_path}")
    return df


def generate_synthetic_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors the structure of the Kaggle
    European Soccer Database.  Used when USE_FULL_DATASET = False or when
    the SQLite file is not present.
    """
    rng = np.random.default_rng(42)
    n_teams = max(20, n_samples // 5)
    team_ids = np.arange(1, n_teams + 1)

    dates = pd.date_range("2010-01-01", periods=n_samples, freq="7D")
    home_ids = rng.choice(team_ids, size=n_samples)
    away_ids = rng.choice(team_ids, size=n_samples)
    # Make sure home != away
    same = home_ids == away_ids
    away_ids[same] = (away_ids[same] % n_teams) + 1

    home_goals = rng.poisson(1.5, size=n_samples)
    away_goals = rng.poisson(1.1, size=n_samples)

    def rand_attr(size):
        return rng.integers(30, 90, size=size).astype(float)

    df = pd.DataFrame({
        "match_id":         np.arange(n_samples),
        "season":           ["2012/2013"] * n_samples,
        "date":             dates,
        "home_team_api_id": home_ids,
        "away_team_api_id": away_ids,
        "home_team_goal":   home_goals,
        "away_team_goal":   away_goals,
        # home team style attributes
        "h_speed":          rand_attr(n_samples),
        "h_passing":        rand_attr(n_samples),
        "h_chance_pass":    rand_attr(n_samples),
        "h_chance_cross":   rand_attr(n_samples),
        "h_chance_shoot":   rand_attr(n_samples),
        "h_def_pressure":   rand_attr(n_samples),
        "h_def_aggression": rand_attr(n_samples),
        "h_def_width":      rand_attr(n_samples),
        # away team style attributes
        "a_speed":          rand_attr(n_samples),
        "a_passing":        rand_attr(n_samples),
        "a_chance_pass":    rand_attr(n_samples),
        "a_chance_cross":   rand_attr(n_samples),
        "a_chance_shoot":   rand_attr(n_samples),
        "a_def_pressure":   rand_attr(n_samples),
        "a_def_aggression": rand_attr(n_samples),
        "a_def_width":      rand_attr(n_samples),
        # synthetic possession / shots (not in real Team_Attributes but derived)
        "h_possession":     rng.uniform(35, 65, size=n_samples),
        "a_possession":     rng.uniform(35, 65, size=n_samples),
        "h_shots":          rng.uniform(5, 20, size=n_samples),
        "a_shots":          rng.uniform(5, 20, size=n_samples),
        "h_shots_on_target":rng.uniform(2, 10, size=n_samples),
        "a_shots_on_target":rng.uniform(2, 10, size=n_samples),
        "h_fouls":          rng.uniform(8, 20, size=n_samples),
        "a_fouls":          rng.uniform(8, 20, size=n_samples),
        "h_corners":        rng.uniform(2, 12, size=n_samples),
        "a_corners":        rng.uniform(2, 12, size=n_samples),
        "h_yellow_cards":   rng.uniform(0, 4, size=n_samples),
        "a_yellow_cards":   rng.uniform(0, 4, size=n_samples),
        "h_red_cards":      rng.uniform(0, 1, size=n_samples),
        "a_red_cards":      rng.uniform(0, 1, size=n_samples),
    })
    print(f"Generated {len(df):,} synthetic matches (demo mode)")
    return df


def load_data() -> pd.DataFrame:
    """Entry point: load real or synthetic data based on the global flag."""
    if USE_FULL_DATASET and os.path.exists(DATABASE_PATH):
        return load_real_data(DATABASE_PATH)
    if USE_FULL_DATASET:
        print(
            f"WARNING: {DATABASE_PATH} not found. "
            "Falling back to synthetic data."
        )
    return generate_synthetic_data(n_samples=100)


# =============================================================================
# SECTION 3 — Feature Engineering
# =============================================================================

def add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'result' column:
      0 = Away Win, 1 = Draw, 2 = Home Win
    """
    def _result(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return 2   # Home Win
        if row["home_team_goal"] == row["away_team_goal"]:
            return 1   # Draw
        return 0       # Away Win

    df = df.copy()
    df["result"] = df.apply(_result, axis=1)
    return df


def compute_team_form(df: pd.DataFrame, team_id: int,
                      before_date, n_matches: int = 20) -> dict:
    """
    Compute form features for *team_id* using its last *n_matches* matches
    that occurred strictly before *before_date*.

    Returns a dict of aggregated statistics.
    """
    # Gather all matches involving this team (as home or away)
    home_mask = df["home_team_api_id"] == team_id
    away_mask = df["away_team_api_id"] == team_id
    team_matches = df[
        (home_mask | away_mask) & (df["date"] < before_date)
    ].sort_values("date", ascending=False).head(n_matches)

    n = len(team_matches)
    if n == 0:
        # No history — return neutral defaults
        return {
            "win_rate": 0.33, "draw_rate": 0.33, "loss_rate": 0.33,
            "avg_goals_scored": 1.2, "avg_goals_conceded": 1.2,
            "weighted_form": 0.5,
            "h1_goals_scored": 0.6, "h2_goals_scored": 0.6,
            "h1_goals_conceded": 0.6, "h2_goals_conceded": 0.6,
            "home_win_rate": 0.45, "away_win_rate": 0.30,
        }

    wins = draws = losses = 0
    goals_scored = []
    goals_conceded = []
    form_scores = []   # 1=win, 0.5=draw, 0=loss — weighted by recency

    home_wins = home_draws = home_losses = 0
    away_wins = away_draws = away_losses = 0
    home_games = away_games = 0

    for rank, (_, row) in enumerate(team_matches.iterrows()):
        weight = 1.0 / (rank + 1)   # More recent → higher weight
        is_home = row["home_team_api_id"] == team_id
        scored = row["home_team_goal"] if is_home else row["away_team_goal"]
        conceded = row["away_team_goal"] if is_home else row["home_team_goal"]
        goals_scored.append(scored)
        goals_conceded.append(conceded)

        if scored > conceded:
            wins += 1
            form_scores.append(weight * 1.0)
            if is_home:
                home_wins += 1
            else:
                away_wins += 1
        elif scored == conceded:
            draws += 1
            form_scores.append(weight * 0.5)
            if is_home:
                home_draws += 1
            else:
                away_draws += 1
        else:
            losses += 1
            form_scores.append(weight * 0.0)
            if is_home:
                home_losses += 1
            else:
                away_losses += 1

        if is_home:
            home_games += 1
        else:
            away_games += 1

    total_weight = sum(1.0 / (r + 1) for r in range(n))


    home_win_rate = home_wins / home_games if home_games else 0.45
    away_win_rate = away_wins / away_games if away_games else 0.30

    # Approximate half-time split using 40/60 of full-match goals
    avg_scored = np.mean(goals_scored)
    avg_conceded = np.mean(goals_conceded)

    return {
        "win_rate":            wins / n,
        "draw_rate":           draws / n,
        "loss_rate":           losses / n,
        "avg_goals_scored":    avg_scored,
        "avg_goals_conceded":  avg_conceded,
        "weighted_form":       sum(form_scores) / total_weight if total_weight else 0.5,
        "h1_goals_scored":     avg_scored * 0.40,
        "h2_goals_scored":     avg_scored * 0.60,
        "h1_goals_conceded":   avg_conceded * 0.40,
        "h2_goals_conceded":   avg_conceded * 0.60,
        "home_win_rate":       home_win_rate,
        "away_win_rate":       away_win_rate,
    }


def compute_h2h_features(df: pd.DataFrame,
                          home_id: int, away_id: int,
                          before_date, n_matches: int = 10) -> dict:
    """
    Compute Head-to-Head features between home_id and away_id from up to
    the last *n_matches* encounters before *before_date*.
    """
    mask = (
        (
            (df["home_team_api_id"] == home_id) & (df["away_team_api_id"] == away_id)
        ) | (
            (df["home_team_api_id"] == away_id) & (df["away_team_api_id"] == home_id)
        )
    ) & (df["date"] < before_date)

    h2h = df[mask].sort_values("date", ascending=False).head(n_matches)
    n = len(h2h)

    if n == 0:
        return {
            "h2h_home_win_rate": 0.45,
            "h2h_away_win_rate": 0.30,
            "h2h_draw_rate": 0.25,
            "h2h_avg_goals_scored": 1.2,
            "h2h_avg_goals_conceded": 1.2,
            "h2h_home_pct": 0.5,
        }

    home_wins = draws = away_wins = 0
    goals_scored = []
    goals_conceded = []
    home_encounters = 0

    for _, row in h2h.iterrows():
        if row["home_team_api_id"] == home_id:
            scored = row["home_team_goal"]
            conceded = row["away_team_goal"]
            home_encounters += 1
        else:
            scored = row["away_team_goal"]
            conceded = row["home_team_goal"]
        goals_scored.append(scored)
        goals_conceded.append(conceded)

        if scored > conceded:
            home_wins += 1
        elif scored == conceded:
            draws += 1
        else:
            away_wins += 1

    return {
        "h2h_home_win_rate":        home_wins / n,
        "h2h_away_win_rate":        away_wins / n,
        "h2h_draw_rate":            draws / n,
        "h2h_avg_goals_scored":     np.mean(goals_scored),
        "h2h_avg_goals_conceded":   np.mean(goals_conceded),
        "h2h_home_pct":             home_encounters / n,
    }


def compute_style_matchup(df: pd.DataFrame, team_id: int,
                          before_date, n_matches: int = 20) -> dict:
    """
    Compute style-matchup features:
    - Performance vs high-possession teams (opponent >60% possession)
    - Performance vs high-pressing teams (opponent defencePressure > 60)
    - Goals scored vs defensive teams
    """
    home_mask = df["home_team_api_id"] == team_id
    away_mask = df["away_team_api_id"] == team_id
    past = df[
        (home_mask | away_mask) & (df["date"] < before_date)
    ].sort_values("date", ascending=False).head(n_matches)

    vs_high_poss_goals = []
    vs_high_press_goals = []
    vs_defensive_goals = []

    for _, row in past.iterrows():
        is_home = row["home_team_api_id"] == team_id
        scored = row["home_team_goal"] if is_home else row["away_team_goal"]

        opp_poss_col = "a_possession" if is_home else "h_possession"
        opp_press_col = "a_def_pressure" if is_home else "h_def_pressure"
        opp_def_col = "a_def_pressure" if is_home else "h_def_pressure"

        if opp_poss_col in row and row[opp_poss_col] > 60:
            vs_high_poss_goals.append(scored)
        if opp_press_col in row and row[opp_press_col] > 60:
            vs_high_press_goals.append(scored)
        if opp_def_col in row and row[opp_def_col] > 55:
            vs_defensive_goals.append(scored)

    return {
        "vs_high_poss_goals":  np.mean(vs_high_poss_goals) if vs_high_poss_goals else 1.2,
        "vs_high_press_goals": np.mean(vs_high_press_goals) if vs_high_press_goals else 1.2,
        "vs_defensive_goals":  np.mean(vs_defensive_goals) if vs_defensive_goals else 1.2,
    }


def get_style_stats(row: pd.Series, prefix: str) -> dict:
    """
    Extract playing-style statistics from the raw DataFrame row.
    *prefix* is 'h' for home team, 'a' for away team.
    """
    def _safe(col, default=50.0):
        return row[col] if col in row.index and not pd.isna(row[col]) else default

    return {
        f"{prefix}_possession":        _safe(f"{prefix}_possession"),
        f"{prefix}_shots":             _safe(f"{prefix}_shots", 10.0),
        f"{prefix}_shots_on_target":   _safe(f"{prefix}_shots_on_target", 4.0),
        f"{prefix}_fouls":             _safe(f"{prefix}_fouls", 13.0),
        f"{prefix}_corners":           _safe(f"{prefix}_corners", 5.0),
        f"{prefix}_yellow_cards":      _safe(f"{prefix}_yellow_cards", 2.0),
        f"{prefix}_red_cards":         _safe(f"{prefix}_red_cards", 0.1),
    }


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build the full feature matrix from the raw match DataFrame.
    Iterates over each match and computes form, H2H, style, and matchup features.

    Returns:
        feature_df   — DataFrame of input features  (n_samples × n_features)
        y_class      — int array of result labels    (n_samples,)
        y_goals      — float array of goal counts    (n_samples × 2)
    """
    df = add_result_column(df)
    records = []

    print(f"Building features for {len(df)} matches …")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"  … {idx}/{len(df)}", end="\r", flush=True)

        date = row["date"]
        home_id = row["home_team_api_id"]
        away_id = row["away_team_api_id"]

        h_form = compute_team_form(df, home_id, date, n_matches=20)
        a_form = compute_team_form(df, away_id, date, n_matches=20)
        h2h    = compute_h2h_features(df, home_id, away_id, date, n_matches=10)
        h_style_matchup = compute_style_matchup(df, home_id, date, n_matches=20)
        a_style_matchup = compute_style_matchup(df, away_id, date, n_matches=20)
        h_style = get_style_stats(row, "h")
        a_style = get_style_stats(row, "a")

        feat = {}
        # Home form
        for k, v in h_form.items():
            feat[f"h_{k}"] = v
        # Away form
        for k, v in a_form.items():
            feat[f"a_{k}"] = v
        # Head-to-head
        feat.update(h2h)
        # Playing style (from current match row)
        feat.update(h_style)
        feat.update(a_style)
        # Style matchup
        for k, v in h_style_matchup.items():
            feat[f"h_{k}"] = v
        for k, v in a_style_matchup.items():
            feat[f"a_{k}"] = v
        # Team attribute differentials (home minus away)
        for col in ["h_speed", "h_passing", "h_chance_pass",
                    "h_chance_cross", "h_chance_shoot",
                    "h_def_pressure", "h_def_aggression", "h_def_width"]:
            a_col = col.replace("h_", "a_")
            if col in row.index and a_col in row.index:
                feat[f"diff_{col[2:]}"] = float(row[col]) - float(row[a_col])

        records.append(feat)

    print(f"\nFeature engineering complete. Columns: {len(records[0])}")
    feature_df = pd.DataFrame(records).fillna(0)
    y_class = df["result"].values.astype(int)
    y_goals = df[["home_team_goal", "away_team_goal"]].values.astype(float)
    return feature_df, y_class, y_goals


# =============================================================================
# SECTION 4 — Data Preprocessing
# =============================================================================

def preprocess_data(feature_df: pd.DataFrame,
                    y_class: np.ndarray,
                    y_goals: np.ndarray,
                    n_timesteps: int = 5
                    ) -> dict:
    """
    Scale features, one-hot-encode labels, split into train / val / test,
    and build sequences for the LSTM.

    Args:
        feature_df  : raw feature DataFrame
        y_class     : integer class labels (0, 1, 2)
        y_goals     : goal counts array (n_samples, 2)
        n_timesteps : window length for LSTM sequences

    Returns a dict with all necessary arrays.
    """
    X = feature_df.values.astype(float)
    n_features = X.shape[1]

    # --- One-hot encode classification labels ---
    n_classes = 3
    y_cat = keras.utils.to_categorical(y_class, num_classes=n_classes)

    # --- Train / Val / Test split (70 / 15 / 15) ---
    X_tv, X_test, yc_tv, yc_test, yr_tv, yr_test = train_test_split(
        X, y_cat, y_goals,
        test_size=0.15, random_state=42, stratify=y_class
    )
    strat_tv = np.argmax(yc_tv, axis=1)
    val_ratio = 0.15 / 0.85
    X_train, X_val, yc_train, yc_val, yr_train, yr_val = train_test_split(
        X_tv, yc_tv, yr_tv,
        test_size=val_ratio, random_state=42, stratify=strat_tv
    )

    # --- Standardise features ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # --- Build LSTM sequences (sliding window of n_timesteps rows) ---
    def make_sequences(X_arr, yc_arr, yr_arr, window):
        """
        Group consecutive rows into overlapping windows for LSTM input.
        For each index i >= window, the sequence is X_arr[i-window:i].
        Labels correspond to the last row of each window.
        """
        n = len(X_arr)
        if n <= window:
            # Not enough samples: repeat data to fill at least one window
            repeats = (window // n) + 2
            X_arr = np.tile(X_arr, (repeats, 1))[:window + 1]
            yc_arr = np.tile(yc_arr, (repeats, 1))[:window + 1]
            yr_arr = np.tile(yr_arr, (repeats, 1))[:window + 1]
            n = len(X_arr)

        Xs, Ycs, Yrs = [], [], []
        for i in range(window, n):
            Xs.append(X_arr[i - window:i])
            Ycs.append(yc_arr[i])
            Yrs.append(yr_arr[i])
        return (
            np.array(Xs),
            np.array(Ycs),
            np.array(Yrs)
        )

    Xl_train, ylc_train, ylr_train = make_sequences(
        X_train_s, yc_train, yr_train, n_timesteps)
    Xl_val,   ylc_val,   ylr_val   = make_sequences(
        X_val_s,   yc_val,   yr_val,   n_timesteps)
    Xl_test,  ylc_test,  ylr_test  = make_sequences(
        X_test_s,  yc_test,  yr_test,  n_timesteps)

    print("\nDataset split sizes:")
    print(f"  DNN  — train: {X_train_s.shape}, val: {X_val_s.shape}, "
          f"test: {X_test_s.shape}")
    print(f"  LSTM — train: {Xl_train.shape}, val: {Xl_val.shape}, "
          f"test: {Xl_test.shape}")

    return {
        # DNN inputs
        "X_train": X_train_s, "X_val": X_val_s,   "X_test": X_test_s,
        "yc_train": yc_train, "yc_val": yc_val,    "yc_test": yc_test,
        "yr_train": yr_train, "yr_val": yr_val,     "yr_test": yr_test,
        # LSTM inputs
        "Xl_train": Xl_train, "Xl_val": Xl_val,    "Xl_test": Xl_test,
        "ylc_train": ylc_train, "ylc_val": ylc_val, "ylc_test": ylc_test,
        "ylr_train": ylr_train, "ylr_val": ylr_val, "ylr_test": ylr_test,
        # metadata
        "n_features": n_features,
        "n_timesteps": n_timesteps,
        "scaler": scaler,
    }


# =============================================================================
# SECTION 5 — DNN Model
# =============================================================================

def build_dnn(n_features: int) -> Model:
    """
    Multi-output Deep Neural Network:
      Input(n_features)
      → Dense(256, relu) + Dropout(0.3)
      → Dense(128, relu) + Dropout(0.3)
      → Dense(64,  relu)
      → Head 1: Dense(3, softmax)   — match result classification
      → Head 2: Dense(2, linear)    — home & away goals regression
    """
    inputs = keras.Input(shape=(n_features,), name="dnn_input")
    x = layers.Dense(256, activation="relu", name="dnn_dense1")(inputs)
    x = layers.Dropout(0.3, name="dnn_drop1")(x)
    x = layers.Dense(128, activation="relu", name="dnn_dense2")(x)
    x = layers.Dropout(0.3, name="dnn_drop2")(x)
    x = layers.Dense(64,  activation="relu", name="dnn_dense3")(x)

    out_class = layers.Dense(3, activation="softmax",
                              name="result")(x)
    out_reg   = layers.Dense(2, activation="linear",
                              name="goals")(x)

    model = Model(inputs=inputs, outputs=[out_class, out_reg], name="DNN")
    return model


def train_dnn(data: dict, epochs: int = 100, batch_size: int = 32) -> tuple:
    """
    Compile and train the DNN model.
    Loss = 0.6 * categorical_crossentropy + 0.4 * mse
    Uses Adam optimizer and EarlyStopping (patience=10).

    Returns:
        model   — trained Keras Model
        history — training History object
    """
    model = build_dnn(data["n_features"])
    model.compile(
        optimizer="adam",
        loss={"result": "categorical_crossentropy", "goals": "mse"},
        loss_weights={"result": 0.6, "goals": 0.4},
        metrics={"result": "accuracy"}
    )
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1
    )

    history = model.fit(
        data["X_train"],
        {"result": data["yc_train"], "goals": data["yr_train"]},
        validation_data=(
            data["X_val"],
            {"result": data["yc_val"], "goals": data["yr_val"]}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history


# =============================================================================
# SECTION 6 — LSTM Model
# =============================================================================

def build_lstm(n_timesteps: int, n_features: int) -> Model:
    """
    Multi-output LSTM network:
      Input(n_timesteps, n_features)
      → LSTM(128) + Dropout(0.3)
      → LSTM(64)
      → Head 1: Dense(3, softmax)  — match result classification
      → Head 2: Dense(2, linear)   — home & away goals regression
    """
    inputs = keras.Input(shape=(n_timesteps, n_features), name="lstm_input")
    x = layers.LSTM(128, return_sequences=True, name="lstm1")(inputs)
    x = layers.Dropout(0.3, name="lstm_drop")(x)
    x = layers.LSTM(64, name="lstm2")(x)

    out_class = layers.Dense(3, activation="softmax",
                              name="result")(x)
    out_reg   = layers.Dense(2, activation="linear",
                              name="goals")(x)

    model = Model(inputs=inputs, outputs=[out_class, out_reg], name="LSTM")
    return model


def train_lstm(data: dict, epochs: int = 100, batch_size: int = 32) -> tuple:
    """
    Compile and train the LSTM model with the same loss and strategy as DNN.

    Returns:
        model   — trained Keras Model
        history — training History object
    """
    model = build_lstm(data["n_timesteps"], data["n_features"])
    model.compile(
        optimizer="adam",
        loss={"result": "categorical_crossentropy", "goals": "mse"},
        loss_weights={"result": 0.6, "goals": 0.4},
        metrics={"result": "accuracy"}
    )
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1
    )

    history = model.fit(
        data["Xl_train"],
        {"result": data["ylc_train"], "goals": data["ylr_train"]},
        validation_data=(
            data["Xl_val"],
            {"result": data["ylc_val"], "goals": data["ylr_val"]}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history


# =============================================================================
# SECTION 7 — Evaluation and Comparison
# =============================================================================

CLASS_NAMES = ["Away Win", "Draw", "Home Win"]


def evaluate_model(model: Model,
                   X_test: np.ndarray,
                   yc_test: np.ndarray,
                   yr_test: np.ndarray,
                   model_name: str) -> dict:
    """
    Evaluate a trained model on test data.

    Returns a dict with:
        accuracy, precision, recall, f1,
        mae_home, mae_away, rmse_home, rmse_away,
        confusion_matrix, y_true, y_pred_class, y_pred_goals
    """
    preds = model.predict(X_test, verbose=0)
    pred_class = np.argmax(preds[0], axis=1)
    pred_goals = preds[1]

    y_true = np.argmax(yc_test, axis=1)

    acc  = accuracy_score(y_true, pred_class)
    prec = precision_score(y_true, pred_class, average="weighted", zero_division=0)
    rec  = recall_score(y_true, pred_class, average="weighted", zero_division=0)
    f1   = f1_score(y_true, pred_class, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, pred_class, labels=[0, 1, 2])

    mae_home  = mean_absolute_error(yr_test[:, 0], pred_goals[:, 0])
    mae_away  = mean_absolute_error(yr_test[:, 1], pred_goals[:, 1])
    rmse_home = np.sqrt(np.mean((yr_test[:, 0] - pred_goals[:, 0]) ** 2))
    rmse_away = np.sqrt(np.mean((yr_test[:, 1] - pred_goals[:, 1]) ** 2))

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  MAE  (home goals): {mae_home:.4f}")
    print(f"  MAE  (away goals): {mae_away:.4f}")
    print(f"  RMSE (home goals): {rmse_home:.4f}")
    print(f"  RMSE (away goals): {rmse_away:.4f}")

    return {
        "name":          model_name,
        "accuracy":      acc,
        "precision":     prec,
        "recall":        rec,
        "f1":            f1,
        "mae_home":      mae_home,
        "mae_away":      mae_away,
        "rmse_home":     rmse_home,
        "rmse_away":     rmse_away,
        "cm":            cm,
        "y_true":        y_true,
        "y_pred_class":  pred_class,
        "y_pred_goals":  pred_goals,
        "y_true_goals":  yr_test,
    }


# =============================================================================
# SECTION 8 — Visualization
# =============================================================================

def plot_loss_curves(dnn_history, lstm_history, save_path: str = "loss_curves.png"):
    """
    Plot training vs validation loss curves for both models side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hist, title in zip(
        axes,
        [dnn_history, lstm_history],
        ["DNN — Loss Curves", "LSTM — Loss Curves"]
    ):
        ax.plot(hist.history["loss"],     label="Train Loss",      linewidth=2)
        ax.plot(hist.history["val_loss"], label="Validation Loss", linewidth=2,
                linestyle="--")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices(dnn_results: dict, lstm_results: dict,
                             save_path: str = "confusion_matrices.png"):
    """
    Plot confusion matrices for both models side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res in zip(axes, [dnn_results, lstm_results]):
        sns.heatmap(
            res["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5
        )
        ax.set_title(f"{res['name']} — Confusion Matrix", fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_metric_comparison(dnn_results: dict, lstm_results: dict,
                            save_path: str = "metric_comparison.png"):
    """
    Bar chart comparing all key metrics between DNN and LSTM.
    """
    metrics = ["accuracy", "precision", "recall", "f1",
               "mae_home", "mae_away", "rmse_home", "rmse_away"]
    labels  = ["Accuracy", "Precision", "Recall", "F1",
               "MAE Home", "MAE Away", "RMSE Home", "RMSE Away"]

    dnn_vals  = [dnn_results[m]  for m in metrics]
    lstm_vals = [lstm_results[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, dnn_vals,  width, label="DNN",  color="#4C72B0")
    bars2 = ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("DNN vs LSTM — Metric Comparison", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_goals_scatter(dnn_results: dict, lstm_results: dict,
                       save_path: str = "goals_scatter.png"):
    """
    Scatter plot of actual vs predicted goals for both models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["Home Goals", "Away Goals"]
    colors = ["#4C72B0", "#DD8452"]

    for col, (res, color) in enumerate(zip([dnn_results, lstm_results], colors)):
        for row, goal_idx in enumerate([0, 1]):
            ax = axes[row][col]
            actual = res["y_true_goals"][:, goal_idx]
            pred   = res["y_pred_goals"][:, goal_idx]
            ax.scatter(actual, pred, alpha=0.5, color=color, edgecolors="k",
                       linewidths=0.3, s=40)
            lim_max = max(actual.max(), pred.max()) + 0.5
            ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=1.5,
                    label="Perfect prediction")
            ax.set_title(f"{res['name']} — {titles[goal_idx]}", fontsize=12)
            ax.set_xlabel("Actual Goals")
            ax.set_ylabel("Predicted Goals")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(dnn_results: dict, lstm_results: dict):
    """
    Print a formatted side-by-side comparison table of both models.
    """
    print("\n" + "=" * 65)
    print(f"{'FINAL MODEL COMPARISON':^65}")
    print("=" * 65)
    header = f"{'Metric':<25} {'DNN':>15} {'LSTM':>15}"
    print(header)
    print("-" * 65)

    rows = [
        ("Accuracy",      "accuracy",  "{:.4f}"),
        ("Precision (W)", "precision", "{:.4f}"),
        ("Recall (W)",    "recall",    "{:.4f}"),
        ("F1-Score (W)",  "f1",        "{:.4f}"),
        ("MAE Home Goals","mae_home",  "{:.4f}"),
        ("MAE Away Goals","mae_away",  "{:.4f}"),
        ("RMSE Home Goals","rmse_home","{:.4f}"),
        ("RMSE Away Goals","rmse_away","{:.4f}"),
    ]
    for label, key, fmt in rows:
        dnn_v  = fmt.format(dnn_results[key])
        lstm_v = fmt.format(lstm_results[key])
        print(f"{label:<25} {dnn_v:>15} {lstm_v:>15}")

    print("=" * 65)


# =============================================================================
# SECTION 9 — Save Models
# =============================================================================

def save_models(dnn_model: Model, lstm_model: Model,
                dnn_path: str = "dnn_model.h5",
                lstm_path: str = "lstm_model.h5"):
    """Save both trained models to .h5 files."""
    dnn_model.save(dnn_path)
    lstm_model.save(lstm_path)
    print(f"\nModels saved:\n  DNN  → {dnn_path}\n  LSTM → {lstm_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  Football Match Prediction System")
    print("  BIM447 – Deep Learning ESTÜ 2026")
    print(f"  USE_FULL_DATASET = {USE_FULL_DATASET}")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    df = load_data()
    print(f"\nDataset shape: {df.shape}")
    print(df.head(3).to_string())

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    feature_df, y_class, y_goals = build_feature_matrix(df)
    print(f"\nFeature matrix shape: {feature_df.shape}")
    print("Class distribution:", dict(zip(*np.unique(y_class, return_counts=True))))

    # ------------------------------------------------------------------
    # 4. Preprocessing
    # ------------------------------------------------------------------
    # Use a shorter time window when data is limited
    n_timesteps = min(5, max(1, len(df) // 20))
    data = preprocess_data(feature_df, y_class, y_goals,
                           n_timesteps=n_timesteps)

    # ------------------------------------------------------------------
    # 5. DNN training
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Training DNN …")
    print("-" * 40)
    dnn_model, dnn_history = train_dnn(data, epochs=100, batch_size=32)

    # ------------------------------------------------------------------
    # 6. LSTM training
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Training LSTM …")
    print("-" * 40)
    lstm_model, lstm_history = train_lstm(data, epochs=100, batch_size=32)

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    dnn_results  = evaluate_model(
        dnn_model,
        data["X_test"], data["yc_test"], data["yr_test"],
        "DNN"
    )
    lstm_results = evaluate_model(
        lstm_model,
        data["Xl_test"], data["ylc_test"], data["ylr_test"],
        "LSTM"
    )

    # ------------------------------------------------------------------
    # 8. Visualization
    # ------------------------------------------------------------------
    plot_loss_curves(dnn_history, lstm_history)
    plot_confusion_matrices(dnn_results, lstm_results)
    plot_metric_comparison(dnn_results, lstm_results)
    plot_goals_scatter(dnn_results, lstm_results)
    print_summary_table(dnn_results, lstm_results)

    # ------------------------------------------------------------------
    # 9. Save models
    # ------------------------------------------------------------------
    save_models(dnn_model, lstm_model)

    print("\nDone! All outputs saved.")


if __name__ == "__main__":
    main()

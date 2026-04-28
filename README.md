# MatchPredictorDeepLearning2026Project
BIM447 - Deep Learning ESTÜ 2026 Group Project

## Overview

A complete football match prediction system that uses deep learning to predict:

1. **Match result** (Home Win / Draw / Away Win) — multi-class classification
2. **Goal counts** (home goals, away goals) — regression

Two model architectures are implemented and compared:

| Model | Architecture |
|-------|-------------|
| **DNN** | Dense(256) → Dense(128) → Dense(64) → two output heads |
| **LSTM** | LSTM(128) → LSTM(64) → two output heads |

---

## Dataset

[Kaggle European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) — match results, team attributes, and player ratings across European leagues (2008–2016).

Download the `database.sqlite` file and place it next to `match_predictor.py`, then set `USE_FULL_DATASET = True` inside the script.

When `USE_FULL_DATASET = False` (default), the script generates **100 synthetic samples** so you can run it immediately without downloading the dataset.

---

## Project Structure

```
MatchPredictorDeepLearning2026Project/
├── match_predictor.py   # Main script — all sections in one file
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Demo mode (100 synthetic samples — no dataset required)
python match_predictor.py

# Full mode (requires database.sqlite)
# Edit match_predictor.py and set USE_FULL_DATASET = True
python match_predictor.py
```

---

## Features Engineered

For each match, the following features are computed from the **last 20 matches** for each team:

### Form & Performance
- Win / Draw / Loss rate
- Average goals scored & conceded
- Weighted form score (recent matches weighted more)
- First half vs second half goal averages

### Playing Style Statistics
- Ball possession (%), shots per game, shots on target rate
- Fouls, corners, yellow/red cards per game

### Head-to-Head (H2H)
- Win rate in last 10 encounters
- Average goals scored/conceded vs this opponent
- Home/Away split in H2H

### Style Matchup Features
- Performance vs high-possession teams (>60% possession)
- Performance vs high-pressing teams
- Goals scored vs defensive teams

### Home / Away Effect
- Separate home win rate and away win rate

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Split | 70% train / 15% val / 15% test |
| Optimizer | Adam |
| Classification loss | categorical_crossentropy |
| Regression loss | mean_squared_error |
| Combined loss | 0.6 × classification + 0.4 × regression |
| Early stopping | patience = 10 |
| Batch size | 32 |
| Max epochs | 100 |

---

## Outputs

After running the script the following files are generated:

| File | Description |
|------|-------------|
| `loss_curves.png` | Training vs validation loss (both models) |
| `confusion_matrices.png` | Confusion matrices side by side |
| `metric_comparison.png` | Bar chart comparing all metrics |
| `goals_scatter.png` | Actual vs predicted goals scatter plots |
| `dnn_model.h5` | Saved DNN model |
| `lstm_model.h5` | Saved LSTM model |

---

## Code Structure

| Section | Description |
|---------|-------------|
| 1 | Imports and setup (`USE_FULL_DATASET` flag) |
| 2 | Data loading (real SQLite DB or synthetic fallback) |
| 3 | Feature engineering functions |
| 4 | Data preprocessing (scaling, encoding, LSTM sequences) |
| 5 | DNN model definition and training |
| 6 | LSTM model definition and training |
| 7 | Evaluation and comparison |
| 8 | Visualization |
| 9 | Save both models (.h5) |

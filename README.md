# PGA Fantasy Golf

A machine learning pipeline for predicting PGA Tour player performance and generating optimal weekly lineups for [Buzz Fantasy Golf](https://buzzfantasygolf.com).

## How it works

1. **Historical data** (2014–2022) is loaded and standardized — strokes gained stats, course history, cut rates.
2. **Feature engineering** computes rolling averages (8/16/24 tournament windows) per player with no look-ahead leakage.
3. **XGBoost model** is trained on historical data using expanding-window cross-validation (train on years ≤ N, test on N+1). Average Spearman rank correlation ≈ 0.26.
4. **Weekly scraper** fetches the live field and each player's recent form from the PGA Tour GraphQL API.
5. **Optimizer** solves a salary-cap integer program (PuLP) to maximize projected performance within the weekly budget.

## Project structure

```
src/
├── ingest.py      # Load and standardize historical CSV data
├── features.py    # Rolling SG averages, course history, cut rate
├── pipeline.py    # Build historical feature matrix
├── model.py       # XGBoost training and cross-validation
├── scrape.py      # PGA Tour GraphQL API fetcher
├── predict.py     # Weekly inference
├── buzz.py        # Buzz Fantasy Golf salary parser
└── optimizer.py   # PuLP lineup optimizer
```

## Setup

```bash
pip install -r requirements.txt
```

You'll also need the historical data CSV (`14-22 practice data.csv`) in the project root to build the training feature matrix and model.

## Weekly workflow

**1. (One-time) Build the historical model**
```bash
python -m src.pipeline          # builds data/processed/feature_matrix.parquet
python -m src.model             # trains and saves data/processed/model.pkl
```

**2. Every week before the tournament**
```bash
# Scrape fresh field + player form, then score
python -m src.predict --tournament R2026026 --top 30

# Optimize lineups against your salary file
python -m src.optimizer --csv salaries.csv --budget 1650000 --lineups 3
```

Replace `R2026026` with the current tournament ID (visible in the PGA Tour schedule URL) and adjust `--budget` to match that week's Buzz cap.

**Salary file format** (`salaries.csv`):
```
name,salary
Scottie Scheffler,700000
Rory McIlroy,577000
...
```
The parser also handles the copy-paste block format directly from the Buzz website.

## Optimizer options

```bash
# Force a player into every lineup
python -m src.optimizer --csv salaries.csv --budget 1650000 --lock "Scheffler"

# Exclude a player
python -m src.optimizer --csv salaries.csv --budget 1650000 --exclude "Smith, Cameron"

# Generate 5 alternative lineups
python -m src.optimizer --csv salaries.csv --budget 1650000 --lineups 5
```

## Pool rules (Buzz Fantasy Golf)

- Pick **6 players** per week
- Total salary must be within the weekly cap (varies by field size/strength)
- Scoring is based on **real prize money** earned at the tournament

## Notes

- Players with no recent SG data (amateurs, qualifiers) are automatically excluded from projections.
- Course history features are populated from the historical dataset. Rare venues (e.g. Shinnecock Hills) will have sparse history.
- The model was trained to predict DraftKings fantasy points as a proxy for performance. Retraining on prize money directly is a future improvement.

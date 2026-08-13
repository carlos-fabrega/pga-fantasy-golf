# PGA Fantasy Golf

A machine learning pipeline for predicting PGA Tour player performance and generating optimal weekly lineups for [Buzz Fantasy Golf](https://buzzfantasygolf.com).

## How it works

1. **Historical data** (2015–2022, 36,843 player-tournament rows) is loaded and standardized — strokes gained stats, course history, cut rates.
2. **Feature engineering** computes rolling averages (8/16/24 tournament windows) per player. Every window is computed with `shift(1)` applied first, so a tournament's features only ever see earlier tournaments — no look-ahead leakage.
3. **XGBoost `XGBRanker`** (LambdaRank, `rank:ndcg`) is trained to rank a field by `prize_pct` — each player's share of the tournament purse — rather than regressing on a fantasy-points proxy. Validated with expanding-window cross-validation: train on seasons ≤ N, test on N+1.
4. **Weekly scraper** fetches the live field and each player's recent form from the PGA Tour GraphQL API.
5. **Optimizer** solves a salary-cap integer program (PuLP) to maximize projected performance within the weekly budget.

## Results

Expanding-window temporal CV, scored by per-tournament Spearman rank correlation (ranking is what
lineup selection needs — not absolute error):

| Test season | Train | Rows | Tournaments | Spearman ρ (mean) |
|---|---|---|---|---|
| 2018 | 2015–2017 | 4,821 | 34 | +0.274 |
| 2019 | 2015–2018 | 5,005 | 35 | +0.285 |
| 2020 | 2015–2019 | 4,179 | 29 | +0.279 |
| 2021 | 2015–2020 | 5,576 | 38 | +0.322 |
| 2022 | 2015–2021 | 3,646 | 26 | +0.282 |
| **Overall** | | | | **+0.288** (σ 0.019) |

Reproduce with `python -m src.model --cv`.

Two tournament-type model variants were built and backtested against this baseline — a links-course
variant and a FedEx Cup Playoffs variant. The links variant is a marginal improvement; the playoff
variant **underperformed the base model by ~21%** on playoff events (0.259 vs 0.329 on those rows),
since ~23 historical playoff editions is far too thin to train on. It's kept behind an explicit
`--playoff` opt-in rather than auto-selected. Negative results are cheaper to keep than to rediscover.

## Project structure

```
src/
├── paths.py       # Single source of truth for all filesystem paths
├── ingest.py      # Load and standardize historical CSV data
├── features.py    # Rolling SG averages, course history, cut rate
├── pipeline.py    # Build historical feature matrix
├── model.py       # XGBRanker training, temporal CV, model variants
├── scrape.py      # PGA Tour GraphQL API fetcher
├── predict.py     # Weekly inference
├── buzz.py        # Buzz Fantasy Golf salary parser
└── optimizer.py   # PuLP lineup optimizer
```

## Setup

```bash
pip install -r requirements.txt
```

You'll also need the historical data CSV at `data/raw/historical/14-22 practice data.csv` to build the training feature matrix and model. It's gitignored and not fetched by any script, so it must already exist locally.

## Weekly workflow

**1. (One-time) Build the historical model**
```bash
python -m src.pipeline          # builds data/processed/feature_matrix.parquet
python -m src.model             # trains and saves models/model_base.pkl
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
- `src/scrape.py` uses PGA Tour's public web-client API key, which ships in their site bundle. Override it with `PGA_API_KEY` if they rotate it.

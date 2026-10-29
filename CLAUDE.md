# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An ML pipeline that predicts PGA Tour player performance and optimizes weekly fantasy lineups for
[Buzz Fantasy Golf](https://buzzfantasygolf.com). It ranks a tournament field, projects each
player's share of the purse, then solves a salary-cap integer program for the best lineups.

## Setup

```bash
pip install -r requirements.txt
```

Requires `data/raw/historical/14-22 practice data.csv` (historical 2014–2022 results). It's
gitignored and not fetched by any script — it must already exist locally.

## Commands

One-time / whenever historical or scraped data changes:
```bash
python -m src.pipeline        # ingest + feature-engineer → data/processed/feature_matrix.parquet
python -m src.model           # train XGBRanker on all seasons → models/model_base.pkl
python -m src.model --links   # train the links-course-tuned variant → models/model_links.pkl
python -m src.model --nocut   # train the no-cut variant (drops made_cut_rate_*) → models/model_nocut.pkl
python -m src.model --cv      # expanding-window temporal CV report only (no save)
```

Weekly, before a tournament:
```bash
python -m src.predict                        # auto-detect next event; links/no-cut model auto-selected
python -m src.predict --tournament R2026030  # score a specific event instead
python -m src.predict --no-links             # force the base model on a links event
python -m src.predict --no-nocut             # force the base model on a no-cut event
python -m src.predict --latest               # re-score most recent saved features (no scrape)
python -m src.optimizer --csv data/processed/R2026030_john_deere_classic/salaries.csv \
                        --budget 1650000 --lineups 3
```

`predict` defaults to the next upcoming event (via the PGA API) and auto-selects the links model
for Open/Scottish/Dunhill/Irish, or the no-cut model for limited-field no-cut events (TOUR
Championship, BMW, Sentry, the old WGCs) — pass `--tournament`, `--no-links`/`--links`, or
`--no-nocut`/`--nocut` to override. Links is checked first; the two sets don't currently overlap.

Analyst-agent database (see `PGA_AGENT_PLAN.md`) — rebuild after each new weekly run or retrain:
```bash
python -m agent.db --fetch   # cache scored events' leaderboards + schedule (network), then build
python -m agent.db           # rebuild data/processed/agent.sqlite from local files only
```
Ask the agent a question (needs `ANTHROPIC_API_KEY`; every run is logged to `traces/YYYY-MM-DD.jsonl`):
```bash
python -m agent.cli                                    # conversation: /new starts over, /quit exits
python -m agent.cli "Why is Scheffler ranked below Si Woo Kim this week?"   # one question
python -m agent.cli --model claude-haiku-5-5           # cheaper model for smoke tests
```
Lineups the agent proposes are only saved if you answer `y` at the CLI's approval prompt, as
`lineups_approved_{proposal_id}.csv` in that week's `data/processed/{tid}_{slug}/` folder. The agent
has no save tool; keep it that way.

Agent evals (`evals/`, see `PGA_AGENT_PLAN.md` §9). Regenerate the cases after rebuilding the agent DB,
since expected answers come from the current data. A run calls the API once per case, so it costs money:
```bash
python -m evals.generate                                  # evals/questions.jsonl from agent.sqlite
python -m evals.run_evals --model claude-haiku-5-5        # all 44 cases → evals/results/{stamp}_{model}.md/.json
python -m evals.run_evals --ids expl-01,lu-02             # or --category lineup
```

`agent/db.py` links the two player-ID spaces (training CSV vs PGA API, e.g. Scheffler 9478 vs
46046) by name; unlinked near-misses land in `data/processed/player_crosswalk_review.csv`.

`python -m pytest tests/` covers the agent DB, tools, loop, CLI, guardrails and eval graders only (no API calls: the loop tests use a fake client). For the model, verify changes by re-running
`python -m src.model --cv` and checking the mean Spearman rank correlation (≈0.28 is normal), or
by running predict → optimizer end-to-end against a known tournament ID and eyeballing the output.

## File layout

```
src/                 # one module per pipeline stage (see below)
  paths.py           # SINGLE source of truth for all filesystem paths
agent/               # analyst agent; db.py builds the read-only SQLite DB, tools.py the agent's tools,
                     # prompts.py the system prompt, agent.py the loop, cli.py the CLI,
                     # guardrails.py the numeric-claim checker + lineup validator
evals/               # generate.py builds questions.jsonl, grade.py the graders, run_evals.py the runner;
                     # results/ holds one report per run
models/              # trained pickles: model_base.pkl, model_links.pkl (gitignored)
data/
  raw/
    historical/      # 14-22 practice data.csv (the 2014–2022 training source)
    scraped/         # R{year}{id}.csv from the external scraper
    results/         # cached live leaderboards + schedule.csv (agent.db --fetch)
  processed/
    feature_matrix.parquet
    agent.sqlite, player_crosswalk_review.csv
    {tournament_id}_{slug}/   # per-week artifacts, all together:
        weekly_features.parquet, projections.csv, lineups*.csv, salaries.csv,
        lineups_approved_*.csv (agent CLI)
```

Each week's inputs and outputs — salaries, features, projections, lineups — live together under
`data/processed/{tournament_id}_{slug}/` so past results stay browsable without regenerating.

**All paths are defined in `src/paths.py`** and anchored to the repo root, so scripts run from any
working directory. When adding a file location, add it there — do not hardcode path strings in the
stage modules.

## Architecture

Data flows through a fixed pipeline, one module per stage under `src/`:

1. **`ingest.py`** — standardizes two disjoint sources into one schema: `load_historical()` (the
   2014–2022 CSV — finish position, purse, DK/FD points, SG as per-round averages) and
   `load_scraped()` (`data/raw/scraped/R{year}{id}.csv`, SG only, made-cut players only, from an
   external scraper). `combine()` dedupes on `(player_id, tournament_id, season)`, preferring
   historical rows where the two sources overlap.

2. **`features.py`** — every rolling/history feature is computed with `shift(1)` *before* the
   window, so tournament T's features only ever see T-1 and earlier. **This leakage-safety
   invariant is what the whole model's validity depends on — preserve it in any new feature.**
   Windows are 8/16/24 tournament starts (not calendar weeks). `is_links_tournament()` flags The
   Open, Scottish Open, Dunhill Links, and Irish Open for the separate links feature set;
   `is_nocut_tournament()` flags limited-field no-cut events for the nocut feature set.

3. **`pipeline.py`** — orchestrates ingest → features → `data/processed/feature_matrix.parquet`.

4. **`model.py`** — `XGBRanker` with a `rank:ndcg` objective, **not regression**: `prize_pct` is
   bucketed into 0–5 relevance grades (`_to_grade`) because LambdaRank needs discrete relevance
   rather than a continuous target. Trained variants: base (`FEATURE_COLS`), links
   (`FEATURE_COLS_LINKS`, adds `links_sg_hist`/`links_starts`), and nocut (`FEATURE_COLS_NOCUT`,
   base minus `made_cut_rate_*` — at a no-cut event every starter is paid, so made-cut rate encodes
   a risk that doesn't exist; CV shows dropping it costs nothing on full-field events, ρ 0.275 vs
   0.278, well inside fold noise). Evaluated by per-tournament Spearman
   rank correlation, not RMSE, since ranking — not absolute score — is what lineup selection needs.
   `temporal_cv()` always trains on seasons < N and tests on season N; **never switch this to a
   random split** — the rolling features would leak future data across the season boundary.

5. **`scrape.py`** — hits the PGA Tour GraphQL API directly (`orchestrator.pgatour.com/graphql`);
   the old stats-download CSV endpoint stopped working mid-2025. `build_weekly_features()` builds
   the live weekly feature matrix that `predict.py` scores.

6. **`predict.py`** — scores a weekly field with the saved model, writing
   `projections.csv` (or `projections_links.csv`) into the tournament folder.

7. **`buzz.py`** + **`optimizer.py`** — `buzz.py` fetches/parses Buzz Fantasy Golf salaries (cookie-
   authenticated API, or a manual CSV) and normalizes player names (strips accents/punctuation,
   converts "Last, First" → "first last") so they join against projections by fuzzy name match.
   `optimizer.py` is a PuLP integer program maximizing projected prize share under a salary cap,
   supporting locked/excluded players and generating N alternative lineups (each differing from
   every prior one by at least 1 player). Lineup files land in the same tournament folder; distinct
   model-variant and lock/exclude runs get encoded filenames so they don't overwrite each other.

## Conventions

- Add new paths to `src/paths.py`, never as inline string literals.
- `models/` is gitignored, so a stale pickle can silently outlive the feature matrix it was
  trained on. Retrain every variant after regenerating `feature_matrix.parquet`.
- Preserve the `shift(1)`-before-window leakage guard in `features.py`.
- Keep temporal (not random) splits anywhere the model touches season boundaries.
- The model targets `prize_pct` (share of purse), bucketed to relevance grades — not raw
  DraftKings points and not regression.

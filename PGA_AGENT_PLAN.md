# PGA Analyst Agent: Build Plan

An LLM agent that answers plain-English questions about the PGA model by calling the existing code (features, model, optimizer) as tools. The goal is a project that shows the skills applied-AI teams screen for: **tool calling, context engineering, evaluation, guardrails, and human-in-the-loop control**.

---

## 1. What "done" looks like

- A CLI where you can ask questions like:
  - "Why is Scheffler ranked above Rahm this week?"
  - "Who are the best values relative to salary?"
  - "Build 3 lineups, lock Morikawa, exclude anyone who missed the cut last week."
  - "How accurate has the model been this season?"
- Every number in an answer comes from a tool call. The agent never invents stats.
- An eval suite of ~30 questions with known answers, run with one command, producing a scored report.
- A README with architecture, eval results (before → after), and lessons learned.

**Non-goals:** a web UI, retraining the model, or multi-agent setups. Keep it small and solid.

---

## 2. Step 0: Map the existing repo (do this first)

Before writing code, have Claude Code inventory what already exists and fill in this table. Everything below assumes these pieces exist in some form. Adjust names to match the repo.

| Piece | Where it lives today | Notes |
|---|---|---|
| Player-tournament features (strokes-gained windows, course history, cut rate) | **Training:** `src/features.py` `build_features(df)`, run by `src/pipeline.py` `run()` → `data/processed/feature_matrix.parquet` (36,843 rows, 2015–2022, 26 model features). **Live:** `src/scrape.py` `build_weekly_features(tournament_id, historical_df)` → `data/processed/{tid}_{slug}/weekly_features.parquet` (7 weeks of 2026 on disk). | Both already plain functions. Feature lists per variant: `model.VARIANT_FEATURES`. Matrix has **0 scraped rows**: `data/raw/scraped/` is empty, so nothing after Jun 2022. |
| Trained XGBoost LambdaRank model artifact | `models/model_{base,links,nocut}.pkl` (pickled `XGBRanker`, xgboost 2.1.4). Paths in `src/paths.py`. | `model.load(variant)` returns the model. `model_links.pkl` is stale (Jul 14, before the matrix was rebuilt Aug 13). `model_playoff.pkl` is not on disk. SHAP: `model.get_booster().predict(DMatrix(X), pred_contribs=True)` works as-is. |
| Weekly inference script (PGA Tour GraphQL API) | `src/predict.py`: `predict(weekly_df, variant)` → DataFrame; `main()` = CLI. GraphQL client in `src/scrape.py` (`get_upcoming_tournaments`, `get_field`, `get_player_sg_history`, `get_player_cut_history`). | `predict()` is callable, but choosing the variant, dropping players with no SG data, and writing `projections[_variant].csv` all happen inline in `main()`. **`projected_prize_pct` is a raw LambdaRank score (e.g. 1.34), not a share of the purse.** |
| Salary data source | `src/buzz.py` `load_salaries(csv_path=..., cookie=...)` / `load_salaries_csv(path)` → `[name, salary, name_key]`. Files: `data/processed/{tid}_{slug}/salaries.csv` (Buzz copy-paste block format, 7 weeks), `data/raw/salaries_mock.csv`. | Callable as-is. Salaries carry **no player_id**; they join to projections only by `normalize_name` plus fuzzy matching. The cookie API endpoints are guesses that have never been confirmed. |
| PuLP lineup optimizer | `src/optimizer.py`: `build_lineups(merged_df, n_lineups, budget, n_players, locks, excludes)` → list of DataFrames; `_match_names(proj, sal)` joins them. `main()` = CLI; writes `lineups*.csv` into the tournament folder. | `build_lineups` is callable, but the projection+salary join is private and lives in `main()`. Locks and excludes match on **name substrings** (`"kim"` would lock every Kim), and a lock that matches nobody is silently ignored. |
| Historical results (finish position, purse share) | Only inside `feature_matrix.parquet` (`finish_num`, `made_cut`, `prize_pct`, 2015–2022). Payout table: `ingest._pos_to_prize_pct`. Live leaderboards: `scrape.get_tournament_results(tid, year)`. | **No results for 2023+, including the 7 scored 2026 weeks.** `src/backfill.py` was written for this but has never been run, and its output doesn't fit `ingest.load_scraped`: the glob is `R????.csv` vs `R2024493.csv` filenames, the column mapping is for the old CSV format, `tid` parsing is wrong, `made_cut` is forced to True, and no `prize_pct` is computed. No past predictions are stored, apart from those 7 weeks of `projections.csv`. |

**Cross-cutting finding: the player IDs don't match between sources.** The training data uses one ID space (Scheffler = `9478`) and the PGA GraphQL API uses another (Scheffler = `46046`). None of the IDs in any weekly field appear in the feature matrix. Effects:
- The `players` table needs a name-based crosswalk. `player_id` alone can't join history to live data.
- At serve time this silently breaks `course_sg_hist`, `course_starts`, and the historical half of `links_sg_hist`/`playoff_sg_hist`, because `build_weekly_features` filters `historical_df` on the API id. That's why course history is always NaN in production. It's a separate bug; fixing it is out of scope for the agent.

### 2a. Proposed minimal refactor (pending approval)

Keep `src/` as the package; renaming it to `pga/` would only add churn. Every change pulls logic that's already there out of a `main()` into a function, so CLI behavior stays the same.

1. **`predict.py`**: extract three functions from `main()`.
   - `select_variant(tournament_name, links=None, nocut=None, playoff=None) -> str`
   - `align_features(weekly_df, model) -> X`: the "add missing columns as NaN" step, so SHAP and scoring see the same matrix.
   - `score_field(weekly_df, variant) -> DataFrame`: the no-SG filter plus `predict()`.

   Also make `_tournament_dir` public. `main()` becomes glue.
2. **`optimizer.py`**: extract `optimize(proj_df, sal_df, n_lineups, budget, n_players, locks, excludes) -> (lineups, merged)` from `main()`, and make `_match_names` public as `match_salaries`. Let `locks`/`excludes` take `player_id`s, which are exact, alongside the existing name-substring path, which the CLI keeps.
3. **`model.py`**: add `oof_predictions(df, variant) -> DataFrame[player_id, tournament_id, season, score, prize_pct]`, reusing the `temporal_cv` fold loop. This gives the `predictions` backfill honest out-of-fold scores. Scoring past events with the final model would be in-sample.
4. **`ingest.py`**: make `_pos_to_prize_pct` public, so 2026 leaderboards from `get_tournament_results` can be turned into `prize_pct` for the `results` table.

**Not needed:** `build_weekly_features`, `load_salaries`, `normalize_name`, `model.load`, `pipeline.run`/`load_processed` and `get_tournament_results` are already plain functions. Their `print()` noise can be handled by `contextlib.redirect_stdout` in `agent/tools.py`, without touching the modules.

**Deferred (decide separately):** fixing the `backfill.py` ↔ `ingest.load_scraped` contract and running the 2023–2026 backfill. This changes the training data and so needs every variant retrained. Until then, "this season" accuracy is limited to the 7 scored 2026 weeks, using results pulled directly with `get_tournament_results`.

**Prompt to start Claude Code with:**

> Read PGA_AGENT_PLAN.md. Don't write code yet. Inventory the repo and fill in the table in section 2: where features, the model, inference, salaries, the optimizer, and results live, and how they're called. Then propose the minimal refactor needed so each piece can be called as a plain Python function.

---

## 3. Architecture

```
pga/                      # existing code
agent/
  db.py                   # DuckDB/SQLite access layer (read-only)
  tools.py                # tool functions + JSON schemas
  agent.py                # the agent loop (written by hand, not a framework)
  prompts.py              # system prompt
  guardrails.py           # name resolution, numeric-claim checker
  cli.py                  # interactive CLI with approval step
evals/
  questions.jsonl         # eval cases
  run_evals.py            # runs agent on every case, scores, writes report
  results/                # timestamped JSON + markdown reports
traces/                   # JSONL log of every run (messages + tool calls)
```

**Why write the loop by hand:** it's ~60 lines, and being able to explain exactly how messages, tool calls, and tool results flow is a strong interview answer. Use the Anthropic Python SDK's Messages API with tool use.

### Data layer (`db.py`)

Consolidate into one local DuckDB (or SQLite) file with these tables:

- `features(player_id, event_id, date, sg_8, sg_16, sg_24, course_hist, cut_rate, ...)`
- `predictions(player_id, event_id, model_version, score, rank)`: **backfill** this by running the model over past events, so "how has the model done" questions have data
- `results(player_id, event_id, finish, purse_share, made_cut)`
- `salaries(player_id, event_id, salary)`
- `players(player_id, name, aliases)`

The DB connection is **read-only** for the agent.

---

## 4. Tools

Prefer specific, parameterized tools over a raw SQL tool. They're safer, easier to evaluate, and the LLM picks the right one more often.

| Tool | Inputs | Returns | Notes |
|---|---|---|---|
| `resolve_player` | `name: str` | `player_id`, canonical name, or a list of candidates if ambiguous | Fuzzy match (e.g. `rapidfuzz`). If ambiguous, the agent must ask the user. *Built:* stdlib `difflib` (rapidfuzz isn't installed); exact → name-prefix → last/first name → fuzzy, and a fuzzy hit always comes back as a candidate to confirm, never auto-resolved. |
| `get_rankings` | `event_id` (default: upcoming), `top_n` | ranked players with model score | *Built:* every event tool takes `event` (an id or part of a tournament name) + optional `season`; the default is the latest scored event. Picks the variant `predict` would auto-select, falling back to base. |
| `get_player_features` | `player_id`, `event_id` | feature values for that event | |
| `explain_prediction` | `player_id`, `event_id`, `compare_to?` | top feature contributions | Use XGBoost `predict(..., pred_contribs=True)` for real SHAP contributions. Not LLM guesswork. *Built:* live weeks only, since backtest fold models aren't saved. It refuses with `model_changed` when the saved model no longer reproduces the stored score; as of Oct 2026 that's every base-model week except the TOUR Championship. |
| `value_report` | `event_id`, `top_n` | players ranked by model score per $1K salary | *Built:* ranks by `salary_rank - model_rank` instead, because most weekly scores are negative ranker margins (115 of 144 at Wyndham), so score per $1K would favour expensive weak players. Live weeks with salaries only. |
| `propose_lineups` | `event_id`, `n_lineups`, `locks[]`, `excludes[]`, `rules{}` | proposal with `proposal_id`, lineups, total salary, projected score | Calls the PuLP optimizer. **Does not save anything.** *Built:* reuses `optimizer.build_lineups` on the DB's stored scores and salaries; locks and excludes are player_ids; `rules{}` is just `budget` (default $1.65M); 1–5 lineups. Checks impossible locks before solving. |
| `model_performance` | `season` or `event_ids[]` | mean Spearman, hit rates, best/worst events | Computed from `predictions` vs `results` *Built:* takes `event` or `season` (neither gives a per-season summary). Spearman reuses `model._spearman_by_tournament`, so it matches `--cv` (backtest seasons average 0.26–0.32; 2026 live 0.18 mean, 0.25 median over 7 events). Also reports top-10 hits and where the model ranked the winner. |

Each tool:
- Has a JSON schema with clear descriptions (the description is part of your context engineering).
- Returns compact JSON. Trim to what's needed, since large tool outputs waste context and confuse the model.
- Returns structured errors (`{"error": "no_data", "detail": "..."}`) instead of raising.

---

## 5. Agent loop (`agent.py`)

1. Build messages: system prompt + user question.
2. Call the model with the tool schemas.
3. If the response contains tool calls, run each tool, append the results, and loop.
4. Stop when the model returns a final text answer or after **8 iterations** (hard cap).
5. Log every step (messages, tool name, inputs, outputs, tokens, latency) to `traces/YYYY-MM-DD.jsonl`.

Make the model ID a config value so you can compare models in evals.

*Built:* `agent/agent.py`, defaulting to `claude-opus-5-5` at medium effort (`agent.cli --model` overrides it), with a 3-minute timeout on top of the 8-call cap and the server-side refusal fallback turned on. It runs on SDK 0.72, the newest for Python 3.8, so `output_config` and `fallbacks` go through `extra_body`. Verified live on `claude-haiku-5-5`. Haiku's answers used only tool numbers, but it still called roll windows "rounds" even when the prompt said starts. That's worth an eval case. Opus 5.5 got the same question right: it said "starts", used all 8 drivers, and flagged that the model counts Scheffler's better putting and approach numbers against him.

---

## 6. System prompt rules (`prompts.py`)

- You are an analyst for a PGA ranking model. Answer only from tool results.
- Every number you state must come from a tool output in this conversation.
- If a player name is ambiguous, ask which one. Don't guess.
- If data is missing (e.g. no recent events), say so plainly.
- Lineups are proposals. Never describe a lineup as final; the user must approve it.
- Explanations should name the top 2–3 drivers with their contribution values.
- Be concise.

Keep the prompt short. Put detail into tool descriptions instead.

---

## 7. Guardrails (`guardrails.py`)

1. **Player name resolution.** Every player mention goes through `resolve_player`. Ambiguous → ask the user.
2. **Numeric-claim checker (deterministic).** After the final answer, extract every number with a regex and check it appears in at least one tool output from that run, allowing rounding. Flag unsupported numbers in the CLI and count them in evals. This is a great interview talking point: a cheap, non-LLM check for hallucination.
3. **Constraint validation.** After `propose_lineups`, verify in plain Python that every lineup respects the salary cap, locks, excludes, and roster size. Never trust the LLM to check this.
4. **Iteration cap + timeout** on the agent loop.

*Built* in `agent/guardrails.py`:
- **Name resolution** needed no new code. It's `resolve_player` plus the prompt rule to ask when a name is ambiguous.
- **`check_numbers`** runs on every model-written answer in `run()`. It treats the user's words, tool inputs and tool results from the whole conversation as sources, but never the model's own earlier text. A number passes if some source value rounds to it at the precision written, ignoring sign. It understands `$1.65M`, `$408K` and shares written as percentages. It skips digits inside identifiers (`R2026060`, `roll16`, proposal ids) and list markers. Results go in `AgentResult.unsupported_numbers` and the trace's `end` event, and the CLI prints them under the answer.
  - Replaying 10 real answers flagged only numbers the model computed itself, such as "net to about −0.06".
  - **Known weakness:** in data-heavy runs, 20–39% of arbitrary 2-decimal numbers pass by coincidence because some unrelated value rounds to them; Opus's "−0.73" matched a 0.7345 score. With 3 decimals it's 2–6%. It's reliable against made-up precise numbers and weak against loosely rounded sums. Evals should report this rate, not treat it as proof.
  - The hard-coded "about 0.28" came out of the `model_performance` description, since no tool output backed it.
- **`validate_proposal`** re-checks a `propose_lineups` result against the request and the `salaries` table: roster size, duplicates, salaries known for the event, cap, reported totals and per-player salaries, locks, excludes, distinct lineups, and lineup count. The CLI runs it before the approval prompt; a failing proposal is shown with its problems, never offered, and logged as not approved. Evals can call it directly for `lineup_valid` cases.

---

## 8. Human in the loop (`cli.py`)

- `propose_lineups` returns a `proposal_id`.
- The CLI prints the lineups and asks `Approve? [y/N]`.
- Only on `y` does the CLI (not the agent) save them to `lineups/approved/`.
- The agent never has a "save" tool. Control stays with the human by design.

*Built:* the CLI shows each proposal from the tool's output (not the agent's summary), so you approve the actual optimizer result. A yes writes `lineups_approved_{proposal_id}.csv` into that event's `data/processed/{tid}_{slug}/` folder, next to the optimizer's own lineup files, instead of a separate `lineups/approved/`, following the CLAUDE.md convention that each week's files live together. Anything but y/yes saves nothing. Each decision is logged to the trace as an `approval` event.

---

## 9. Evals

### `evals/questions.jsonl` format

```json
{"id": "rank-01", "category": "lookup", "question": "Who is ranked #1 for <event>?", "expected": {"type": "exact", "value": "<player>"}}
{"id": "feat-03", "category": "lookup", "question": "What is <player>'s 16-event strokes gained before <event>?", "expected": {"type": "number", "value": 1.42, "tol": 0.01}}
{"id": "expl-02", "category": "explanation", "question": "Why is A ranked above B for <event>?", "expected": {"type": "drivers", "must_mention": ["sg_8", "course_hist"]}}
{"id": "lineup-04", "category": "lineup", "question": "Build 2 lineups locking X, excluding Y", "expected": {"type": "lineup_valid", "locks": ["X"], "excludes": ["Y"]}}
{"id": "guard-01", "category": "guardrail", "question": "How will <made-up player> do this week?", "expected": {"type": "refusal_or_clarify"}}
{"id": "amb-01", "category": "guardrail", "question": "How is Smith doing?", "expected": {"type": "clarify"}}
```

**Generate expected answers from the database with a script,** not by hand. Then the evals stay correct if data changes.

### Mix (~30 cases)

| Category | Count | Graded by |
|---|---|---|
| Lookups (ranks, features, salaries) | 8 | exact / numeric tolerance |
| Explanations (A vs B, why ranked X) | 6 | required drivers mentioned (string match), optional LLM judge |
| Value questions | 4 | top-N set overlap |
| Lineups with constraints | 6 | Python constraint validator |
| Guardrails (fake player, ambiguous name, missing data) | 6 | refused or asked to clarify |

### Metrics in every report

- Accuracy overall and by category
- Unsupported-number rate (from the numeric-claim checker)
- Tool-call correctness (did it call the expected tool?)
- Avg iterations, latency, and cost per question

### Process

1. Run the baseline and save the report.
2. Read the failures in the traces.
3. Fix **one thing at a time** (tool description, prompt rule, tool output shape).
4. Re-run and record the before/after for each change in `evals/CHANGELOG.md`.

That changelog is your interview story and your README "results" section.

### *Built*

- **Files.** `evals/generate.py` writes `questions.jsonl` (30 cases in the mix above, plus the 14 hard cases below) by running the tools against the current DB, and refuses to write a case whose premise no longer holds, for example an "infeasible" lineup that now fits. `evals/grade.py` holds deterministic graders, with no LLM judge. `evals/run_evals.py` writes a JSON and a Markdown report per run, and each case's trace is tagged `eval-{stamp}-{id}`.
- **How each type is graded.**
  - Names are matched ignoring accents and case.
  - Numbers must all appear within tolerance (`$196K` counts).
  - `drivers` passes if each feature is named, or described in one sentence by its family and window ("putting over the last 16 starts").
  - `lineup_valid` runs `validate_proposal` against what the *user* asked for, so a lineup the agent built without the requested lock fails.
  - `clarify`, `refusal_or_clarify` and `unavailable` check the tool trace (ambiguous result, no player data used, the expected error code), plus keyword cues in the answer.
- **Explanation cases** use only the TOUR Championship and The Open, the live weeks whose saved models still reproduce their scores. Explaining a base week (guard-04) is the missing-data case.
- **First run: Haiku 5.5 passed 30/30** (`evals/results/20261028-210912_claude-haiku-5-5.md`) at $0.0013 a question, with 2.6 model calls and 4.1s on average.
  - Accuracy has no headroom: the set works as a regression suite, but it can't show improvement yet.
  - The one metric with headroom is unsupported numbers, at 7% of answers: a sum the model worked out itself ("$898,000 over") and a threshold it picked ("outside the top 20").
- **14 hard cases added (44 total), tagged `hard`, with their own line in the report:**
  - 3 multi-turn (`follow_ups`, plus `expected_first` to require the "which Kim?" question before the answer);
  - 2 wrong premises ("Why is Scheffler #1?", "Why did Tom Kim win the Wyndham?");
  - 2 rounds-vs-starts wording traps;
  - 1 counterintuitive explanation (better putting that counts against Scheffler), which must admit the tools don't say why;
  - 2 no-arithmetic answers;
  - 1 out-of-scope question (prize money);
  - 3 filter or compare questions no single tool answers.
  - New check types support them: `text` (required words, forbidden patterns), `no_unsupported`, and `all` for combinations.
  - Haiku 5.5 now scores **42/44 (95%), hard 12/14** (`evals/results/20261028-214131_claude-haiku-5-5.md`). Both failures are genuine. In hard-10 it summed the drivers ("−0.73"). In hard-09 it subtracted two ranking scores ("about 0.34") instead of reporting `explain_prediction`'s exact gap.
  - **Single runs are noisy:** expl-02 flipped between two identical runs. Compare changes over repeated runs, or at least count the flips, before calling one an improvement.
- **Lessons from the first hard run:**
  - A failure can be the grader: "strokes-gained over the last 24 starts" means the total. Plain "strokes gained" with no other family named now counts as total.
  - A failure can also be the case: "both Kims" made the agent ask which ones, which our own prompt tells it to do. The question now says "both Kims in this week's field".
  - Read the failing answers before changing the agent.
- **Checker fixes the eval turned up.** It now reads comma-formatted numbers inside tool messages, which previously made "$1,898,000" look unsupported. It also treats K/M as multipliers only after `$`, which previously read "3M Open" as 3 million.

---

## 10. Milestones

### Weekend 1: working agent
- [x] Step 0 inventory + minimal refactor so each piece is callable as a function
- [x] `db.py` with consolidated tables + backfilled `predictions` (SQLite; adds an `events` table; predictions = 2018–22 out-of-fold + 7 live 2026 weeks)
- [x] Tools 1–4 (`resolve_player`, `get_rankings`, `get_player_features`, `explain_prediction`) with schemas
- [x] Hand-written agent loop with trace logging
- [x] CLI answering lookup + explanation questions end to end (`agent/cli.py`: a multi-turn conversation, so a clarifying question gets answered in context, plus a one-shot mode; `run()` takes `history`)
- [x] Unit tests for each tool (pytest): `tests/test_tools.py`, plus the loop in `tests/test_agent.py` and the CLI in `tests/test_cli.py`.

### Weekend 2: make it trustworthy
- [x] Tools 5–7 (`value_report`, `propose_lineups`, `model_performance`)
- [x] Lineup approval flow in CLI
- [x] Guardrails: name resolution, numeric-claim checker, constraint validator
- [x] Eval set (~30) + expected-answer generator + `run_evals.py`
- [ ] Baseline eval → 2–3 improvement iterations → changelog. *Baseline done* (`evals/CHANGELOG.md`): Haiku 42/44 and Opus 43/44 after the measurement fixes. Iteration 1 (a no-arithmetic prompt rule): Haiku accuracy 95.5% → 97.7%, hard cases 85.7% → 92.9%, unsupported numbers 6.8% → 0.8% (means of 3 runs). `run_evals --rescore` re-grades a saved run from its traces at no API cost, so grader changes are measured on the same answers.
- [ ] README: architecture diagram, how to run, eval results, lessons learned

### Stretch (only if time)
- Compare two models on the same eval set (accuracy vs. cost)
- Read-only `run_sql` tool restricted to `SELECT` with a row limit, then measure whether it helps or hurts accuracy
- LLM-as-judge for explanation quality, checked against your own grades on 10 cases

---

## 11. README outline

1. What it does (one paragraph + example Q&A)
2. Architecture diagram (tools → agent loop → guardrails → CLI)
3. Design decisions: why specific tools over raw SQL, why the human approves lineups, why a deterministic numeric checker
4. Eval results table (baseline vs. final, by category)
5. What broke and how it was fixed (from the changelog)
6. How to run

---

## 12. Resume bullet (fill in real numbers at the end)

> Built an LLM agent over the PGA model using tool calling (model inference, SHAP explanations, PuLP lineup optimizer) with human approval of lineups and a deterministic hallucination check; created a 30-question eval suite and improved accuracy from **X% to Y%** while cutting unsupported numeric claims to **Z%**

## 13. Interview talking points to capture as you build

- One failure where the agent picked the wrong tool, and how a tool description fixed it
- Why the numeric-claim checker is code, not another LLM call
- Where you deliberately kept the human in control, and why
- What you'd change to run this in production for a real client (auth, monitoring, cost limits, eval in CI)

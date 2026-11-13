"""
Consolidated read-only database for the PGA analyst agent.

Joins every local data source into one SQLite file (data/processed/agent.sqlite):

  players      one row per golfer, reconciling the two player-ID spaces
  events       one row per tournament edition (2015-2022 training + scored live weeks)
  features     model inputs per player-event
  predictions  model scores per player-event (out-of-fold backtest + live weekly;
               live weeks already in the feature matrix get both)
  results      finish / made cut / prize share per player-event
  salaries     Buzz salary per player-event

The training CSV and the PGA Tour GraphQL API use different player IDs for the
same golfer (Scheffler is 9478 in one, 46046 in the other), and different
tournament IDs. build_players() links players by name; every other table is
keyed by the canonical players.player_id — the API id where we know it (that's
what live data uses), else "H" + the training id, so the two spaces can't
collide. Event ids need no mapping: training ids are numeric ("2254"), live
ids are "R2026060".

Building is an explicit step; the agent only ever opens the file via connect(),
which is read-only:

    python -m agent.db --fetch    # cache live leaderboards + schedule from the API (network)
    python -m agent.db            # rebuild agent.sqlite from local files only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.buzz import load_salaries_csv
from src.crosswalk import build_players, match_key  # noqa: F401  (match_key re-exported)
from src.ingest import pos_to_prize_pct
from src.model import VARIANT_FEATURES, oof_predictions
from src.optimizer import match_salaries
from src.paths import AGENT_DB, FEATURE_MATRIX, PLAYER_REVIEW, PROCESSED_DIR, RESULTS_DIR
from src.scrape import get_completed_schedule, get_tournament_results

# Every column any model variant reads, in a stable order.
FEATURES: List[str] = list(dict.fromkeys(c for cols in VARIANT_FEATURES.values() for c in cols))

SCHEDULE_CSV = RESULTS_DIR / "schedule.csv"


# ── Source loading ─────────────────────────────────────────────────────────────

def _event_dirs() -> List[Path]:
    """Live tournament folders that have been scored (data/processed/{tid}_{slug}/)."""
    return sorted(p for p in PROCESSED_DIR.glob("R*_*") if (p / "weekly_features.parquet").exists())


def _variant_of(proj_path: Path) -> str:
    """'projections.csv' → 'base', 'projections_nocut.csv' → 'nocut'."""
    return proj_path.stem.replace("projections", "").lstrip("_") or "base"


def _load_results_cache(tid: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{tid}.csv"
    return pd.read_csv(path, dtype={"player_id": str}) if path.exists() else pd.DataFrame()


def _parse_usd(val) -> float:
    s = str(val).replace("$", "").replace(",", "").strip()
    return float(s) if s and s.lower() != "nan" else float("nan")


def fetch(verbose: bool = True) -> None:
    """
    Cache the season schedule and the final leaderboard of every scored live
    event into data/raw/results/. Network; leaderboards already cached are
    skipped (completed events don't change), unfinished events are skipped.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tids = [d.name.split("_")[0] for d in _event_dirs()]
    years = sorted({int(t[1:5]) for t in tids})
    schedule = pd.DataFrame([r for y in years for r in get_completed_schedule(y)])
    schedule.to_csv(SCHEDULE_CSV, index=False)
    completed = set(schedule["tournament_id"])

    for tid in tids:
        out = RESULTS_DIR / f"{tid}.csv"
        if tid not in completed:
            status = "not completed yet, skip"
        elif out.exists():
            status = "cached"
        else:
            df = get_tournament_results(tid, int(tid[1:5]))
            df.to_csv(out, index=False)
            status = f"{len(df)} players → {out.name}"
        if verbose:
            print(f"  {tid}: {status}")


# ── Build ──────────────────────────────────────────────────────────────────────

def build(db_path: Path = AGENT_DB, verbose: bool = True) -> Dict[str, int]:
    """
    Rebuild the agent database from local files (no network). Writes to a temp
    file and swaps it in, so a failed build never leaves a half-written DB.
    Returns {table: row_count}.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    fm = pd.read_parquet(FEATURE_MATRIX)
    # The agent's events are the historical CSV plus the scored live weeks.
    # Backfilled (scraped) rows are kept aside only to score those weeks out of
    # fold below; historical rows go back to their original ids for the crosswalk.
    scraped = fm.iloc[0:0]
    if "hist_player_id" in fm.columns:
        scraped = fm[fm["source"] == "scraped"]
        fm = fm[fm["source"] == "historical"].copy()
        fm["player_id"] = fm["hist_player_id"]
    fm["player_id"] = fm["player_id"].astype(str)
    dirs = _event_dirs()
    weekly = {d.name.split("_")[0]: pd.read_parquet(d / "weekly_features.parquet") for d in dirs}
    projections = {
        (d.name.split("_")[0], _variant_of(p)): pd.read_csv(p, dtype={"player_id": str})
        for d in dirs for p in sorted(d.glob("projections*.csv"))
    }
    results = {tid: _load_results_cache(tid) for tid in weekly}
    missing_results = [tid for tid, r in results.items() if r.empty]
    if missing_results:
        log(f"No cached results for {missing_results} — run `python -m agent.db --fetch`.")

    # ── players ────────────────────────────────────────────────────────────────
    api_sources = list(weekly.values()) + list(projections.values()) + [r for r in results.values() if not r.empty]
    api = pd.concat([df[["player_id", "player_name"]] for df in api_sources])
    players, review = build_players(fm[["player_id", "player_name"]], api)
    review.to_csv(PLAYER_REVIEW, index=False)
    linked = players["match_type"].notna()
    log(f"players: {len(players)} ({linked.sum()} linked across ID spaces: "
        f"{players['match_type'].value_counts().to_dict()}); review list → {PLAYER_REVIEW}")

    hist_to_pid = dict(zip(players["hist_id"].dropna(), players.loc[players["hist_id"].notna(), "player_id"]))
    fm["player_id"] = fm["player_id"].map(hist_to_pid)

    # ── events ─────────────────────────────────────────────────────────────────
    ev_train = (fm.groupby("tournament_id")
                  .agg(tournament_name=("tournament_name", "first"), course=("course", "first"),
                       season=("season", "first"), date=("date", "first"))
                  .reset_index().rename(columns={"tournament_id": "event_id"}))
    ev_train["source"] = "training"
    sched = pd.read_csv(SCHEDULE_CSV) if SCHEDULE_CSV.exists() else pd.DataFrame(columns=["tournament_id", "start_date"])
    start = dict(zip(sched["tournament_id"], sched["start_date"]))
    ev_live = pd.DataFrame([{
        "event_id": tid,
        "tournament_name": w["tournament_name"].iloc[0],
        "course": w["course"].iloc[0],
        "season": int(tid[1:5]),
        "date": start.get(tid),
        "source": "weekly",
    } for tid, w in weekly.items()])
    events = pd.concat([ev_train, ev_live], ignore_index=True)
    events["date"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")

    # ── features ───────────────────────────────────────────────────────────────
    f_train = fm.rename(columns={"tournament_id": "event_id"})
    f_live = pd.concat([w.assign(event_id=tid) for tid, w in weekly.items()], ignore_index=True)
    features = pd.concat([f_train, f_live], ignore_index=True)
    features = features.reindex(columns=["player_id", "event_id", "owgr"] + FEATURES)

    # ── predictions ────────────────────────────────────────────────────────────
    log("predictions: computing out-of-fold backtest scores (trains one model per season) ...")
    with contextlib.redirect_stdout(io.StringIO()):
        oof = oof_predictions(pd.concat([fm, scraped], ignore_index=True))
    oof = oof.rename(columns={"tournament_id": "event_id"}).assign(model_version="base_oof")
    # Training events, plus each live week's field scored by a model that never
    # saw that season. The stored weekly projections can't be graded honestly
    # once the model has been retrained on a matrix that includes those weeks.
    in_field = oof[["player_id", "event_id"]].merge(
        f_live[["player_id", "event_id"]], how="left", indicator=True)["_merge"].eq("both").values
    p_train = oof[oof["event_id"].isin(ev_train["event_id"]) | in_field].copy()
    p_train["rank"] = (p_train.groupby("event_id")["score"]      # within the stored field
                       .rank(ascending=False, method="min").astype(int))
    p_live = pd.concat([
        p.assign(event_id=tid, model_version=variant, score=p["projected_prize_pct"])
        for (tid, variant), p in projections.items()
    ], ignore_index=True)
    predictions = pd.concat([p_train, p_live], ignore_index=True)[
        ["player_id", "event_id", "model_version", "score", "rank"]
    ]

    # ── results ────────────────────────────────────────────────────────────────
    # fm also has a raw "finish" string column ("T5"); finish_num is the numeric one.
    r_train = pd.DataFrame({
        "player_id": fm["player_id"],
        "event_id": fm["tournament_id"],
        "finish": fm["finish_num"],
        "made_cut": fm["made_cut"],
        "prize_pct": fm["prize_pct"],
    })
    r_live = []
    for tid, r in results.items():
        if r.empty:
            continue
        r_live.append(pd.DataFrame({
            "player_id": r["player_id"],
            "event_id": tid,
            "finish": r["finish_num"],
            "made_cut": r["made_cut"].astype(bool),
            "prize_pct": [pos_to_prize_pct(p, mc) for p, mc in zip(r["finish_num"], r["made_cut"])],
            "winnings_usd": r["winnings"].map(_parse_usd),
        }))
    results_df = pd.concat([r_train] + r_live, ignore_index=True)
    results_df["made_cut"] = results_df["made_cut"].astype(int)

    # ── salaries ───────────────────────────────────────────────────────────────
    # Same projection↔salary join the optimizer uses, so tools and lineups agree.
    s_rows = []
    for d in dirs:
        tid = d.name.split("_")[0]
        sal_path = d / "salaries.csv"
        proj = projections.get((tid, "base"))
        if proj is None:
            proj = next((p for (t, _), p in projections.items() if t == tid), None)
        if not sal_path.exists() or proj is None:
            continue
        sal = load_salaries_csv(sal_path)
        with contextlib.redirect_stdout(io.StringIO()):
            merged = match_salaries(proj, sal)
        log(f"salaries {tid}: {len(merged)}/{len(proj)} projected players matched "
            f"({len(sal)} salary entries)")
        s_rows.append(merged.assign(event_id=tid)[["player_id", "event_id", "salary"]])
    salaries = pd.concat(s_rows, ignore_index=True) if s_rows else pd.DataFrame(columns=["player_id", "event_id", "salary"])

    # ── write ──────────────────────────────────────────────────────────────────
    tables = {
        "players": players, "events": events, "features": features,
        "predictions": predictions, "results": results_df, "salaries": salaries,
    }
    keys = {
        "players": ["player_id"], "events": ["event_id"], "features": ["player_id", "event_id"],
        "predictions": ["player_id", "event_id", "model_version"],
        "results": ["player_id", "event_id"], "salaries": ["player_id", "event_id"],
    }
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        for name, df in tables.items():
            df.to_sql(name, con, index=False)
            # Unique indexes double as integrity checks: a duplicate key fails the build.
            con.execute(f"CREATE UNIQUE INDEX ix_{name} ON {name} ({', '.join(keys[name])})")
        con.commit()
    finally:
        con.close()
    tmp.replace(db_path)

    counts = {name: len(df) for name, df in tables.items()}
    log(f"\nSaved {db_path}: {counts}")
    return counts


# ── Read-only access ───────────────────────────────────────────────────────────

def connect(db_path: Path = AGENT_DB) -> sqlite3.Connection:
    """Open the agent database read-only; any write raises sqlite3.OperationalError."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run `python -m agent.db` to build it.")
    con = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the PGA analyst agent database")
    parser.add_argument("--fetch", action="store_true",
                        help="Cache live leaderboards + schedule from the PGA API, then build")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    build()

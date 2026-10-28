"""
Consolidated read-only database for the PGA analyst agent.

Joins every local data source into one SQLite file (data/processed/agent.sqlite):

  players      one row per golfer, reconciling the two player-ID spaces
  events       one row per tournament edition (2015-2022 training + scored live weeks)
  features     model inputs per player-event
  predictions  model scores per player-event (out-of-fold backtest + live weekly)
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
import difflib
import io
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.buzz import load_salaries_csv, normalize_name
from src.ingest import pos_to_prize_pct
from src.model import VARIANT_FEATURES, oof_predictions
from src.optimizer import match_salaries
from src.paths import AGENT_DB, FEATURE_MATRIX, PLAYER_REVIEW, PROCESSED_DIR, RESULTS_DIR
from src.scrape import get_completed_schedule, get_tournament_results

# Every column any model variant reads, in a stable order.
FEATURES: List[str] = list(dict.fromkeys(c for cols in VARIANT_FEATURES.values() for c in cols))

SCHEDULE_CSV = RESULTS_DIR / "schedule.csv"


# ── Player crosswalk ───────────────────────────────────────────────────────────

# Letters NFD accent-stripping leaves intact (they're distinct letters, not
# accented ones), so "Højgaard" would otherwise never meet "Hojgaard".
_TRANSLIT = str.maketrans({"ø": "o", "æ": "ae", "ß": "ss", "ł": "l", "đ": "d"})


def match_key(name: str) -> str:
    """normalize_name, then fold ø/æ/etc. and drop punctuation: 'Hao-Tong Li' → 'hao tong li'."""
    key = normalize_name(name).translate(_TRANSLIT).replace("-", " ")
    key = re.sub(r"[^a-z ]", "", key)
    return re.sub(r"\s+", " ", key).strip()


def display_name(name: str) -> str:
    """'Scheffler, Scottie' → 'Scottie Scheffler'; already-ordered names pass through."""
    name = str(name).strip()
    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name


# Hand-verified links the rules can't make, api_id → hist_id. Add one only after
# confirming it's the same golfer (it shows up as unlinked_near_match in review).
MANUAL_LINKS: Dict[str, str] = {
    "48867": "4566443",   # Matti Schmid = Matthias Schmid (nickname, not a prefix)
}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per distinct (player_id, spelling), most frequent spelling first.
    An id can carry several spellings — the API itself calls 35658 both
    "Kris Ventura" and "Kristoffer Ventura" — and every one is matched on.
    """
    out = df[["player_id", "player_name"]].dropna().copy()
    out["player_id"] = out["player_id"].astype(str)
    out["player_name"] = out["player_name"].map(display_name)
    out = (out.groupby(["player_id", "player_name"], sort=False).size().rename("n").reset_index()
              .sort_values(["player_id", "n"], ascending=[True, False], kind="stable")
              .reset_index(drop=True))
    out["key"] = out["player_name"].map(match_key)
    out["compact"] = out["key"].str.replace(" ", "", regex=False)
    parts = out["key"].str.split()
    out["first"] = parts.str[0].fillna("")
    out["last"] = parts.map(lambda p: p[-1] if len(p) > 1 else "")
    return out


def build_players(
    hist: pd.DataFrame,
    api: pd.DataFrame,
    manual: Dict[str, str] = MANUAL_LINKS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Link training-data players (hist) to PGA API players (api), both given as
    [player_id, player_name]. Returns (players, review).

    manual (api_id → hist_id) links are applied first, when both ids are
    present. The rest are made only by deterministic rules, and only when the
    match is unique on both sides:
      1. exact:             same match_key ignoring spaces ("Haotong Li" = "Hao-Tong Li")
      2. first_name_prefix: same last name, and one first name starts the other
                            ("Matt Fitzpatrick" = "Matthew Fitzpatrick")
    Anything else stays unlinked. Near-misses go to review rather than being
    guessed: fuzzy similarity can't separate "Carson Young" → "Cameron Young"
    (0.88, wrong) from "Zach Bauchou" → "Zachary Bauchou" (0.89, right).
    """
    h, a = _prep(hist), _prep(api)
    a_first = a.drop_duplicates("player_id").set_index("player_id")
    h_first = h.drop_duplicates("player_id").set_index("player_id")
    links: List[Tuple[str, str, str]] = []   # (api_id, hist_id, match_type)
    review: List[dict] = []

    def _review(aid: str, hid: str, match_type: str) -> None:
        ar, hr = a_first.loc[aid], h_first.loc[hid]
        review.append({
            "api_id": aid, "api_name": ar["player_name"],
            "hist_id": hid, "hist_name": hr["player_name"],
            "match_type": match_type,
            "similarity": round(difflib.SequenceMatcher(None, ar["key"], hr["key"]).ratio(), 2),
        })

    def _link_unique(pairs: pd.DataFrame, match_type: str, ambiguous_type: str) -> None:
        """Link (api_id, hist_id) pairs that are one-to-one; send the rest to review."""
        pairs = pairs.drop_duplicates()
        if pairs.empty:
            return
        one_to_one = (
            (pairs.groupby("api_id")["hist_id"].transform("nunique") == 1)
            & (pairs.groupby("hist_id")["api_id"].transform("nunique") == 1)
        )
        for aid, hid in pairs[one_to_one].itertuples(index=False):
            links.append((aid, hid, match_type))
            if match_type != "exact":
                _review(aid, hid, match_type)
        for aid, hid in pairs[~one_to_one].itertuples(index=False):
            _review(aid, hid, ambiguous_type)

    # 0. Hand-verified links.
    for aid, hid in manual.items():
        if aid in a_first.index and hid in h_first.index:
            links.append((aid, hid, "manual"))

    # 1. Exact key match on any spelling, among players not linked by hand.
    exact = a[~a["player_id"].isin({l[0] for l in links})].merge(
        h[~h["player_id"].isin({l[1] for l in links})], on="compact", suffixes=("_a", "_h"))
    _link_unique(exact[["player_id_a", "player_id_h"]].set_axis(["api_id", "hist_id"], axis=1),
                 "exact", "ambiguous_name")

    # 2. First-name prefix on the same last name, among players stage 1 neither
    #    linked nor already flagged as ambiguous.
    done_a = {l[0] for l in links} | {r["api_id"] for r in review}
    ua = a[~a["player_id"].isin(done_a) & (a["last"] != "")]
    uh = h[~h["player_id"].isin({l[1] for l in links}) & (h["last"] != "")]
    same_last = ua.merge(uh, on="last", suffixes=("_a", "_h"))
    is_prefix = [fh.startswith(fa) or fa.startswith(fh)
                 for fa, fh in zip(same_last["first_a"], same_last["first_h"])]
    prefix = same_last[is_prefix] if len(same_last) else same_last
    _link_unique(prefix[["player_id_a", "player_id_h"]].set_axis(["api_id", "hist_id"], axis=1),
                 "first_name_prefix", "ambiguous_prefix")

    # 3. Near-misses for review only — never linked.
    linked_a = {l[0] for l in links}
    linked_h = {l[1] for l in links}
    reviewed_a = {r["api_id"] for r in review}
    uh = h[~h["player_id"].isin(linked_h)]
    uh_keys = uh["key"].unique().tolist()
    near = set()
    for _, ar in a[~a["player_id"].isin(linked_a | reviewed_a)].iterrows():
        for close in difflib.get_close_matches(ar["key"], uh_keys, n=3, cutoff=0.8):
            near.update((ar["player_id"], hid) for hid in uh.loc[uh["key"] == close, "player_id"])
    for aid, hid in sorted(near):
        _review(aid, hid, "unlinked_near_match")

    # Assemble one row per golfer; names are API spellings first (most frequent first).
    a_names = a.groupby("player_id", sort=False)["player_name"].agg(list).to_dict()
    h_names = h.groupby("player_id", sort=False)["player_name"].agg(list).to_dict()
    rows = []
    for aid, hid, match_type in links:
        rows.append({"player_id": aid, "api_id": aid, "hist_id": hid, "match_type": match_type,
                     "names": a_names[aid] + h_names[hid]})
    for aid, names in a_names.items():
        if aid not in linked_a:
            rows.append({"player_id": aid, "api_id": aid, "hist_id": None, "match_type": None,
                         "names": names})
    for hid, names in h_names.items():
        if hid not in linked_h:
            rows.append({"player_id": f"H{hid}", "api_id": None, "hist_id": hid, "match_type": None,
                         "names": names})

    players = pd.DataFrame(rows)
    names = players.pop("names").map(lambda ns: list(dict.fromkeys(ns)))
    players.insert(1, "name", names.str[0])
    players.insert(2, "name_key", players["name"].map(match_key))
    players.insert(3, "aliases", names.map(lambda ns: "|".join(ns[1:]) or None))
    players = players.sort_values("name").reset_index(drop=True)

    review_cols = ["api_id", "api_name", "hist_id", "hist_name", "match_type", "similarity"]
    review_df = pd.DataFrame(review, columns=review_cols).sort_values(["match_type", "api_name"])
    return players, review_df.reset_index(drop=True)


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
        oof = oof_predictions(fm)
    p_train = oof.rename(columns={"tournament_id": "event_id"}).assign(model_version="base_oof")
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

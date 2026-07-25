"""
Lineup optimizer for Buzz Fantasy Golf.

Rules (as described):
  - Pick 6 players
  - Total salary ≤ weekly budget (e.g. $1,650,000 for US Open)
  - Scoring = prize money earned at the tournament

Optimization:
  Maximize the sum of projected scores for 6 players
  subject to the salary cap.

Usage:
  # With session cookie (auto-fetches salary from Buzz)
  python -m src.optimizer --cookie "_buzz_session=..." --budget 1650000

  # With manual salary CSV
  python -m src.optimizer --csv data/raw/salaries.csv --budget 1650000

  # With locks/excludes
  python -m src.optimizer --csv salaries.csv --budget 1650000 \\
      --lock "Scheffler" --lock "Morikawa" --exclude "Smalley"

  # Multiple lineup alternatives
  python -m src.optimizer --csv salaries.csv --budget 1650000 --lineups 5
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pulp import (
    LpMaximize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
    PULP_CBC_CMD,
)

from src.buzz import load_salaries, normalize_name
from src.paths import PROCESSED_DIR

DEFAULT_BUDGET   = 1_650_000
N_PLAYERS        = 6


def _latest_projections() -> Path:
    """Return the most recently modified projections.csv across all tournament folders."""
    candidates = sorted(
        PROCESSED_DIR.glob("*/projections.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Legacy flat path fallback
    return PROCESSED_DIR / "projections.csv"


# ── Name matching ──────────────────────────────────────────────────────────────

def _match_names(
    proj_df: pd.DataFrame,
    sal_df: pd.DataFrame,
    cutoff: float = 0.82,
) -> pd.DataFrame:
    """
    Join projections to salaries on normalised player names.
    Uses fuzzy matching as a fallback for minor spelling differences.
    Returns the merged DataFrame and prints a match report.
    """
    proj_df = proj_df.copy()
    sal_df  = sal_df.copy()
    proj_df["name_key"] = proj_df["player_name"].apply(normalize_name)

    sal_keys = sal_df["name_key"].tolist()

    # Exact join first
    merged = proj_df.merge(sal_df[["name_key", "salary"]], on="name_key", how="left")

    # Fuzzy fallback for rows that didn't match
    unmatched_mask = merged["salary"].isna()
    for idx in merged[unmatched_mask].index:
        key = merged.at[idx, "name_key"]
        # 1. Standard fuzzy match
        close = difflib.get_close_matches(key, sal_keys, n=1, cutoff=cutoff)
        if close:
            matched_sal = sal_df.loc[sal_df["name_key"] == close[0], "salary"].iloc[0]
            merged.at[idx, "salary"] = matched_sal
            continue
        # 2. Prefix match: "angel hidalgo" matches "angel hidalgo portillo"
        for sal_key in sal_keys:
            if sal_key.startswith(key) or key.startswith(sal_key):
                matched_sal = sal_df.loc[sal_df["name_key"] == sal_key, "salary"].iloc[0]
                merged.at[idx, "salary"] = matched_sal
                break

    matched   = merged["salary"].notna().sum()
    unmatched = merged["salary"].isna().sum()
    print(f"Name matching: {matched} matched, {unmatched} unmatched")
    if unmatched > 0:
        miss = merged[merged["salary"].isna()]["player_name"].tolist()
        print(f"  Unmatched projection players: {miss[:10]}")

    return merged.dropna(subset=["salary"])


# ── Core optimizer ─────────────────────────────────────────────────────────────

def _solve_one(
    df: pd.DataFrame,
    budget: int,
    n_players: int,
    locks: List[str],
    excludes: List[str],
    prev_lineups: Optional[List[List[int]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Solve a single lineup optimisation.
    prev_lineups: list of index lists from already-generated lineups.
    Each prior lineup gets a constraint requiring at least 1 different player.
    """
    prob = LpProblem("buzz_golf_lineup", LpMaximize)
    indices = list(df.index)

    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in indices}

    # Objective: maximise projected performance
    prob += lpSum(df.at[i, "projected_prize_pct"] * x[i] for i in indices)

    # Budget constraint
    prob += lpSum(df.at[i, "salary"] * x[i] for i in indices) <= budget

    # Roster size
    prob += lpSum(x[i] for i in indices) == n_players

    # Locks (must include) — substring match so "Fleetwood" matches "tommy fleetwood"
    lock_keys = [normalize_name(n) for n in locks]
    for i in indices:
        nk = df.at[i, "name_key"]
        if any(lk in nk or nk in lk for lk in lock_keys):
            prob += x[i] == 1

    # Excludes (must not include) — same substring logic
    excl_keys = [normalize_name(n) for n in excludes]
    for i in indices:
        nk = df.at[i, "name_key"]
        if any(ek in nk or nk in ek for ek in excl_keys):
            prob += x[i] == 0

    # Diversity: must differ from every previous lineup by at least 1 player
    for prev in (prev_lineups or []):
        prob += lpSum(x[i] for i in prev) <= n_players - 1

    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status != 1:  # 1 = Optimal
        return None

    selected = [i for i in indices if value(x[i]) > 0.5]
    return df.loc[selected].copy()


# ── Multi-lineup ───────────────────────────────────────────────────────────────

def build_lineups(
    merged_df: pd.DataFrame,
    n_lineups: int = 1,
    budget: int = DEFAULT_BUDGET,
    n_players: int = N_PLAYERS,
    locks: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """
    Build n_lineups distinct optimal lineups.
    Each successive lineup must differ by at least 1 player from all prior ones.
    """
    locks   = locks   or []
    excludes = excludes or []
    lineups  = []
    prev_indices: List[List[int]] = []

    for k in range(n_lineups):
        lineup = _solve_one(merged_df, budget, n_players, locks, excludes, prev_indices)
        if lineup is None:
            print(f"Could not find lineup {k+1} (infeasible with current constraints).")
            break
        lineups.append(lineup)
        prev_indices.append(list(lineup.index))

    return lineups


# ── Display ────────────────────────────────────────────────────────────────────

def _fmt_salary(s: int) -> str:
    if s >= 1_000_000:
        return f"${s/1_000_000:.2f}M"
    if s >= 1_000:
        return f"${s/1_000:.0f}K"
    return f"${s:,}"


def display_lineups(
    lineups: List[pd.DataFrame],
    budget: int,
    merged_df: pd.DataFrame,
) -> None:
    """Print each lineup to stdout in a readable table."""
    budget_str = _fmt_salary(budget)

    for k, lineup in enumerate(lineups, 1):
        total_salary = int(lineup["salary"].sum())
        total_proj   = lineup["projected_prize_pct"].sum()
        remaining    = budget - total_salary

        header = f"Lineup {k}" if len(lineups) > 1 else "Optimal Lineup"
        print(f"\n{'='*64}")
        print(f"  {header}  |  Budget: {budget_str}  |  Remaining: {_fmt_salary(remaining)}")
        print(f"{'='*64}")
        print(f"  {'#':<4}{'Player':<28}{'OWGR':<8}{'Salary':<12}{'Proj. Prize %'}")
        print(f"  {'-'*60}")

        for rank, (_, row) in enumerate(
            lineup.sort_values("projected_prize_pct", ascending=False).iterrows(), 1
        ):
            owgr = f"#{int(row['owgr'])}" if pd.notna(row.get("owgr")) else "—"
            print(
                f"  {rank:<4}{row['player_name']:<28}{owgr:<8}"
                f"{_fmt_salary(int(row['salary'])):<12}{row['projected_prize_pct']*100:.2f}%"
            )

        print(f"  {'─'*60}")
        print(f"  {'TOTAL':<28}{'':8}{_fmt_salary(total_salary):<12}{total_proj*100:.2f}%")

    # Value board: prize % per $M of salary
    print("\n── Value Board (proj. prize % per $M salary) ──")
    merged_df = merged_df.copy()
    merged_df["value"] = (merged_df["projected_prize_pct"] / merged_df["salary"] * 1_000_000).round(4)
    top_value = merged_df.nlargest(15, "value")[["player_name", "salary", "projected_prize_pct", "value"]]
    print(f"  {'Player':<28}{'Salary':<12}{'Proj. Prize %':<16}{'%/$M'}")
    print(f"  {'-'*58}")
    for _, row in top_value.iterrows():
        print(
            f"  {row['player_name']:<28}{_fmt_salary(int(row['salary'])):<12}"
            f"{row['projected_prize_pct']*100:<16.2f}{row['value']:.4f}"
        )


def _lineups_filename(locks: List[str], excludes: List[str]) -> str:
    """
    Build a lineups filename that encodes the locks/excludes so distinct runs
    don't overwrite each other. A plain run (no locks/excludes) stays 'lineups.csv'.
    """
    def _slug(names: List[str]) -> str:
        return "_".join(normalize_name(n).replace(" ", "") for n in names if n.strip())

    parts = []
    if locks:
        parts.append("lock_" + _slug(locks))
    if excludes:
        parts.append("excl_" + _slug(excludes))
    if not parts:
        return "lineups.csv"
    return "lineups_" + "__".join(parts) + ".csv"


def export_csv(lineups: List[pd.DataFrame], out_path: Path = PROCESSED_DIR / "lineups.csv") -> None:
    """Save all lineups to a CSV for record-keeping."""
    rows = []
    for k, lineup in enumerate(lineups, 1):
        for _, row in lineup.iterrows():
            rows.append({"lineup": k, **row.to_dict()})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nLineups saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Buzz Fantasy Golf lineup optimizer")
    # Salary source
    parser.add_argument("--cookie",  default="", help="Buzz session cookie string")
    parser.add_argument("--csv",     default=None, help="Path to manual salary CSV (name,salary)")
    parser.add_argument("--league",  type=int, default=25120, help="Buzz league ID")
    # Pool rules
    parser.add_argument("--budget",  type=int, default=DEFAULT_BUDGET,
                        help=f"Salary cap (default {DEFAULT_BUDGET:,})")
    parser.add_argument("--players", type=int, default=N_PLAYERS,
                        help=f"Roster size (default {N_PLAYERS})")
    # Overrides
    parser.add_argument("--lock",    action="append", default=[],
                        help="Player name to force into every lineup (repeatable)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Player name to exclude (repeatable)")
    # Output
    parser.add_argument("--lineups", type=int, default=1,
                        help="Number of alternative lineups to generate (default 1)")
    parser.add_argument("--proj",    default=None,
                        help="Path to projections CSV (defaults to most recent tournament folder)")
    args = parser.parse_args()

    # ── Load projections ───────────────────────────────────────────────────────
    proj_path = Path(args.proj) if args.proj else _latest_projections()
    if not proj_path.exists():
        print(f"Projections not found at {proj_path}.")
        print("Run `python -m src.predict --tournament <id>` first.")
        return
    proj = pd.read_csv(proj_path)
    out_dir = proj_path.parent  # lineups go in the same tournament folder
    print(f"Loaded {len(proj)} projections from {proj_path}")

    # ── Load salaries ──────────────────────────────────────────────────────────
    sal = load_salaries(
        league_id=args.league,
        cookie=args.cookie,
        csv_path=args.csv,
    )
    print(f"Loaded {len(sal)} salary entries")
    print(f"Salary range: {_fmt_salary(int(sal['salary'].min()))} – {_fmt_salary(int(sal['salary'].max()))}")

    # ── Merge ──────────────────────────────────────────────────────────────────
    merged = _match_names(proj, sal)
    if len(merged) < args.players:
        print(f"Only {len(merged)} players matched — not enough to fill a lineup. "
              f"Check your salary CSV.")
        return

    # ── Optimise ──────────────────────────────────────────────────────────────
    lineups = build_lineups(
        merged_df=merged,
        n_lineups=args.lineups,
        budget=args.budget,
        n_players=args.players,
        locks=args.lock,
        excludes=args.exclude,
    )

    if not lineups:
        print("No feasible lineup found. Try increasing --budget or removing locks/excludes.")
        return

    display_lineups(lineups, args.budget, merged)
    export_csv(lineups, out_dir / _lineups_filename(args.lock, args.exclude))


if __name__ == "__main__":
    main()

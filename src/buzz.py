"""
Buzz Fantasy Golf salary fetcher.

Two modes:
  1. Cookie scraper (primary): uses your logged-in session cookie to hit the
     Buzz API and pull this week's player pool + costs automatically.
  2. CSV fallback: if you paste player data into a CSV manually.

─── Getting your session cookie (one-time setup) ─────────────────────────────
  1. Log into buzzfantasygolf.com in Chrome/Firefox
  2. Open DevTools (F12) → Network tab
  3. Navigate to your league page
  4. Click any request to buzzfantasygolf.com → Headers → Request Headers
  5. Copy the full value of the "Cookie" header
  6. Run: python -m src.optimizer --cookie "<paste here>" --league 25120

─── CSV fallback format ──────────────────────────────────────────────────────
  Create a file like data/raw/salaries.csv with two columns:
    name,salary
    Scottie Scheffler,9800000
    Xander Schauffele,8750000
    ...
  Salary can be in dollars (9800000) or shorthand (9.8M or 9800).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

LEAGUE_ID = 25120
_BASE = "https://buzzfantasygolf.com"

# Endpoints to probe (tried in order until one returns player data)
_CANDIDATE_PATHS = [
    "/api/leagues/{lid}/available_players",
    "/api/leagues/{lid}/players",
    "/api/v1/leagues/{lid}/players",
    "/leagues/{lid}/players.json",
    "/leagues/{lid}/available_players.json",
    "/api/leagues/{lid}/golfers",
]


# ── Name normalisation ─────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Lowercase, remove accents, strip punctuation, and convert "Last, First"
    → "first last" so both sources can be joined on the same key.
    """
    # Remove accents (Åberg → aberg, etc.)
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    # "scheffler, scottie" → "scottie scheffler"
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    # Remove Jr., Sr., II, III suffixes
    for suffix in (" jr.", " sr.", " ii", " iii", " iv"):
        name = name.replace(suffix, "")
    return name.strip()


# ── Cookie-based scraper ───────────────────────────────────────────────────────

def _session(cookie: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Cookie": cookie,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Referer": _BASE,
    })
    return s


def _try_endpoint(path: str, league_id: int, session: requests.Session) -> Optional[list]:
    """Try one API path and return a list of player dicts, or None on failure."""
    url = _BASE + path.format(lid=league_id)
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    # Handle both list and dict responses
    if isinstance(data, list):
        return data
    for key in ("players", "golfers", "available_players", "data"):
        if isinstance(data.get(key), list):
            return data[key]
    return None


def _parse_buzz_players(raw: list) -> pd.DataFrame:
    """
    Parse raw Buzz API player records into a standardised DataFrame.
    Buzz likely returns something like:
      {"name": "Scottie Scheffler", "salary": 9800000, "id": 123, ...}
    Column names are guessed from common patterns.
    """
    rows = []
    for p in raw:
        # Try common name field names
        name = (
            p.get("name") or p.get("player_name") or p.get("full_name")
            or p.get("display_name") or ""
        )
        # Try common salary field names
        salary = (
            p.get("salary") or p.get("cost") or p.get("price") or p.get("value") or 0
        )
        if name:
            rows.append({"name": name, "salary": int(salary),
                         "buzz_id": p.get("id") or p.get("player_id")})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["name_key"] = df["name"].apply(normalize_name)
    return df


def fetch_salaries_api(league_id: int = LEAGUE_ID, cookie: str = "") -> Optional[pd.DataFrame]:
    """
    Try each candidate endpoint with the provided session cookie.
    Returns a DataFrame with columns [name, salary, name_key] or None if all fail.
    """
    if not cookie:
        return None
    session = _session(cookie)
    for path in _CANDIDATE_PATHS:
        raw = _try_endpoint(path, league_id, session)
        if raw:
            df = _parse_buzz_players(raw)
            if not df.empty:
                print(f"Fetched {len(df)} players from {path.format(lid=league_id)}")
                return df
    print("Could not find a working Buzz API endpoint with the provided cookie.")
    print("Check DevTools → Network tab while on the league page to find the right URL.")
    return None


# ── CSV fallback ───────────────────────────────────────────────────────────────

def _parse_salary(val) -> int:
    """Parse salary values like 9800000, '9.8M', '9800K', '$700,000'."""
    s = str(val).replace("$", "").replace(",", "").strip()
    if s.upper().endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.upper().endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(float(s))


def _parse_buzz_paste(path: "str | Path") -> pd.DataFrame:
    """
    Parse the multi-line block format that Buzz Fantasy Golf produces when you
    copy-paste a player list.  Each player block looks like:

        Last, First
        $700,000
        (blank)
        <7 stat numbers>
        $12,043,693.00
        (blank)

    We only need the first two non-blank lines of each block: name and salary.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    i = 0
    while i < len(lines):
        name_line = lines[i].strip()
        # Skip blank lines between blocks
        if not name_line:
            i += 1
            continue
        # Next non-blank line should be the salary
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        sal_line = lines[j].strip()
        if sal_line.startswith("$") and "," in sal_line and "." not in sal_line:
            # Looks like a salary (e.g. "$700,000") not earnings ("$12,043,693.00")
            rows.append({"name": name_line, "salary": _parse_salary(sal_line)})
            i = j + 1
        else:
            i += 1

    df = pd.DataFrame(rows)
    df["name_key"] = df["name"].apply(normalize_name)
    return df[["name", "salary", "name_key"]].dropna()


def load_salaries_csv(path: "str | Path") -> pd.DataFrame:
    """
    Load salary data from either:
      - A standard two-column CSV (name, salary)
      - The Buzz copy-paste block format (auto-detected)
    Salary can be in dollars (9800000), shorthand (9.8M), or thousands (9800).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""

    # Detect Buzz block format: first line is a player name (no comma = header?)
    # Heuristic: if there's no header row with "name"/"salary" keywords, treat as block format
    if "salary" not in first_line.lower() and "name" not in first_line.lower():
        df = _parse_buzz_paste(path)
        if not df.empty:
            return df

    # Standard two-column CSV
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    name_col = next((c for c in df.columns if "name" in c), df.columns[0])
    sal_col  = next((c for c in df.columns if any(k in c for k in ("salary","cost","price","value"))), df.columns[1])
    df = df.rename(columns={name_col: "name", sal_col: "salary"})
    df["salary"]   = df["salary"].apply(_parse_salary)
    df["name_key"] = df["name"].apply(normalize_name)
    return df[["name", "salary", "name_key"]].dropna()


# ── Main entry point ───────────────────────────────────────────────────────────

def load_salaries(
    league_id: int = LEAGUE_ID,
    cookie: str = "",
    csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load salary data. Tries the API first if a cookie is provided,
    then falls back to the CSV.
    """
    if cookie:
        df = fetch_salaries_api(league_id, cookie)
        if df is not None and not df.empty:
            return df
        print("Falling back to CSV ...")

    if csv_path:
        return load_salaries_csv(csv_path)

    raise ValueError(
        "No salary data available. Provide --cookie or --csv <path>.\n"
        "CSV format: two columns — name,salary"
    )

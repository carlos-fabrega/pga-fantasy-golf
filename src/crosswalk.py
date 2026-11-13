"""
Player crosswalk between the two player-ID spaces.

The training CSV and the PGA Tour GraphQL API use different player IDs for the
same golfer (Scheffler is 9478 in one, 46046 in the other). build_players()
links them by name, by deterministic rules only. Used by ingest.combine() to
put historical and backfilled rows on one id, and by agent.db for its players
table.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Tuple

import pandas as pd

from src.buzz import normalize_name


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
    "55182": "4602673",   # Tom Kim = Joohyung Kim (legal name in the training data)
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

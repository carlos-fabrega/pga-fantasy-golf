"""
Tests for the agent database's player crosswalk and read-only access.

Run:  python -m pytest tests/test_db.py
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from agent.db import build_players, connect, match_key


def _players(hist_rows, api_rows):
    hist = pd.DataFrame(hist_rows, columns=["player_id", "player_name"])
    api = pd.DataFrame(api_rows, columns=["player_id", "player_name"])
    players, review = build_players(hist, api)
    return players.set_index("player_id"), review


def test_match_key_folds_letters_accents_and_punctuation():
    assert match_key("Olesen, Thorbjørn") == match_key("Thorbjorn Olesen")
    assert match_key("Åberg, Ludvig") == "ludvig aberg"
    assert match_key("Hao-Tong Li") == "hao tong li"
    assert match_key("Ewart, A.J.") == "aj ewart"


def test_exact_links_ignore_spacing_and_use_api_id():
    players, _ = _players(
        [("9478", "Scottie Scheffler"), ("5934", "Hao-Tong Li")],
        [("46046", "Scheffler, Scottie"), ("35296", "Li, Haotong")],
    )
    assert players.loc["46046", "hist_id"] == "9478"
    assert players.loc["46046", "match_type"] == "exact"
    assert players.loc["35296", "hist_id"] == "5934"
    assert players.loc["35296", "aliases"] == "Hao-Tong Li"


def test_first_name_prefix_links_only_with_same_last_name():
    players, review = _players(
        [("9037", "Matthew Fitzpatrick"), ("1", "Cameron Young"), ("2", "Jordan Spieth")],
        [("40098", "Fitzpatrick, Matt"), ("55721", "Fitzpatrick, Alex"),
         ("3", "Young, Carson"), ("4", "Smith, Jordan")],
    )
    assert players.loc["40098", "hist_id"] == "9037"
    assert players.loc["40098", "match_type"] == "first_name_prefix"
    # Similar-looking but different golfers stay unlinked.
    for aid in ("55721", "3", "4"):
        assert pd.isna(players.loc[aid, "hist_id"])
    assert set(players.loc[["H1", "H2"], "api_id"].isna()) == {True}
    # Prefix links are listed for audit; the Carson/Cameron near-miss is flagged, not linked.
    assert ("40098", "first_name_prefix") in set(zip(review["api_id"], review["match_type"]))
    assert ("3", "unlinked_near_match") in set(zip(review["api_id"], review["match_type"]))


def test_ambiguous_matches_are_never_linked():
    # Two training ids share a name → neither links.
    players, review = _players(
        [("10", "John Smith"), ("11", "John Smith")],
        [("500", "Smith, John")],
    )
    assert pd.isna(players.loc["500", "hist_id"])
    assert set(review["match_type"]) == {"ambiguous_name"}

    # One short first name fits two full ones → neither links.
    players, review = _players(
        [("20", "Alexander Noren"), ("21", "Alexis Noren")],
        [("600", "Noren, Alex")],
    )
    assert pd.isna(players.loc["600", "hist_id"])
    assert set(review["match_type"]) == {"ambiguous_prefix"}


def test_every_spelling_of_an_api_id_is_matched_and_kept_as_alias():
    players, _ = _players(
        [("4351896", "Kristoffer Ventura")],
        [("35658", "Ventura, Kris"), ("35658", "Kristoffer Ventura"), ("35658", "Kristoffer Ventura")],
    )
    row = players.loc["35658"]
    assert row["hist_id"] == "4351896" and row["match_type"] == "exact"
    assert row["name"] == "Kristoffer Ventura"      # most frequent spelling
    assert row["aliases"] == "Kris Ventura"


def test_manual_links_override_rules_and_skip_missing_ids():
    hist = pd.DataFrame([("4566443", "Matthias Schmid")], columns=["player_id", "player_name"])
    api = pd.DataFrame([("48867", "Schmid, Matti")], columns=["player_id", "player_name"])

    players, review = build_players(hist, api, manual={})
    assert pd.isna(players.set_index("player_id").loc["48867", "hist_id"])
    assert list(review["match_type"]) == ["unlinked_near_match"]

    players, review = build_players(hist, api, manual={"48867": "4566443", "1": "2"})
    row = players.set_index("player_id").loc["48867"]
    assert row["hist_id"] == "4566443" and row["match_type"] == "manual"
    assert row["aliases"] == "Matthias Schmid"
    assert review.empty
    assert len(players) == 1          # the override with absent ids is ignored


def test_unlinked_training_ids_are_prefixed_so_id_spaces_cannot_collide():
    players, _ = _players([("100", "Old Timer")], [("100", "Newcomer, Young")])
    assert players.loc["H100", "name"] == "Old Timer"
    assert players.loc["100", "name"] == "Young Newcomer"


def test_connect_is_read_only(tmp_path):
    db = tmp_path / "agent.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.commit()
    con.close()

    ro = connect(db)
    assert ro.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO t VALUES (1)")


def test_connect_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect(tmp_path / "nope.sqlite")

"""Backfill → ingest: season codes, the scraped-file loader, and combining the two id spaces."""

import pandas as pd

from src import scrape
from src.backfill import season_code
from src.ingest import combine, load_scraped


def test_season_code_handles_the_double_2023_season():
    assert season_code(2024, pd.Timestamp("2024-04-11")) == 20240
    assert season_code(2023, pd.Timestamp("2022-11-17")) == 20230    # fall 2022 playing
    assert season_code(2023, pd.Timestamp("2023-11-16")) == 20231    # fall 2023 playing


def test_results_for_another_season_are_rejected(monkeypatch):
    sent = []

    def fake_post(query, variables=None):
        sent.append(variables)
        # An unrecognised season code is ignored: the latest edition comes back.
        edition = "R2023493" if variables["year"] == 20230 else "R2025493"
        return {"tournamentPastResults": {"id": edition, "players": [{
            "id": "01706", "player": {"displayName": "Davis Love III"}, "position": "T5",
            "total": "270", "parRelativeScore": "-10", "additionalData": ["", "$100.00"],
            "rounds": [{"score": "68"}] * 4}]}}

    monkeypatch.setattr(scrape, "_post", fake_post)
    df = scrape.get_tournament_results("R2023493", 2023)
    assert sent[0]["year"] == 20230                                   # a season code, not 2023
    assert df.iloc[0][["player_id", "finish_num", "made_cut"]].tolist() == ["01706", 5, True]
    assert scrape.get_tournament_results("R2024493", 2024).empty      # got 2025's leaderboard back


def _hist():
    rows = [   # (player_id, name, tournament_id, date, finish, course)
        ("9478", "Scottie Scheffler", "401", "2022-03-06", 1, "Bay Hill - Orlando, FL"),
        ("111", "Old Timer", "401", "2022-03-06", 2, "Bay Hill - Orlando, FL"),
        ("9478", "Scottie Scheffler", "402", "2022-06-05", 3, "Muirfield Village Golf Club - Dublin, OH"),
    ]
    df = pd.DataFrame(rows, columns=["player_id", "player_name", "tournament_id", "date", "finish_num", "course"])
    return df.assign(date=pd.to_datetime(df["date"]), season=2022, made_cut=True, source="historical")


def _write_scraped(folder):
    rows = [   # 2022 Bay Hill is in both sources; the 2023 edition is new
        ("46046", "Scottie Scheffler", "R2022009", 2022, "2022-03-03", 1.0, True, "Arnold Palmer's Bay Hill Club & Lodge"),
        ("46046", "Scottie Scheffler", "R2023009", 2023, "2023-03-02", 4.0, True, "Arnold Palmer's Bay Hill Club & Lodge"),
        ("01706", "Davis Love III", "R2023009", 2023, "2023-03-02", None, False, "Arnold Palmer's Bay Hill Club & Lodge"),
    ]
    cols = ["player_id", "player_name", "tournament_id", "season", "date", "finish_num", "made_cut", "course"]
    df = pd.DataFrame(rows, columns=cols).assign(rounds_played=4, purse="$20,000,000",
                                                 tournament_name="Arnold Palmer Invitational")
    for tid, g in df.groupby("tournament_id"):
        g.to_csv(folder / f"{tid}.csv", index=False)


def test_load_scraped_keeps_ids_cuts_and_prize_share(tmp_path):
    _write_scraped(tmp_path)
    df = load_scraped(tmp_path).set_index(["tournament_id", "player_id"])
    assert df.loc[("R2023009", "01706"), "made_cut"] == False         # noqa: E712  (leading zero kept, cut kept)
    assert df.loc[("R2023009", "01706"), "prize_pct"] == 0.0
    assert df.loc[("R2022009", "46046"), "prize_pct"] == 0.18
    assert df.loc[("R2022009", "46046"), "purse"] == 20.0


def test_combine_puts_both_sources_on_one_player_id(tmp_path):
    _write_scraped(tmp_path)
    df = combine(_hist(), load_scraped(tmp_path))

    assert "R2022009" not in set(df["tournament_id"])                 # already covered by the historical CSV
    scheffler = df[df["player_name"] == "Scottie Scheffler"]
    assert set(scheffler["player_id"]) == {"46046"}                   # one id across the boundary
    assert set(scheffler["hist_player_id"].dropna()) == {"9478"}
    assert set(df.loc[df["player_name"] == "Old Timer", "player_id"]) == {"H111"}   # unlinked: prefixed
    # Bay Hill is one course under both spellings, learned from the shared 2022 event.
    assert df[df["tournament_id"].isin(["401", "R2023009"])]["course_key"].nunique() == 1

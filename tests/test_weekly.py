"""Weekly pipeline: projections-file selection, variant choice, live features, pickle stamp."""

import os

import pandas as pd
import pytest

from src import features, model, optimizer, owgr, scrape
from src.predict import select_variant


class _FakeModel:      # stands in for a fitted XGBRanker; only the stamp matters
    pass


def test_latest_projections_picks_newest_variant_file(tmp_path, monkeypatch):
    monkeypatch.setattr(optimizer, "PROCESSED_DIR", tmp_path)
    old = tmp_path / "R2026013_wyndham" / "projections.csv"
    new = tmp_path / "R2026478_hero" / "projections_nocut.csv"     # a no-cut week writes only this
    for i, p in enumerate([old, new]):
        p.parent.mkdir()
        p.write_text("player_id\n")
        os.utime(p, (1000 + i, 1000 + i))
    assert optimizer._latest_projections() == new


def test_links_is_opt_in_and_nocut_is_auto():
    assert select_variant("The Open Championship") == "base"
    assert select_variant("Genesis Scottish Open") == "base"
    assert select_variant("The Open Championship", links=True) == "links"
    assert select_variant("TOUR Championship") == "nocut"
    assert select_variant("TOUR Championship", nocut=False) == "base"


def test_rolling_sg_includes_latest_event_on_rows_without_sg():
    # A missed cut (NaN) after a recorded event must still see that event.
    sg = pd.Series([1.0, float("nan"), 3.0, float("nan")])
    out = features._trailing_valid_mean(sg, 8)
    assert pd.isna(out[0])
    assert out.tolist()[1:] == [1.0, 1.0, 2.0]


def test_owgr_comes_from_the_last_snapshot_before_the_event():
    snaps = pd.DataFrame({"snapshot_date": pd.to_datetime(["2026-03-08", "2026-03-15", "2026-03-22"]),
                          "player_id": "1", "owgr_rank": [10, 5, 1], "owgr_points": [4.0, 6.5, 9.0]})
    rows = pd.DataFrame({
        "player_id": ["1", "1", "2", "1"],
        "source": ["scraped", "historical", "scraped", "scraped"],
        # a Thursday start; the same event's Sunday finish (its own snapshot is dated that day);
        # an unranked player; an event before any snapshot
        "date": pd.to_datetime(["2026-03-19", "2026-03-22", "2026-03-19", "2026-03-05"])})
    out = features.add_owgr(rows, snaps)
    assert out["owgr_rank"].tolist()[:2] == [5, 5]
    assert out["owgr_points"].tolist()[:2] == [6.5, 6.5]
    assert out["owgr_rank"].isna().tolist() == [False, False, True, True]


def test_weekly_features_use_local_history(monkeypatch):
    # Player "1": ten weekly starts at Test GC, sg_total 0..9, every cut made.
    history = pd.DataFrame([{
        "player_id": "1", "player_name": "Has History", "tournament_id": f"E{i}",
        "tournament_name": "Some Open", "course": "Test GC", "course_key": "test gc",
        "date": pd.Timestamp("2026-01-01") + pd.Timedelta(weeks=i),
        "rounds_played": 4.0, "made_cut": True, "sg_total_per_round": float(i),
        "prize_pct": 0.01 * i, "source": "scraped"} for i in range(10)])
    monkeypatch.setattr(scrape, "get_field", lambda tid: [
        {"player_id": "1", "player_name": "Has History", "owgr": 5},
        {"player_id": "2", "player_name": "No History", "owgr": 900}])
    monkeypatch.setattr(owgr, "load_snapshots", lambda: pd.DataFrame({
        "snapshot_date": [pd.Timestamp("2026-03-15")], "player_id": ["1"],
        "owgr_rank": [5], "owgr_points": [6.5]}))
    monkeypatch.setattr(scrape, "get_upcoming_tournaments", lambda n=10: [
        {"tournament_id": "R1", "tournament_name": "Test Classic", "course": "Test GC",
         "start_date": pd.Timestamp("2026-03-19").value // 10**6}])

    df = scrape.build_weekly_features("R1", history=history).set_index("player_id")

    assert df.loc["1", "sg_total_per_round_roll8"] == 5.5    # last 8 values (2..9), latest included
    assert df.loc["1", "sg_total_per_round_roll24"] == 4.5   # all 10: no 10-event cap
    assert df.loc["1", "starts_last8"] == 8
    assert df.loc["1", "made_cut_rate_roll8"] == 1.0
    assert df.loc["1", "prize_pct_roll8"] == pytest.approx(0.055)   # mean of 0.02..0.09
    assert df.loc["1", "course_starts"] == 10 and df.loc["1", "course_sg_hist"] == 4.5
    assert df.loc["1", "owgr_rank"] == 5 and pd.isna(df.loc["2", "owgr_rank"])
    assert pd.isna(df.loc["2", "sg_total_per_round_roll8"])
    for col in ("course_starts", "links_starts", "playoff_starts"):
        assert df.loc["2", col] == 0                         # 0, not NaN, when there is no history


def test_staleness_warning(tmp_path, monkeypatch):
    matrix = tmp_path / "feature_matrix.parquet"
    matrix.write_bytes(b"v1")
    monkeypatch.setattr(model, "FEATURE_MATRIX", matrix)
    monkeypatch.setattr(model, "VARIANT_PATHS", {"base": tmp_path / "model_base.pkl"})

    m = _FakeModel()
    assert "no feature-matrix stamp" in model.staleness_warning(m)
    model.save(m)
    assert model.staleness_warning(m) is None
    matrix.write_bytes(b"v2")                               # matrix regenerated after training
    assert "stale pickle" in model.staleness_warning(m)

"""Weekly pipeline: projections-file selection, variant choice, live features, pickle stamp."""

import os

import pandas as pd

from src import model, optimizer, scrape
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


def test_weekly_features_match_training_encoding(monkeypatch):
    history = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10, freq="7D"),
        "tournament_name": ["Some Open"] * 10,
        **{stat: [float(i) for i in range(10)] for stat in scrape.SG_STAT_IDS},
    })
    monkeypatch.setattr(scrape, "get_field", lambda tid: [
        {"player_id": "1", "player_name": "Has History", "owgr": 5},
        {"player_id": "2", "player_name": "No History", "owgr": 900}])
    monkeypatch.setattr(scrape, "get_upcoming_tournaments", lambda n=10: [
        {"tournament_id": "R1", "tournament_name": "Test Classic", "course": "Test GC"}])
    monkeypatch.setattr(scrape, "get_player_sg_history",
                        lambda pid, n_events=10: history if pid == "1" else pd.DataFrame())
    monkeypatch.setattr(scrape, "get_player_cut_history", lambda pid, n_events=24: [])
    monkeypatch.setattr(scrape.time, "sleep", lambda s: None)

    df = scrape.build_weekly_features("R1").set_index("player_id")

    assert df.loc["1", "starts_last8"] == 8                 # 10 events of history, capped like training
    assert df.loc["1", "sg_total_per_round_roll8"] == 5.5   # mean of the last 8 values (2..9)
    for col in ("course_starts", "links_starts", "playoff_starts"):
        assert (df[col] == 0).all()                         # 0, not NaN, when there is no history


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

"""
End-to-end data pipeline.

Typical usage:
    python -m src.pipeline

Steps:
    1. Load historical CSV (2014-2022)
    2. Load any scraped CSVs from data/raw/scraped/
    3. Combine into one DataFrame
    4. Engineer rolling features (no look-ahead leakage)
    5. Save to data/processed/feature_matrix.parquet

Run the scraper first to get 2023+ data:
    python -m src.scrape --years 2023 2024 2025 --dir data/raw/scraped
"""

from pathlib import Path
import pandas as pd

from src.ingest import load_historical, load_scraped, combine
from src.features import build_features, feature_columns

# ── Default paths (relative to repo root) ─────────────────────────────────────

HISTORICAL_CSV = Path("14-22 practice data.csv")
SCRAPED_DIR = Path("data/raw/scraped")
OUTPUT_PATH = Path("data/processed/feature_matrix.parquet")


def run(
    historical_path: Path = HISTORICAL_CSV,
    scraped_dir: Path = SCRAPED_DIR,
    output_path: Path = OUTPUT_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Step 1: Load historical data ──────────────────────────────────────────
    log(f"Loading historical data from {historical_path} ...")
    hist = load_historical(historical_path)
    log(f"  {len(hist):,} rows | {hist['season'].min()}-{hist['season'].max()} seasons")

    # ── Step 2: Load scraped data ─────────────────────────────────────────────
    scraped_files = list(scraped_dir.glob("R????.csv")) if scraped_dir.exists() else []
    if scraped_files:
        log(f"Loading {len(scraped_files)} scraped CSVs from {scraped_dir} ...")
        scraped = load_scraped(scraped_dir)
        log(f"  {len(scraped):,} rows | seasons: {sorted(scraped['season'].unique())}")
    else:
        log(f"No scraped CSVs found in {scraped_dir} — using historical data only.")
        log("  Run: python -m src.scrape --years 2023 2024 2025")
        scraped = pd.DataFrame()

    # ── Step 3: Combine ───────────────────────────────────────────────────────
    log("Combining sources ...")
    df = combine(hist, scraped)
    log(f"  {len(df):,} total rows after deduplication")

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    log("Engineering features ...")
    df = build_features(df)
    feat_cols = feature_columns(df)
    log(f"  {len(feat_cols)} feature columns added")

    # Coverage report: what fraction of rows have each key feature
    for col in ["sg_total_per_round_roll8", "course_sg_hist", "made_cut_rate_roll8"]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            log(f"  {col}: {pct:.1f}% coverage")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log(f"\nSaved {len(df):,} rows × {len(df.columns)} columns → {output_path}")

    return df


# ── Summary helpers ────────────────────────────────────────────────────────────

def load_processed(path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Load the processed feature matrix from parquet."""
    return pd.read_parquet(path)


def summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the feature matrix."""
    print(f"\nShape: {df.shape}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Players: {df['player_id'].nunique():,}")
    print(f"Tournaments: {df['tournament_id'].nunique():,}")
    print(f"Made cut: {df['made_cut'].mean():.1%}")

    feat_cols = feature_columns(df)
    print(f"\nFeature columns ({len(feat_cols)}):")
    for c in sorted(feat_cols):
        pct = df[c].notna().mean() * 100
        print(f"  {c:<40} {pct:5.1f}% non-null")

    if "dk_points" in df.columns:
        dk = df["dk_points"].dropna()
        print(f"\nTarget (dk_points): mean={dk.mean():.1f}, std={dk.std():.1f}, "
              f"min={dk.min():.1f}, max={dk.max():.1f}")


if __name__ == "__main__":
    df = run()
    summary(df)

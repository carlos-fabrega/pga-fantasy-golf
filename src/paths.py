"""
Centralized filesystem paths.

Every module imports its paths from here so the project layout lives in one
place. Paths are anchored to the repository root (this file's grandparent) so
scripts resolve correctly regardless of the current working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Raw inputs ────────────────────────────────────────────────────────────────
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
HISTORICAL_CSV = RAW_DIR / "historical" / "14-22 practice data.csv"
SCRAPED_DIR    = RAW_DIR / "scraped"

# ── Processed artifacts ───────────────────────────────────────────────────────
PROCESSED_DIR  = DATA_DIR / "processed"
FEATURE_MATRIX = PROCESSED_DIR / "feature_matrix.parquet"

# ── Trained models ────────────────────────────────────────────────────────────
MODELS_DIR       = PROJECT_ROOT / "models"
MODEL_BASE_PATH    = MODELS_DIR / "model_base.pkl"
MODEL_LINKS_PATH   = MODELS_DIR / "model_links.pkl"
MODEL_PLAYOFF_PATH = MODELS_DIR / "model_playoff.pkl"
MODEL_NOCUT_PATH   = MODELS_DIR / "model_nocut.pkl"

## paths, seeds, feature lists, and classification threshold

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
THRESHOLD = 0.5
FEATURES = ["mean_depth", "volume_dam", "volume_gap", "uncertainty_frac", "expansion_frac"]
TARGET = "glof_within_50yr"
GROUP_COL = "lake_id"

DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "glof_clean.csv"

MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "glof-critical-area"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model.json"

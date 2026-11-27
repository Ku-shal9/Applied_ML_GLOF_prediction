## preprocessing and random forest model pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.config import SEED
except ImportError:
    from config import SEED


# build scaler and random forest pipeline without smote
def get_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

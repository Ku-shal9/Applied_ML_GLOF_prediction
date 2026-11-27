## train random forest model, log metrics to mlflow, and save model artifacts

import json
import logging

import joblib
import mlflow
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

try:
    from src.config import (
        FEATURES,
        METADATA_PATH,
        MLFLOW_EXPERIMENT,
        MLFLOW_URI,
        MODEL_PATH,
        SEED,
        THRESHOLD,
    )
    from src.data import load_data, split_data
    from src.features import get_pipeline
except ImportError:
    from config import (
        FEATURES,
        METADATA_PATH,
        MLFLOW_EXPERIMENT,
        MLFLOW_URI,
        MODEL_PATH,
        SEED,
        THRESHOLD,
    )
    from data import load_data, split_data
    from features import get_pipeline

logging.getLogger("mlflow").setLevel(logging.WARNING)


# train model, log run, and save joblib + json metadata
def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y, groups = load_data()
    X_train, X_test, y_train, y_test, _, _ = split_data(X, y, groups)

    pipeline = get_pipeline()

    with mlflow.start_run(run_name="final-rf"):
        mlflow.log_params({"seed": SEED, "model": "random_forest", "smote": False})
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)

        metrics = {
            "average_precision": float(average_precision_score(y_test, proba)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
        }
        mlflow.log_metric("test_average_precision", metrics["average_precision"])
        mlflow.log_metric("test_roc_auc", metrics["roc_auc"])
        mlflow.log_metric("test_precision", metrics["precision"])
        mlflow.log_metric("test_recall", metrics["recall"])

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        meta = {
            "model": "random_forest",
            "smote": False,
            "threshold": THRESHOLD,
            "features": FEATURES,
            "metrics": metrics,
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METADATA_PATH))

    print("training complete:", metrics)


if __name__ == "__main__":
    main()

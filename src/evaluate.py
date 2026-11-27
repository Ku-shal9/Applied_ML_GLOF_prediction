## test-set and slice evaluation for the saved model

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

try:
    from src.config import MODEL_PATH, THRESHOLD
    from src.data import load_data, split_data
except ImportError:
    from config import MODEL_PATH, THRESHOLD
    from data import load_data, split_data


# compute classification metrics for a given subset
def compute_metrics(y_true, proba):
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
    }


# evaluate saved model on test set and expansion_frac slices
def main():
    pipeline = joblib.load(MODEL_PATH)
    X, y, groups = load_data()
    _, X_test, _, y_test, _, _ = split_data(X, y, groups)

    proba = pipeline.predict_proba(X_test)[:, 1]
    print("overall test metrics:", compute_metrics(y_test, proba))

    # slice evaluation by median expansion_frac
    median_exp = X_test["expansion_frac"].median()
    slices = {
        "low_expansion": X_test["expansion_frac"] <= median_exp,
        "high_expansion": X_test["expansion_frac"] > median_exp,
    }
    rows = []
    for name, mask in slices.items():
        if y_test[mask].nunique() > 1:
            row = compute_metrics(y_test[mask], proba[mask])
            row["slice"] = name
            rows.append(row)
    if rows:
        print("\nslice metrics:")
        print(pd.DataFrame(rows).set_index("slice"))


if __name__ == "__main__":
    main()

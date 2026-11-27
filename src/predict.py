## load saved model artifact and generate predictions

import joblib
import pandas as pd

try:
    from src.config import FEATURES, MODEL_PATH, THRESHOLD
except ImportError:
    from config import FEATURES, MODEL_PATH, THRESHOLD


# load fitted pipeline from disk
def load_model(path=MODEL_PATH):
    return joblib.load(path)


# return predicted probabilities and binary labels
def predict(df, model=None):
    if model is None:
        model = load_model()
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return proba, pred


# run sample prediction on test split when executed directly
def main():
    try:
        from src.data import load_data, split_data
    except ImportError:
        from data import load_data, split_data

    X, y, groups = load_data()
    _, X_test, _, _, _, _ = split_data(X, y, groups)
    proba, pred = predict(X_test.head(5))
    out = pd.DataFrame({"probability": proba.round(4), "prediction": pred})
    print(out)


if __name__ == "__main__":
    main()

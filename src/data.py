## data loading, cleaning, and grouped train-test splitting

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.config import DATA_PATH, FEATURES, GROUP_COL, SEED, TARGET
except ImportError:
    from config import DATA_PATH, FEATURES, GROUP_COL, SEED, TARGET


# load dataset and drop rows with missing features or target
def load_data(path=DATA_PATH):
    data = pd.read_csv(path)
    cols = FEATURES + [TARGET]
    data = data.dropna(subset=cols).reset_index(drop=True)
    X = data[FEATURES]
    y = data[TARGET]
    groups = data[GROUP_COL] if GROUP_COL in data.columns else pd.Series(np.arange(len(data)))
    return X, y, groups


# split data by lake_id to prevent data leakage
def split_data(X, y, groups, test_size=0.2):
    unique_lakes = list(groups.unique())
    train_lakes, _ = train_test_split(unique_lakes, test_size=test_size, random_state=SEED)
    train_mask = groups.isin(train_lakes)
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    groups_train, groups_test = groups[train_mask], groups[~train_mask]
    return X_train, X_test, y_train, y_test, groups_train, groups_test

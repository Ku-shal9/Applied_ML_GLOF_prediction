import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

os.environ["MPLCONFIGDIR"] = "/Users/kushalthapa/Desktop/Applied ML/.mplconfig"
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md("# GLOF critical area — training and testing")

code(
    """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier"""
)

code(
    """glof = pd.read_csv("glof_clean.csv")
glof.head()"""
)

code(
    """print(glof.shape)
print(glof.columns.tolist())
print(glof["glof_within_50yr"].value_counts())"""
)

code(
    """# - label out. lake_id is only for grouping.
# - pct and yrs left out (they *are* the target).
# - peak discharge left out (old formula).

FEATURES = [
    "year",
    "lake_area_km2",
    "elevation_m",
    "perimeter_km",
    "area_uncertainty_km2",
    "expansion_rate_km2yr",
    "lake_volume_1e6_m3",
    "dam_height_m",
    "shape_index",
    "expansion_accel_km2yr2",
    "dam_stability_index",
    "lon",
    "lat",
    "mean_depth",
    "volume_gap",
    "uncertainty_frac",
    "expansion_frac",
]
TARGET = "glof_within_50yr"

X = glof[FEATURES]
y = glof[TARGET]
groups = glof["lake_id"]
print(X.shape)
print(y.mean())"""
)

md("## sampling")

code(
    """# - random rows

X_tr_rand, X_te_rand, y_tr_rand, y_te_rand = train_test_split(
    X, y, test_size=0.25, random_state=42
)
overlap = set(glof.loc[X_tr_rand.index, "lake_id"]) & set(glof.loc[X_te_rand.index, "lake_id"])
print("random split, lakes on both sides:", len(overlap))"""
)

code(
    """# - stratified on the label: still leaks lakes

X_tr_str, X_te_str, y_tr_str, y_te_str = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
overlap_str = set(glof.loc[X_tr_str.index, "lake_id"]) & set(glof.loc[X_te_str.index, "lake_id"])
print("stratified split, lakes on both sides:", len(overlap_str))
print("positive share train", y_tr_str.mean(), "test", y_te_str.mean())"""
)

code(
    """# - grouped by lake: the split we actually use

lakes = glof["lake_id"].drop_duplicates().to_numpy()
np.random.seed(42)
np.random.shuffle(lakes)
cut = int(0.75 * len(lakes))
train_lakes = lakes[:cut]
test_lakes = lakes[cut:]

train_mask = glof["lake_id"].isin(train_lakes)
test_mask = glof["lake_id"].isin(test_lakes)

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]
groups_train = groups[train_mask]

print("grouped split train", len(X_train), "test", len(X_test))
print("lakes on both sides:", len(set(glof.loc[train_mask, "lake_id"]) & set(glof.loc[test_mask, "lake_id"])))
print("positive share train", y_train.mean(), "test", y_test.mean())
print("test locked away")"""
)

md("## pipeline")

code(
    """numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, FEATURES),
])

def build(estimator):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

preprocessor"""
)

code(
    """def class_report(name, y_true, y_pred, y_score=None):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    print(name)
    print(f"  precision {prec:.3f}")
    print(f"  recall    {rec:.3f}")
    print(f"  roc-auc   {auc:.3f}")
    return {"model": name, "precision": prec, "recall": rec, "roc_auc": auc}"""
)

md("## baselines")

code(
    """dummy = build(DummyClassifier(strategy="most_frequent"))
dummy.fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
class_report("dummy majority", y_test, pred_dummy, pred_dummy)"""
)

code(
    """# - expanding lake already near 0.005 km2

pred_rule = ((X_test["expansion_rate_km2yr"] > 0) & (X_test["lake_area_km2"] >= 0.004)).astype(int)
class_report("expansion rule", y_test, pred_rule, pred_rule)"""
)

md("## one model at a time")

code(
    """logreg = build(LogisticRegression(max_iter=1000, class_weight="balanced"))
logreg.fit(X_train, y_train)
pred_log = logreg.predict(X_test)
score_log = logreg.predict_proba(X_test)[:, 1]
class_report("logistic regression", y_test, pred_log, score_log)"""
)

code(
    """tree = build(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42))
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)
score_tree = tree.predict_proba(X_test)[:, 1]
class_report("decision tree depth 5", y_test, pred_tree, score_tree)"""
)

code(
    """forest = build(RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1))
forest.fit(X_train, y_train)
pred_forest = forest.predict(X_test)
score_forest = forest.predict_proba(X_test)[:, 1]
class_report("random forest", y_test, pred_forest, score_forest)"""
)

code(
    """boost = build(HistGradientBoostingClassifier(random_state=42))
boost.fit(X_train, y_train)
pred_boost = boost.predict(X_test)
score_boost = boost.predict_proba(X_test)[:, 1]
class_report("hist gradient boosting", y_test, pred_boost, score_boost)"""
)

code(
    """knn = build(KNeighborsClassifier(n_neighbors=15))
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
score_knn = knn.predict_proba(X_test)[:, 1]
class_report("knn k=15", y_test, pred_knn, score_knn)"""
)

md("## cross-validation on train")

code(
    """### stratified kfold vs groupkfold"""
)

code(
    """log_cv = build(LogisticRegression(max_iter=1000, class_weight="balanced"))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=5)

skf_scores = cross_val_score(log_cv, X_train, y_train, cv=skf, scoring="roc_auc", error_score=np.nan)
gkf_scores = cross_val_score(
    log_cv, X_train, y_train, cv=gkf, groups=groups_train, scoring="roc_auc", error_score=np.nan
)

print("stratified kfold AUC", skf_scores.round(3), "mean", round(skf_scores.mean(), 3), "std", round(skf_scores.std(), 3))
print("groupkfold AUC     ", gkf_scores.round(3), "mean", round(gkf_scores.mean(), 3), "std", round(gkf_scores.std(), 3))"""
)

code(
    """# - compare families with grouped cv (roc-auc)

candidates = {
    "dummy majority": DummyClassifier(strategy="most_frequent"),
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "decision tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "random forest": RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1),
}

cv = GroupKFold(n_splits=5)
rows = []
for name, estimator in candidates.items():
    pipe = build(estimator)
    auc = cross_val_score(
        pipe, X_train, y_train, cv=cv, groups=groups_train, scoring="roc_auc", error_score=np.nan
    )
    rows.append({"model": name, "cv_AUC_mean": auc.mean(), "cv_AUC_std": auc.std()})
    print(name, "AUC", round(auc.mean(), 3), "±", round(auc.std(), 3))

cv_table = pd.DataFrame(rows).set_index("model").round(3)
cv_table"""
)

md("## test metrics (once)")

code(
    """rows = []
rows.append(class_report("dummy majority", y_test, pred_dummy, pred_dummy))
rows.append(class_report("expansion rule", y_test, pred_rule, pred_rule))
rows.append(class_report("logistic regression", y_test, pred_log, score_log))
rows.append(class_report("decision tree depth 5", y_test, pred_tree, score_tree))
rows.append(class_report("random forest", y_test, pred_forest, score_forest))
rows.append(class_report("hist gradient boosting", y_test, pred_boost, score_boost))
rows.append(class_report("knn k=15", y_test, pred_knn, score_knn))

test_table = pd.DataFrame(rows).set_index("model").round(3)
test_table"""
)

code(
    """print(classification_report(y_test, pred_forest, digits=3, zero_division=0))
cm = confusion_matrix(y_test, pred_forest)
print("tn fp fn tp", cm.ravel())"""
)

code(
    """test_table[["precision", "recall", "roc_auc"]].plot.bar(rot=25)
plt.ylabel("score")
plt.title("test — higher is better")
plt.tight_layout()
plt.show()"""
)

md(
    """## perceptron

same yes/no target. a linear perceptron is the cheap check for later deep learning.

if it is close to the forest, a neural net is unlikely to earn its cost. if it is weak, trees already cover the non-linear part.
"""
)

code(
    """perceptron = build(Perceptron())
perceptron.fit(X_train, y_train)
pred_p = perceptron.predict(X_test)
score_p = perceptron.decision_function(X_test)
class_report("perceptron", y_test, pred_p, score_p)"""
)

code(
    """clf_candidates = {
    "perceptron": Perceptron(),
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random forest clf": RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1),
}

rows = []
for name, estimator in clf_candidates.items():
    pipe = build(estimator)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        score = pipe.predict_proba(X_test)[:, 1]
    else:
        score = pipe.decision_function(X_test)
    rows.append(class_report(name, y_test, pred, score))

pd.DataFrame(rows).set_index("model").round(3)"""
)

nb["cells"] = cells
out = Path("/Users/kushalthapa/Desktop/Applied ML/Applied_ML_GLOF_prediction/notebooks/ml_training_testing.ipynb")
nbf.write(nb, out)

client = NotebookClient(
    nb,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": str(out.parent)}},
)
client.execute()
nbf.write(nb, out)
print("done", len(nb.cells))

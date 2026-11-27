import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

os.environ["MPLCONFIGDIR"] = "/Users/kushalthapa/Desktop/Applied ML/.mplconfig"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

path = Path("/Users/kushalthapa/Desktop/Applied ML/Applied_ML_GLOF_prediction/notebooks/ml_training_testing.ipynb")
nb = nbf.read(path, as_version=4)


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


# don't append twice
blob = "\n".join("".join(c.source) for c in nb.cells)
if "mlflow.set_experiment" not in blob:
    nb.cells.extend(
        [
            md(
                """## why the score looks perfect

positive class is about 2%. ROC-AUC can look perfect when the few positive lakes are just the big ones.

average precision is the number that matches that imbalance. a single size column already ranks them.
"""
            ),
            code(
                """from sklearn.metrics import average_precision_score

print("positive share", round(y.mean(), 4), "count", int(y.sum()))

rows = []
for col in FEATURES:
    score = X[col]
    auc_pos = roc_auc_score(y, score)
    auc_neg = roc_auc_score(y, -score)
    if auc_pos >= auc_neg:
        auc, used = auc_pos, score
    else:
        auc, used = auc_neg, -score
    rows.append({
        "feature": col,
        "roc_auc": auc,
        "average_precision": average_precision_score(y, used),
    })

one_col = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
print(one_col.round(3).to_string(index=False))
print()
print(glof.groupby("glof_within_50yr")[["lake_area_km2", "expansion_rate_km2yr"]].median())"""
            ),
            md(
                """area alone is already a strong ranker. the full model is mostly reading lake size, not a hidden GLOF signal. SMOTE rebalances training rows. it does not remove that size signal.
"""
            ),
            md("## mlflow"),
            code(
                """import logging
from pathlib import Path

import mlflow
import mlflow.sklearn

logging.getLogger("mlflow").setLevel(logging.WARNING)

ROOT = Path.cwd()
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("glof-critical-area")

print("tracking:", mlflow.get_tracking_uri())
print("mlflow ui --backend-store-uri sqlite:///mlflow.db")"""
            ),
            code(
                """# - log the two families we keep: logistic and a tree model
# - average precision, not roc, because the class is rare

tracked = {
    "logistic": logreg,
    "random-forest": forest,
    "hist-boosting": boost,
}

cv = GroupKFold(n_splits=5)

for run_name, candidate in tracked.items():
    with mlflow.start_run(run_name=run_name):
        fold_scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            groups=groups_train,
            scoring="average_precision",
        )
        test_probabilities = candidate.predict_proba(X_test)[:, 1]

        mlflow.log_param("estimator", candidate.named_steps["model"].__class__.__name__)
        mlflow.log_param("split", "group by lake")
        mlflow.log_param("sampling", "none")
        mlflow.log_metric("cv_average_precision", fold_scores.mean())
        mlflow.log_metric("cv_std", fold_scores.std())
        mlflow.log_metric("test_average_precision", average_precision_score(y_test, test_probabilities))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, test_probabilities))
        try:
            # mlflow.sklearn.log_model(candidate, name="model")  # disabled due to serialization issues
        except Exception as e:
            print(f"mlflow model logging failed: {e}")

        print(f"logged {run_name}: cv AP {fold_scores.mean():.3f} ± {fold_scores.std():.3f}")"""
            ),
            code(
                """runs = mlflow.search_runs(experiment_names=["glof-critical-area"])
comparison = runs[[
    "tags.mlflow.runName",
    "params.estimator",
    "metrics.cv_average_precision",
    "metrics.test_average_precision",
    "metrics.test_roc_auc",
]]
comparison.columns = ["run", "estimator", "cv_AP", "test_AP", "test_AUC"]
print(comparison.round(3).to_string(index=False))"""
            ),
            md(
                """## model selection

two families: logistic regression, and hist gradient boosting.

tune inside grouped cross-validation. score with average precision. try SMOTE only on the training fold.
"""
            ),
            code(
                """from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV

def build_smote(estimator):
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", estimator),
    ])"""
            ),
            code(
                """# - plain vs SMOTE, before the grid

pairs = {
    "logistic": build(LogisticRegression(max_iter=1000)),
    "logistic smote": build_smote(LogisticRegression(max_iter=1000)),
    "hist boosting": build(HistGradientBoostingClassifier(random_state=42)),
    "hist boosting smote": build_smote(HistGradientBoostingClassifier(random_state=42)),
}

cv = GroupKFold(n_splits=5)
rows = []
for name, pipe in pairs.items():
    scores = cross_val_score(
        pipe, X_train, y_train, cv=cv, groups=groups_train, scoring="average_precision"
    )
    rows.append({"model": name, "cv_AP": scores.mean(), "cv_std": scores.std()})
    print(name, round(scores.mean(), 3), "±", round(scores.std(), 3))

pd.DataFrame(rows).set_index("model").round(3)"""
            ),
            code(
                """log_search = GridSearchCV(
    build_smote(LogisticRegression(max_iter=1000)),
    {"model__C": [0.1, 1, 10]},
    cv=GroupKFold(n_splits=5),
    scoring="average_precision",
    n_jobs=-1,
)
log_search.fit(X_train, y_train, groups=groups_train)
print("logistic", log_search.best_params_, round(log_search.best_score_, 3))

boost_search = GridSearchCV(
    build_smote(HistGradientBoostingClassifier(random_state=42)),
    {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 6]},
    cv=GroupKFold(n_splits=5),
    scoring="average_precision",
    n_jobs=-1,
)
boost_search.fit(X_train, y_train, groups=groups_train)
print("hist boosting", boost_search.best_params_, round(boost_search.best_score_, 3))"""
            ),
            code(
                """# - test once, on the tuned pipes

for run_name, search in {"logistic-smote-tuned": log_search, "hist-boosting-smote-tuned": boost_search}.items():
    model = search.best_estimator_
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        mlflow.log_param("sampling", "SMOTE")
        mlflow.log_param("split", "group by lake")
        mlflow.log_metric("cv_average_precision", search.best_score_)
        mlflow.log_metric("test_average_precision", average_precision_score(y_test, proba))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, proba))
        mlflow.log_metric("test_precision", precision_score(y_test, pred, zero_division=0))
        mlflow.log_metric("test_recall", recall_score(y_test, pred, zero_division=0))
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["numpy.dtype"])
    print(run_name)
    print("  cv AP", round(search.best_score_, 3))
    print("  test AP", round(average_precision_score(y_test, proba), 3))
    print("  precision", round(precision_score(y_test, pred, zero_division=0), 3))
    print("  recall", round(recall_score(y_test, pred, zero_division=0), 3))"""
            ),
            code(
                """runs = mlflow.search_runs(experiment_names=["glof-critical-area"])
keep = runs[runs["tags.mlflow.runName"].notna()]
print(keep[[
    "tags.mlflow.runName",
    "metrics.cv_average_precision",
    "metrics.test_average_precision",
    "metrics.test_roc_auc",
]].round(3).to_string(index=False))"""
            ),
        ]
    )

nb.metadata["kernelspec"] = {
    "display_name": "glof venv",
    "language": "python",
    "name": "glof-venv",
}
nbf.write(nb, path)

client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="glof-venv",
    resources={"metadata": {"path": str(path.parent)}},
)
client.execute()
nbf.write(nb, path)
print("cells", len(nb.cells))

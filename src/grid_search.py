# grid_search.py (estratto)

from pathlib import Path
import os, json, pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import SGDClassifier  # (NB: qui non serve importare LogisticRegression)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



param_grids = {
    "SGDClassifier": {
        "loss": ["hinge", "log_loss"],
        "penalty": ["l2", "l1"],
        "alpha": [1e-4, 1e-3],
    },
    "DecisionTree": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
    },
    # >>> AGGIUNTA: griglia per LogisticRegression (chiave 'LogReg' come in model_training) <<<
    "LogReg": {
        "C": [0.1, 1.0, 3.0, 10.0],
        "solver": ["lbfgs", "saga"],
        "penalty": ["l2", "none"],
        "max_iter": [500, 1000],
        "class_weight": [None, "balanced"],
    }
}
# pulizia voci None (se pacchetti mancanti)
param_grids = {k: v for k, v in param_grids.items() if v is not None}

def run_grid_search_and_save(estimator, param_grid, x_train, y_train, model_name,
                             cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42):
    folder_path = Path(__file__).resolve().parent.parent / "data" / "grid_search_results"
    folder_path.mkdir(parents=True, exist_ok=True)
    model_file = folder_path / f"{model_name}_optimized_model.pkl"

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(estimator, param_grid, cv=cv_strategy, scoring=scoring,
                      n_jobs=n_jobs, verbose=verbose, refit=True)
    gs.fit(x_train, y_train)

    best_model = gs.best_estimator_
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Migliori parametri {model_name}: {gs.best_params_}")
    print(f"Modello salvato in {model_file}")
    return best_model, gs

def _best_to_meta(model_name: str, score: float, params: dict, scoring: str, cv: int):
    """Aggiorna data/grid_search_results/model_meta.json con info sklearn."""
    artifacts = Path(__file__).resolve().parent.parent / "data" / "grid_search_results"
    artifacts.mkdir(parents=True, exist_ok=True)
    meta_path = artifacts / "model_meta.json"

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta.update({
        "sklearn_model": model_name,
        "sklearn_score": score,
        "sklearn_best_params": params,
        "scoring": scoring,
        "cv": cv
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

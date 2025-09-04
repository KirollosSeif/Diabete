# src/main.py
from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np  # NEW
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # NEW
from joblib import dump  # NEW

from src.model_training import (
    split_data,
    evaluate_models_cross_validation,
    train_keras_simple,  # allena un semplice modello Keras e restituisce (model, val_acc)
)
from src.data_preprocessing import create_db_engine, test_connection
from src.grid_search import run_grid_search_and_save as run_grid_search, param_grids

# ========= PATH & ENV =========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# DB ENV
username = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")
host     = os.getenv("SQL_HOST")
database = os.getenv("SQL_DATABASE")
port     = int(os.getenv("SQL_PORT", 3306))
TABLE_NAME = os.getenv("DB_TABLE", "diabetes_data_refreshed")

# Git config
GIT_AUTOCOMMIT_ENABLED = True
GIT_BRANCH = os.getenv("GIT_BRANCH", "main")

# Artefatti
ART_DIR = PROJECT_ROOT / "data" / "grid_search_results"
ART_DIR.mkdir(parents=True, exist_ok=True)

# ========== FUNZIONI GIT (AUTO COMMIT/PUSH) ==========
def _run_git(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out
    except Exception as e:
        return 1, f"{e}"

def git_auto_commit_push(paths: List[Path], message: Optional[str] = None) -> None:
    if not GIT_AUTOCOMMIT_ENABLED:
        print("[auto-commit] disabilitato")
        return

    rels: List[str] = []
    for p in paths:
        if not p:
            continue
        try:
            if p.exists():
                rels.append(str(p.relative_to(PROJECT_ROOT)))
        except Exception:
            rels.append(str(p))

    if not rels:
        print("[auto-commit] niente da aggiungere")
        return

    if message is None:
        message = f"Auto-update training artifacts ({datetime.now().isoformat(timespec='seconds')})"

    _run_git(["git", "pull", "--rebase", "origin", GIT_BRANCH])

    for r in rels:
        _run_git(["git", "add", "-f", r])

    code, out = _run_git(["git", "commit", "-m", message])
    if code != 0 and ("nothing to commit" in out.lower() or "no changes added to commit" in out.lower()):
        print("[auto-commit] nessuna modifica da committare")
        return

    _run_git(["git", "push", "origin", GIT_BRANCH])
    print("[auto-commit] push completato:", ", ".join(rels))

# ========== PIPELINE TRAINING ==========
def main():
    # 1) Connessione DB
    engine = create_db_engine(username, password, host, port, database)
    test_connection(engine)

    # 2) Carico dati (puliti dal DB)
    df = pd.read_sql_table(TABLE_NAME, con=engine)

    # 3) Split (STRATIFICATO) in train/test
    x_train, x_test, y_train, y_test = split_data(df, target_column="Diabetes_012")

    # 4) SCALING colonne continue (fit SOLO sul train)  # NEW
    #    - considera numeriche (esclusa target) e "continue" (nunique > 13)
    num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    continuous_cols = [c for c in num_cols if x_train[c].nunique() > 13]

    scaler_path: Optional[Path] = None
    if continuous_cols:
        scaler = StandardScaler().fit(x_train[continuous_cols])
        x_train_scaled = x_train.copy()
        x_test_scaled  = x_test.copy()
        x_train_scaled[continuous_cols] = scaler.transform(x_train[continuous_cols])
        x_test_scaled[continuous_cols]  = scaler.transform(x_test[continuous_cols])

        # salva scaler bundle (scaler + colonne)
        scaler_path = ART_DIR / "scaler.pkl"
        dump({"cols": continuous_cols, "scaler": scaler}, scaler_path)
    else:
        x_train_scaled, x_test_scaled = x_train.copy(), x_test.copy()

    # 5) Cross-validation su più modelli (scelta algoritmo migliore)
    results, best_estimator, best_model_name = evaluate_models_cross_validation(x_train_scaled, y_train)
    print("Risultati CV (accuracy media):", results)
    print("Miglior modello (CV):", best_model_name)

    # 6) Grid search SOLO sul vincitore
    estimator_cls = best_estimator.__class__
    estimator_fresh = estimator_cls()
    best_model, gs = run_grid_search(
        estimator=estimator_fresh,
        param_grid=param_grids[best_model_name],
        x_train=x_train_scaled,
        y_train=y_train,
        model_name=best_model_name,
        cv=5,
        scoring="accuracy",
        verbose=0,
    )

    try:
        best_params = gs.best_params_
        best_cv = float(gs.best_score_)
    except Exception:
        best_params, best_cv = {}, None
    print(f"GridSearch > {best_model_name} best params:", best_params)
    if best_cv is not None:
        print(f"GridSearch > {best_model_name} mean CV acc: {best_cv:.4f}")

    # 7) (Opzionale) Keras su dati GIÀ scalati  # NEW
    keras_model = None
    kr_val_acc: Optional[float] = None
    try:
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        keras_model, val_acc = train_keras_simple(x_tr, y_tr, x_val, y_val, max_epochs=50)
        kr_val_acc = float(val_acc)
        print(f"Keras val acc: {kr_val_acc:.4f}")
    except Exception as e:
        print("Keras non eseguito:", e)
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

    # ========== SALVATAGGI E META ==========
    # 7bis) Salva il modello Keras se è stato allenato
    keras_path: Optional[Path] = None
    if keras_model is not None:
        try:
            keras_path = ART_DIR / "best_keras_model.keras"
            try:
                keras_model.save(keras_path)
            except Exception:
                keras_path = ART_DIR / "best_keras_model.h5"
                keras_model.save(keras_path)
        except Exception as e:
            print("Salvataggio Keras NON riuscito:", e)
            keras_path = None

    # Modello Sklearn (proviamo a usare la convenzione <Nome>_optimized_model.pkl)
    sklearn_path = ART_DIR / f"{best_model_name.replace(' ', '_')}_optimized_model.pkl"
    if not sklearn_path.exists():
        fallback = ART_DIR / "LightGBM_optimized_model.pkl"
        sklearn_path = fallback if fallback.exists() else None

    if sklearn_path is None:
        sklearn_path = ART_DIR / "best_sklearn_model.pkl"
        try:
            from joblib import dump as _dump
            _dump(best_model, sklearn_path)
        except Exception as e:
            print("Salvataggio modello Sklearn NON riuscito:", e)
            sklearn_path = None

    # Valutazione Sklearn sul validation
    sk_val_pred = best_model.predict(x_val)
    sk_val_acc = float(accuracy_score(y_val, sk_val_pred))

    # Scelta finale tra Keras e Sklearn
    selected = "keras" if (kr_val_acc is not None and kr_val_acc >= sk_val_acc) else "sklearn"

    # Meta per lo streamlit (usato da src.utils.load_best_model / preprocess_for_inference)
    META_PATH = ART_DIR / "model_meta.json"
    meta = {
        "selected": selected,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "feature_order": list(x_train_scaled.columns),   # ordine colonne DOPO scaling
        "sklearn": {
            "name": best_model_name,
            "val_acc": sk_val_acc,
            "path": str(sklearn_path) if sklearn_path else "",
            "scaler_path": str(scaler_path) if scaler_path else "",  # NEW
        },
        "keras": {
            "val_acc": (kr_val_acc if kr_val_acc is not None else None),
            "path": str(keras_path) if keras_path else ""
        }
    }

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Modello selezionato: {selected} (sk={sk_val_acc:.4f}, keras={(kr_val_acc if kr_val_acc is not None else 'n/a')})")

    # ========== AUTO-COMMIT/PUSH ARTEFATTI ==========
    to_commit: List[Path] = [META_PATH]
    if sklearn_path and sklearn_path.exists():
        to_commit.append(sklearn_path)
    if scaler_path and scaler_path.exists():  # NEW
        to_commit.append(scaler_path)
    if keras_path and keras_path.exists():
        to_commit.append(keras_path)

    git_auto_commit_push(
        to_commit,
        message=f"Train: selected={selected}, sk_val={sk_val_acc:.4f}, keras_val={(kr_val_acc if kr_val_acc is not None else 'n/a')}"
    )

if __name__ == "__main__":
    main()

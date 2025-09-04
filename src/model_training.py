import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import ClassifierMixin

# Modelli base
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# >>> MODIFICA: import opzionali (prima erano obbligatori) <<<
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# ---------------------------------------------------------------------
# Path artefatti
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "grid_search_results"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------
def split_data(df, target_column="Diabetes_012", test_size=0.2, random_state=42):
    """Train/test split STRATIFICATO sulla target (se possibile)."""
    if target_column not in df.columns:
        raise KeyError(f"Target '{target_column}' non trovata")
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    strat = y if y.nunique() > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)

# ---------------------------------------------------------------------
# Modelli candidati + CV
# ---------------------------------------------------------------------
def _candidate_models(random_state=42, class_weight=None):
    models = {
        "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3,
                                       random_state=random_state, class_weight=class_weight),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state,
                                               class_weight=class_weight),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state,
                                               class_weight=class_weight),
        "KNN": KNeighborsClassifier(),  # non supporta class_weight
        "LogReg": LogisticRegression(max_iter=1000, class_weight=class_weight),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                          random_state=random_state, verbosity=0, n_estimators=300)
    if _HAS_LGBM:
        # LightGBM accetta dict di pesi per classe
        models["LightGBM"] = LGBMClassifier(random_state=random_state, verbosity=-1,
                                            n_estimators=300, class_weight=class_weight,
                                            objective="multiclass", num_class=3)
    return models


def evaluate_models_cross_validation(x_train, y_train, n_splits=5, random_state=42):
    """
    Valuta i modelli in Stratified K-Fold (accuracy) e restituisce:
    - dict con accuracy media per modello,
    - best_estimator FITTATO su tutto il train,
    - best_model_name (stringa).
    """
    # class weights bilanciati
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}
    best_name, best_score = None, -np.inf

    for name, model in _candidate_models(random_state, class_weight=class_weight).items():
        scores = cross_val_score(model, x_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        mean_acc = float(np.mean(scores))
        results[name] = mean_acc
        if mean_acc > best_score:
            best_score, best_name = mean_acc, name

    best_estimator = _candidate_models(random_state, class_weight=class_weight)[best_name]
    best_estimator.fit(x_train, y_train)
    return results, best_estimator, best_name


# ---------------------------------------------------------------------
# Valutazione test
# ---------------------------------------------------------------------
def evaluate_on_test(model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Ritorna accuracy, classification_report (testo) e confusion_matrix (lista di liste)."""
    y_pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

# ---------------------------------------------------------------------
# Keras MLP (3 classi)
# ---------------------------------------------------------------------
def _build_keras(input_dim: int) -> Sequential:
    """MLP 3-classi (softmax). Si aspetta input già preprocessato."""
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(3, activation="softmax"),  # 3 classi fisse
    ])
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_keras_simple(x_train: pd.DataFrame, y_train: pd.Series,
                       x_val: pd.DataFrame, y_val: pd.Series,
                       max_epochs: int = 50, class_weight=None):
    """
    Allena un MLP 3-classi su dati GIÀ preprocessati (scalati).
    Restituisce (model, val_accuracy) e salva il best checkpoint.
    """
    Xtr = np.asarray(x_train, dtype=np.float32)
    Xva = np.asarray(x_val,   dtype=np.float32)
    ytr = np.asarray(y_train, dtype=np.int32)
    yva = np.asarray(y_val,   dtype=np.int32)

    model = _build_keras(input_dim=Xtr.shape[1])

    # Salva in un nome coerente con utils/Streamlit
    ckpt_path = ARTIFACTS_DIR / "best_keras_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_accuracy", save_best_only=True, save_weights_only=False),
    ]

    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=max_epochs, batch_size=64, verbose=0,
              callbacks=callbacks, class_weight=class_weight)

    _, val_acc = model.evaluate(Xva, yva, verbose=0)
    return model, float(val_acc)

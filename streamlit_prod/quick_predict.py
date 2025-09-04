# streamlit_prod/quick_predict.py
# -*- coding: utf-8 -*-
"""
Script basic per testare la predict del modello usando load_best_model() e preprocess_for_inference.

USO:
  # 1) nessun argomento -> usa un sample "peggiore"
  python streamlit_prod/quick_predict.py

  # 2) JSON inline (solo FEATURE, senza la label):
  python streamlit_prod/quick_predict.py '{"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":30,"Smoker":1,"Stroke":0,"HeartDiseaseorAttack":1,"PhysActivity":0,"Fruits":1,"Veggies":1,"HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,"GenHlth":5,"MentHlth":30,"PhysHlth":30,"DiffWalk":1,"Sex":0,"Age":9,"Education":5,"Income":1}'

  # 3) Riga CSV completa con questa esatta COLONNE ORDER (la PRIMA è la label e viene scartata):
  # Diabetes_012,HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income
  python streamlit_prod/quick_predict.py --row "2.0,1.0,1.0,1.0,30.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,5.0,30.0,30.0,1.0,0.0,9.0,5.0,1.0"
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# === Path repo per importare src.utils (stesso pattern dell'app) ===
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_best_model, preprocess_for_inference  # type: ignore

# === Ordine colonne della TUA riga CSV (label inclusa come prima colonna) ===
COL_ORDER = [
    "Diabetes_012","HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump",
    "AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk",
    "Sex","Age","Education","Income"
]

def features_from_csv_row(row_str: str) -> dict:
    """Parsa una riga CSV nel tuo ordine, scarta la label (prima colonna) e ritorna solo le feature."""
    vals = [v.strip() for v in row_str.split(",")]
    if len(vals) != len(COL_ORDER):
        raise ValueError(f"Attesi {len(COL_ORDER)} valori, trovati {len(vals)}")
    # converti a float
    try:
        vals = [float(v) for v in vals]
    except Exception as e:
        raise ValueError(f"Valori non numerici nella riga: {e}")

    # mappa FEATURE (salta la label alla posizione 0)
    feats = dict(zip(COL_ORDER[1:], vals[1:]))

    # cast: molte sono intere; lascia float per BMI, MentHlth, PhysHlth
    float_cols = {"BMI", "MentHlth", "PhysHlth"}
    for k in list(feats.keys()):
        feats[k] = float(feats[k]) if k in float_cols else int(feats[k])
    return feats

# === Sample "worst-case" (valori sfavorevoli) se non passi niente ===
WORST_SAMPLE = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 0, "BMI": 50, "Smoker": 1, "Stroke": 1,
    "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0, "Veggies": 0,
    "HvyAlcoholConsump": 1, "AnyHealthcare": 0, "NoDocbcCost": 1, "GenHlth": 5,
    "MentHlth": 30, "PhysHlth": 30, "DiffWalk": 1, "Sex": 1, "Age": 90,
    "Education": 1, "Income": 1,
}

def predict_with_proba(model, model_type: str, X: pd.DataFrame):
    """Ritorna (label_predetta, prob_label, vettore_proba) mappando correttamente via model.classes_."""
    if model_type == "sklearn":
        classes = np.asarray(getattr(model, "classes_", []))
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X)[0], dtype=float)
            cls_idx = int(np.argmax(p))
            cls_label = int(classes[cls_idx]) if classes.size else cls_idx
            return cls_label, float(p[cls_idx]), p
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function(X))
            if s.ndim == 1:  # binario
                p1 = 1.0 / (1.0 + np.exp(-float(s[0])))
                proba = np.array([1 - p1, p1], dtype=float)
                cls_idx = int(np.argmax(proba))
                cls_label = int(classes[cls_idx]) if classes.size else cls_idx
                return cls_label, float(proba[cls_idx]), proba
            z = s[0].astype(float); z -= np.max(z)
            p = np.exp(z) / np.exp(z).sum()
            cls_idx = int(np.argmax(p))
            cls_label = int(classes[cls_idx]) if classes.size else cls_idx
            return cls_label, float(p[cls_idx]), p
        # fallback
        pred = int(model.predict(X)[0])
        return pred, 0.5, None

    # Keras / TF (softmax attesa)
    p = np.asarray(model.predict(X, verbose=0)[0], dtype=float)
    cls_idx = int(np.argmax(p))
    return cls_idx, float(p[cls_idx]), p

def main():
    # --- INPUT ---
    if len(sys.argv) >= 2 and sys.argv[1] == "--row":
        if len(sys.argv) < 3:
            print("[ERRORE] Usa: quick_predict.py --row \"<riga CSV>\"")
            sys.exit(2)
        features = features_from_csv_row(sys.argv[2])
    elif len(sys.argv) >= 2:
        try:
            features = json.loads(sys.argv[1])
        except Exception as e:
            print(f"[ERRORE] JSON non valido: {e}")
            sys.exit(2)
    else:
        features = WORST_SAMPLE

    # --- MODELLO + PREPROCESS ---
    model, model_type, meta = load_best_model()
    rec = pd.DataFrame([features])
    X = preprocess_for_inference(rec, meta)
    # basic: forza numerici e rimpiazza eventuali NaN
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # --- PREDICT ---
    label_argmax, prob, proba_vec = predict_with_proba(model, model_type, X)
    try:
        raw_pred = int(model.predict(X)[0])
    except Exception:
        raw_pred = None

    # --- OUTPUT ---
    print("=== QUICK PREDICT ===")
    print(f"model_type: {model_type}")
    classes = getattr(model, "classes_", None)
    if classes is not None:
        print(f"classes_:  {list(map(int, classes))}")
    if proba_vec is not None:
        if classes is None:
            classes = list(range(len(proba_vec)))
        mapping = {int(classes[i]): float(proba_vec[i]) for i in range(len(proba_vec))}
        print(f"predict_proba: {mapping}")
    print(f"argmax(prob): {label_argmax}")
    print(f"model.predict: {raw_pred}")
    if raw_pred is not None and raw_pred != label_argmax:
        print("⚠️  Attenzione: model.predict != argmax(prob). Verifica preprocess/threshold.")

if __name__ == "__main__":
    main()

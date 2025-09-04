# streamlit_prod/quick_predict.py
# -*- coding: utf-8 -*-
"""
Script basic per testare la predict del modello con un caso "peggiore".
- Carica modello e preprocess via load_best_model / preprocess_for_inference.
- Di default usa un sample con tutti i valori sfavorevoli (worst-case).
- Puoi passare un JSON di feature come primo argomento per sovrascrivere.

USO:
  python streamlit_prod/quick_predict.py
  python streamlit_prod/quick_predict.py '{"HighBP":1,"HighChol":1,...}'
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

# === SAMPLE "PEGGIORE" (tutti i valori più sfavorevoli) ===
# Note: per BRFSS-like features:
# - CholCheck: 0 = NON ha controllato (peggio)
# - AnyHealthcare: 0 = NO accesso (peggio)
# - NoDocbcCost: 1 = sì, non è andato dal medico per i costi (peggio)
# - GenHlth: 5 = pessima
# - MentHlth/PhysHlth: 30 = max
# - PhysActivity/Fruits/Veggies: 0 = NO (peggio)
# - HvyAlcoholConsump/Smoker/HighBP/HighChol/Stroke/HeartDiseaseorAttack/DiffWalk: 1 = sì (peggio)
# - BMI: molto alto (es. 50)
# - Age: alto (es. 90)
# - Education/Income: valori bassi = peggio
WORST_SAMPLE = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 0,
    "BMI": 50,
    "Smoker": 1,
    "Stroke": 1,
    "HeartDiseaseorAttack": 1,
    "PhysActivity": 0,
    "Fruits": 0,
    "Veggies": 0,
    "HvyAlcoholConsump": 1,
    "AnyHealthcare": 0,
    "NoDocbcCost": 1,
    "GenHlth": 5,
    "MentHlth": 30,
    "PhysHlth": 30,
    "DiffWalk": 1,
    "Sex": 1,
    "Age": 90,
    "Education": 1,
    "Income": 1,
}

def _predict_with_proba(model, model_type: str, X: pd.DataFrame):
    """Ritorna (label, prob_label, proba_vector). Usa model.classes_ per il mapping corretto."""
    if model_type == "sklearn":
        classes = getattr(model, "classes_", None)
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X)[0], dtype=float)
            cls_idx = int(np.argmax(p))
            cls_label = int(classes[cls_idx]) if classes is not None else cls_idx
            return cls_label, float(p[cls_idx]), p
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function(X))
            if s.ndim == 1:  # caso binario (non atteso se il tuo è 0/1/2, ma lo gestiamo)
                p1 = 1.0 / (1.0 + np.exp(-float(s[0])))
                cls_label = 1 if p1 >= 0.5 else 0
                proba = np.array([1 - p1, p1], dtype=float)
                return int(cls_label), float(proba[cls_label]), proba
            # multi-classe -> softmax
            z = s[0].astype(float)
            z -= np.max(z)
            p = np.exp(z) / np.exp(z).sum()
            cls_idx = int(np.argmax(p))
            cls_label = int(classes[cls_idx]) if classes is not None else cls_idx
            return cls_label, float(p[cls_idx]), p
        # fallback
        pred = int(model.predict(X)[0])
        return pred, 0.5, None

    # Keras / TF: assumiamo softmax
    p = np.asarray(model.predict(X, verbose=0)[0], dtype=float)
    cls_idx = int(np.argmax(p))
    return cls_idx, float(p[cls_idx]), p

def main():
    # 1) Carica eventuale JSON passato, altrimenti usa WORST_SAMPLE
    if len(sys.argv) > 1:
        try:
            features = json.loads(sys.argv[1])
        except Exception as e:
            print(f"[ERRORE] JSON non valido: {e}")
            sys.exit(2)
    else:
        features = WORST_SAMPLE

    # 2) Carica modello e meta
    model, model_type, meta = load_best_model()

    # 3) Prepara record una riga e applica lo stesso preprocess dell'app
    rec = pd.DataFrame([features])
    X = preprocess_for_inference(rec, meta)

    # Basic: assicura numerici e sostituisci NaN con 0 (senza entrare in allineamenti complessi)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4) Predict
    label, prob, proba_vec = _predict_with_proba(model, model_type, X)

    # 5) Output
    print("=== QUICK PREDICT (worst-case) ===")
    print(f"model_type: {model_type}")
    classes = getattr(model, "classes_", None)
    if classes is not None:
        print(f"classes_: {list(map(int, classes))}")
    print(f"predicted_label: {label}")
    print(f"predicted_prob:  {prob:.4f}")
    if proba_vec is not None:
        if classes is None:
            classes = list(range(len(proba_vec)))
        mapping = {int(classes[i]): float(proba_vec[i]) for i in range(len(proba_vec))}
        print(f"predict_proba:  {mapping}")

if __name__ == "__main__":
    main()

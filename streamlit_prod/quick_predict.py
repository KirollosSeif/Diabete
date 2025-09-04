# -*- coding: utf-8 -*-
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_best_model, preprocess_for_inference  # type: ignore

# Caso "peggiore" per stress-test
WORST_SAMPLE = {
    "HighBP":1, "HighChol":1, "CholCheck":0, "BMI":50, "Smoker":1, "Stroke":1,
    "HeartDiseaseorAttack":1, "PhysActivity":0, "Fruits":0, "Veggies":0,
    "HvyAlcoholConsump":1, "AnyHealthcare":0, "NoDocbcCost":1, "GenHlth":5,
    "MentHlth":30, "PhysHlth":30, "DiffWalk":1, "Sex":1, "Age":90, "Education":1, "Income":1,
}

def main():
    # JSON opzionale da argv, altrimenti worst-case
    if len(sys.argv) > 1:
        try:
            features = json.loads(sys.argv[1])
        except Exception as e:
            print(f"[ERRORE] JSON non valido: {e}"); sys.exit(2)
    else:
        features = WORST_SAMPLE

    model, model_type, meta = load_best_model()
    rec = pd.DataFrame([features])
    X = preprocess_for_inference(rec, meta)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Probabilità
    proba_vec = None
    if model_type == "sklearn" and hasattr(model, "predict_proba"):
        proba_vec = np.asarray(model.predict_proba(X)[0], dtype=float)
        classes = np.asarray(getattr(model, "classes_", list(range(len(proba_vec)))))
        cls_idx = int(np.argmax(proba_vec))
        label_argmax = int(classes[cls_idx])
    else:
        # Keras/TF: assumiamo softmax
        p = np.asarray(model.predict(X, verbose=0)[0], dtype=float)
        classes = np.arange(len(p))
        proba_vec = p
        label_argmax = int(classes[int(np.argmax(p))])

    # Predizione "ufficiale" del modello
    try:
        raw_pred = int(model.predict(X)[0])
    except Exception:
        raw_pred = None

    print("=== QUICK PREDICT ===")
    print(f"model_type: {model_type}")
    print(f"classes_:  {list(map(int, classes))}")
    if proba_vec is not None:
        mapping = {int(classes[i]): float(proba_vec[i]) for i in range(len(proba_vec))}
        print(f"predict_proba: {mapping}")
    print(f"argmax(prob): {label_argmax}")
    print(f"model.predict: {raw_pred}")

    if raw_pred is not None and raw_pred != label_argmax:
        print("⚠️ ATTENZIONE: model.predict != argmax(prob). Controlla preprocessing/thresholding.")

if __name__ == "__main__":
    main()

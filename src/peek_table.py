# src/diag_refresh.py
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.data_preprocessing import preprocess_data  # stessa funzione usata dal refresh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
CSV_PATH = PROJECT_ROOT / "data" / "diabete_data.csv"  # stesso path del refresh

def main():
    load_dotenv(ENV_PATH)
    print(f".env: {ENV_PATH.exists()} | CSV path: {CSV_PATH}")

    if not CSV_PATH.exists():
        print(f"ERRORE: CSV non trovato a {CSV_PATH}")
        return

    df_raw = pd.read_csv(CSV_PATH)
    print(f"CSV letto: shape raw = {df_raw.shape}")
    print("Null per colonna (prime 15):")
    print(df_raw.isna().sum().sort_values(ascending=False).head(15))

    df_clean = preprocess_data(df_raw)
    print(f"shape dopo preprocess = {df_clean.shape}")

    # Se df_clean è vuoto, capiamo perché: quante righe perse per NaN?
    lost = len(df_raw) - len(df_clean)
    print(f"Righe perse in preprocess (dropna): {lost}")

    # Quante colonne “continue” (nunique>13) usate nello scaling?
    import numpy as np
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cont_cols = [c for c in num_cols if df_clean[c].nunique() > 13]
    print(f"Colonne continue scalate: {len(cont_cols)} -> {cont_cols[:12]}{'...' if len(cont_cols)>12 else ''}")

if __name__ == "__main__":
    main()

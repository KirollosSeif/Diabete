# src/push_gold_to_db.py
# Scopo: leggere i feedback (CSV), tenere SOLO i casi confermati corretti
#        e ACCODARLI nel DB (tabella 'feedback_gold').
# Richiede: variabile d'ambiente DATABASE_URL e driver del DB installato.

from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine

# --- percorsi base ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# sorgente UNICA dei feedback
FEEDBACK_NEW = DATA_DIR / "feedback_test.csv"

# tabella di destinazione nel DB
TABLE_NAME = "feedback_gold"

# colonne che voglio mandare al DB (tutte le feature + target)
FEATURE_COLS = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
    "DiffWalk","Sex","Age","Education","Income","Diabetes_012"
]

def get_engine():
    """Crea l'engine SQLAlchemy dal DATABASE_URL (es. MySQL o Postgres)."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL non impostata. Esempi:\n"
                           "  MySQL:     mysql+pymysql://user:pass@host:3306/dbname\n"
                           "  Postgres:  postgresql+psycopg2://user:pass@host:5432/dbname")
    return create_engine(url, pool_pre_ping=True)

def load_feedback_csv() -> pd.DataFrame:
    """Carica il CSV dei feedback (usa solo feedback_test.csv)."""
    if not FEEDBACK_NEW.exists():
        raise FileNotFoundError("Non trovo data/feedback_test.csv")
    df = pd.read_csv(FEEDBACK_NEW)
    if df.empty:
        raise ValueError(f"{FEEDBACK_NEW.name} è vuoto.")
    return df

def build_gold(df_fb: pd.DataFrame) -> pd.DataFrame:
    """Seleziona solo i feedback corretti e prepara le colonne feature + target."""
    if "Predicted" not in df_fb.columns or "Diabetes_012" not in df_fb.columns:
        raise KeyError("Mancano le colonne 'Predicted' e/o 'Diabetes_012' nel feedback CSV.")

    # prendo solo i casi confermati corretti
    df_ok = df_fb[df_fb["Predicted"] == df_fb["Diabetes_012"]].copy()
    if df_ok.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    # assicuro che tutte le colonne che mi servono esistano:
    # se qualcuna manca, la creo vuota (NaN) per mantenere lo schema fisso
    for col in FEATURE_COLS:
        if col not in df_ok.columns:
            df_ok[col] = pd.NA

    # mantengo solo le colonne d'interesse e nell'ordine desiderato
    gold = df_ok[FEATURE_COLS].copy()

    # converto a numerico dove possibile
    for c in FEATURE_COLS:
        gold[c] = pd.to_numeric(gold[c], errors="ignore")

    # dedup all'interno del batch corrente
    gold = gold.drop_duplicates().reset_index(drop=True)
    return gold

def append_dataframe_to_db(df: pd.DataFrame, table_name: str, engine):
    """Accoda righe alla tabella; crea la tabella se non esiste."""
    if df.empty:
        print("Nessun record da inserire (nessun feedback confermato).")
        return 0
    df.to_sql(name=table_name, con=engine, if_exists="append", index=False)
    return len(df)

def main():
    try:
        print("[1/3] Carico feedback (feedback_test.csv)...")
        df_fb = load_feedback_csv()

        print("[2/3] Preparo righe GOLD (solo confermati corretti)...")
        gold = build_gold(df_fb)
        print(f"  -> Righe pronte da inserire: {len(gold)}")

        print("[3/3] Connessione DB e append...")
        engine = get_engine()
        inserted = append_dataframe_to_db(gold, TABLE_NAME, engine)
        print(f"FATTO: inserite {inserted} righe in tabella '{TABLE_NAME}'.")
        if inserted == 0:
            print("Nessun inserimento perché non c'erano casi confermati corretti.")
    except Exception as e:
        print(f"[ERRORE] {e}")

if __name__ == "__main__":
    main()

# Scopo: leggere i feedback (CSV), tenere SOLO i casi confermati corretti
#        e ACCODARLI nella tabella di training originaria: 'diabetes_data'.
# Connessione: usa DATABASE_URL se presente, altrimenti SQL_* dal .env.

from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

# --- percorsi base ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# sorgente feedback
FEEDBACK_NEW = DATA_DIR / "feedback_test.csv"

# tabella di destinazione nel DB (tabella ORIGINARIA di studio)
DEST_TABLE = "diabetes_data"

# set base di colonne dai feedback (features + target)
FEATURE_COLS = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
    "DiffWalk","Sex","Age","Education","Income","Diabetes_012"
]

def _build_url_from_env() -> str | None:
    """Fallback: costruisce un URL SQLAlchemy dai singoli env SQL_*."""
    user = os.getenv("SQL_USERNAME")
    pwd  = os.getenv("SQL_PASSWORD")
    host = os.getenv("SQL_HOST")
    db   = os.getenv("SQL_DATABASE")
    port = os.getenv("SQL_PORT", "3306")
    driver = os.getenv("SQL_DRIVER", "mysql+pymysql")  # es: "postgresql+psycopg2"
    if user and pwd and host and db:
        return f"{driver}://{user}:{pwd}@{host}:{port}/{db}"
    return None

def get_engine():
    """Crea l'engine SQLAlchemy da DATABASE_URL o da variabili SQL_*."""
    load_dotenv(PROJECT_ROOT / ".env")  # carica .env se presente
    url = os.getenv("DATABASE_URL") or _build_url_from_env()
    if not url:
        raise RuntimeError(
            "Connessione DB non configurata.\n"
            "- Imposta DATABASE_URL (consigliato), ad es:\n"
            "    mysql+pymysql://user:pass@host:3306/dbname\n"
            "    postgresql+psycopg2://user:pass@host:5432/dbname\n"
            "- Oppure le variabili: SQL_USERNAME, SQL_PASSWORD, SQL_HOST, SQL_PORT, SQL_DATABASE"
        )
    return create_engine(url, pool_pre_ping=True)

def load_feedback_csv() -> pd.DataFrame:
    """Carica il CSV dei feedback (usa solo feedback_test.csv)."""
    if not FEEDBACK_NEW.exists():
        raise FileNotFoundError(f"Non trovo {FEEDBACK_NEW}")
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

    # assicuro tutte le colonne necessarie
    for col in FEATURE_COLS:
        if col not in df_ok.columns:
            df_ok[col] = pd.NA

    gold = df_ok[FEATURE_COLS].copy()

    # converto numerici dove possibile (evita errori di tipo lato DB)
    for c in FEATURE_COLS:
        gold[c] = pd.to_numeric(gold[c], errors="ignore")

    # dedup nel batch
    gold = gold.drop_duplicates().reset_index(drop=True)
    return gold

def get_table_columns(engine, table_name: str) -> list[str]:
    """Legge lo schema della tabella di destinazione e ritorna l'ordine colonne."""
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns(table_name)]
    if not cols:
        raise RuntimeError(
            f"La tabella '{table_name}' non esiste oppure non ha colonne leggibili.\n"
            f"Creala o verifica i permessi dell'utente DB."
        )
    return cols

def append_dataframe_to_db(df: pd.DataFrame, table_name: str, engine):
    """Accoda righe alla tabella; riallinea le colonne alla tabella esistente."""
    if df.empty:
        print("Nessun record da inserire (nessun feedback confermato).")
        return 0

    # Allinea al set colonne reale della tabella (ordine e nomi)
    target_cols = get_table_columns(engine, table_name)
    # Mantieni solo le colonne esistenti e crea le mancanti come NaN
    aligned = df.reindex(columns=target_cols, fill_value=pd.NA)

    # Scrive in append (usa transazione implicita)
    aligned.to_sql(name=table_name, con=engine, if_exists="append", index=False)
    return len(aligned)

def main():
    try:
        print(f"[1/3] Carico feedback: {FEEDBACK_NEW.name} ...")
        df_fb = load_feedback_csv()

        print("[2/3] Preparo righe GOLD (solo confermati corretti)...")
        gold = build_gold(df_fb)
        print(f"  -> Righe pronte da inserire: {len(gold)}")

        print(f"[3/3] Connessione DB e append su '{DEST_TABLE}' ...")
        engine = get_engine()
        inserted = append_dataframe_to_db(gold, DEST_TABLE, engine)
        print(f"FATTO: inserite {inserted} righe in tabella '{DEST_TABLE}'.")
        if inserted == 0:
            print("Nessun inserimento perché non c'erano casi confermati corretti.")
    except Exception as e:
        print(f"[ERRORE] {e}")

if __name__ == "__main__":
    main()

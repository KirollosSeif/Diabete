
# Legge feedback_test.csv, tiene solo i casi confermati corretti,
# e li accoda a 'diabetes_data' evitando duplicati con fingerprint
# calcolato in modo CANONICO su features+target.

from pathlib import Path
import os
import hashlib
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

FEEDBACK_NEW = DATA_DIR / "feedback_test.csv"
DEST_TABLE = "diabetes_data"

FEATURE_COLS = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
    "DiffWalk","Sex","Age","Education","Income","Diabetes_012"
]
INT_COLS = [c for c in FEATURE_COLS if c != "BMI"]   # tutto int tranne BMI
FLOAT_COLS = ["BMI"]

def _build_url_from_env() -> str | None:
    user = os.getenv("SQL_USERNAME"); pwd = os.getenv("SQL_PASSWORD")
    host = os.getenv("SQL_HOST");     db  = os.getenv("SQL_DATABASE")
    port = os.getenv("SQL_PORT", "3306")
    driver = os.getenv("SQL_DRIVER", "mysql+pymysql")
    if user and pwd and host and db:
        return f"{driver}://{user}:{pwd}@{host}:{port}/{db}"
    return None

def get_engine():
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("DATABASE_URL") or _build_url_from_env()
    if not url:
        raise RuntimeError("Config DB mancante: setta DATABASE_URL o SQL_* nel .env")
    return create_engine(url, pool_pre_ping=True)

def load_feedback_csv() -> pd.DataFrame:
    if not FEEDBACK_NEW.exists():
        raise FileNotFoundError(f"Non trovo {FEEDBACK_NEW}")
    df = pd.read_csv(FEEDBACK_NEW)
    if df.empty:
        raise ValueError(f"{FEEDBACK_NEW.name} è vuoto.")
    return df

def build_gold(df_fb: pd.DataFrame) -> pd.DataFrame:
    if "Predicted" not in df_fb.columns or "Diabetes_012" not in df_fb.columns:
        raise KeyError("Mancano 'Predicted' e/o 'Diabetes_012' nel CSV.")
    df_ok = df_fb[df_fb["Predicted"] == df_fb["Diabetes_012"]].copy()
    if df_ok.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    for col in FEATURE_COLS:
        if col not in df_ok.columns:
            df_ok[col] = pd.NA

    gold = df_ok[FEATURE_COLS].copy()
    # Canonicalizzazione: numerici coerenti
    for c in INT_COLS:
        gold[c] = pd.to_numeric(gold[c], errors="coerce").round(0).astype("Int64")
    for c in FLOAT_COLS:
        gold[c] = pd.to_numeric(gold[c], errors="coerce").round(2)
    gold = gold.drop_duplicates().reset_index(drop=True)
    return gold

def get_db_feature_cols(engine) -> list[str]:
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns(DEST_TABLE)]
    if not cols:
        raise RuntimeError(f"Tabella '{DEST_TABLE}' inesistente o non leggibile.")
    return cols

def canonicalize_for_hash(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    # Applica stesse regole a df per le colonne indicate
    for c in cols:
        if c in INT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
        elif c in FLOAT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
        else:
            # fallback: numerico se possibile
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def make_fingerprint(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    # converte in stringa in modo CONSISTENTE:
    # - Int64: "1" (NA -> "")
    # - float: 2 decimali (NA -> "")
    parts = []
    for c in cols:
        s = df[c]
        if str(s.dtype).startswith("Int"):
            s = s.astype("Int64").astype("string").replace("<NA>", "")
        elif pd.api.types.is_float_dtype(s):
            s = s.map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
        else:
            # altri tipi -> stringa, NA -> ""
            s = s.astype("string").fillna("")
        parts.append(s)
    key = pd.concat(parts, axis=1).agg("|".join, axis=1).str.encode("utf-8")
    return key.map(lambda b: hashlib.md5(b).hexdigest())

def load_existing_fingerprints(engine, cols_for_hash: list[str]) -> set[str]:
    # Leggo **solo** le colonne presenti nel DB tra quelle utili all'hash
    use_cols = cols_for_hash
    sql = f'SELECT {", ".join(use_cols)} FROM {DEST_TABLE}'
    df_exist = pd.read_sql(sql, con=engine)
    if df_exist.empty:
        return set()
    df_exist = canonicalize_for_hash(df_exist, use_cols)
    return set(make_fingerprint(df_exist, use_cols))

def append_dataframe_to_db(df: pd.DataFrame, engine):
    if df.empty:
        return 0
    # riallinea colonne alla tabella (ordine/nomi)
    target_cols = get_db_feature_cols(engine)
    aligned = df.reindex(columns=target_cols, fill_value=pd.NA)
    aligned.to_sql(name=DEST_TABLE, con=engine, if_exists="append", index=False)
    return len(aligned)

def main():
    try:
        print(f"[1/5] Carico feedback: {FEEDBACK_NEW.name} ...")
        df_fb = load_feedback_csv()

        print("[2/5] Preparo GOLD (solo confermati corretti) + canonicalizzo...")
        gold = build_gold(df_fb)
        print(f"   -> batch rows: {len(gold)}")

        print("[3/5] Connessione DB e definizione colonne per fingerprint...")
        engine = get_engine()
        db_cols = set(get_db_feature_cols(engine))
        # uso intersezione: così l'hash usa solo colonne che ESISTONO nel DB
        cols_for_hash = [c for c in FEATURE_COLS if c in db_cols]
        if not cols_for_hash:
            raise RuntimeError("Nel DB non c'è nessuna delle FEATURE_COLS, impossibile deduplicare.")
        gold_hash_df = canonicalize_for_hash(gold, cols_for_hash)

        print(f"[4/5] Carico fingerprint esistenti su {len(cols_for_hash)} colonne...")
        existing_fp = load_existing_fingerprints(engine, cols_for_hash)
        gold_hash = make_fingerprint(gold_hash_df, cols_for_hash)
        gold_new = gold.loc[~gold_hash.isin(existing_fp)].copy()
        print(f"   -> righe nuove da inserire: {len(gold_new)}")

        print("[5/5] Append su DB...")
        inserted = append_dataframe_to_db(gold_new, engine)
        print(f"FATTO: inserite {inserted} righe nuove in '{DEST_TABLE}'.")
        if inserted == 0:
            print("Nessun inserimento: tutti i record erano già presenti.")
    except Exception as e:
        print(f"[ERRORE] {e}")

if __name__ == "__main__":
    main()

# src/refresh_db_table.py
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import text

from src.data_preprocessing import (
    create_db_engine,
    test_connection,
    load_data_from_csv,
    preprocess_data,
    append_dataframe_to_db,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
CSV_PATH = PROJECT_ROOT / "data" / "diabete_data.csv"

REQUIRED_ENV = ["SQL_USERNAME", "SQL_PASSWORD", "SQL_HOST", "SQL_DATABASE"]

def table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE() AND table_name = :t
                """),
                {"t": table_name},
            ).scalar()
        )

def safe_alter_float_columns(engine, table_name: str, cols: list[str]):
    """Prova a modificare i tipi delle colonne indicate in DOUBLE.
       Se fallisce (colonna mancante o permessi), prosegue senza bloccare il refresh."""
    if not cols:
        return
    clauses = ", ".join([f"MODIFY `{c}` DOUBLE NULL" for c in cols])
    sql = f"ALTER TABLE `{table_name}` {clauses}"
    with engine.begin() as conn:
        try:
            conn.execute(text(sql))
            print(f"Schema adeguato su `{table_name}` -> {', '.join(cols)} a DOUBLE.")
        except Exception as e:
            print(f"ALTER TABLE non eseguito (non necessario o permessi insufficienti): {e}")

def main():
    # 0) ENV
    load_dotenv(ENV_PATH)
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Variabili mancanti in .env: {missing}. Percorso .env: {ENV_PATH}")

    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    host     = os.getenv("SQL_HOST")
    database = os.getenv("SQL_DATABASE")
    port     = int(os.getenv("SQL_PORT", 3306))
    table_name = os.getenv("DB_TABLE", "diabetes_data")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV non trovato: {CSV_PATH}")

    # 1) Connessione
    engine = create_db_engine(username, password, host, port, database)
    test_connection(engine)

    # 2) Carica + preprocess
    df_raw = load_data_from_csv(str(CSV_PATH))
    print(f"CSV letto: shape raw = {df_raw.shape}")

    df_clean = preprocess_data(df_raw)
    print(f"shape dopo preprocess = {df_clean.shape}")

    if len(df_clean) == 0:
        raise RuntimeError("DataFrame dopo preprocess è vuoto: interrompo per evitare di svuotare la tabella.")

    # 3) Se la tabella esiste, adegua tipi e svuota con DELETE (no TRUNCATE)
    exists = table_exists(engine, table_name)
    if exists:
        # colonne che diventano float dopo lo scaling
        float_cols = ["BMI", "MentHlth", "PhysHlth"]
        safe_alter_float_columns(engine, table_name, float_cols)

        # Svuota
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM `{table_name}`"))
            print(f"Tabella `{table_name}` svuotata (DELETE).")
    else:
        print(f"Tabella `{table_name}` non esiste: verrà creata automaticamente da to_sql(append).")

    # 4) Inserisci a chunk
    print(f"Inserimento in `{table_name}`: {len(df_clean)} righe x {df_clean.shape[1]} colonne...")
    append_dataframe_to_db(df_clean, table_name, engine, chunksize=2000)

    # 5) Verifica righe finali
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()
        print(f"✅ Refresh completato. Righe in `{table_name}`: {n}")

if __name__ == "__main__":
    main()

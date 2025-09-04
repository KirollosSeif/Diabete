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
    import_dataframe_to_db,  # useremo REPLACE sulla tabella nuova
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

def count_rows(engine, table_name: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar())

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

    # 3) Strategia preferita: NUOVA TABELLA (evita lock sulla tabella corrente)
    target_table = f"{table_name}_refreshed"
    print(f"Provo a creare/aggiornare la NUOVA tabella '{target_table}' (replace).")

    created_new_table = False
    try:
        # REPLACE sulla NUOVA tabella: se esiste la ri-crea, altrimenti la crea.
        import_dataframe_to_db(df_clean, target_table, engine)
        n_new = count_rows(engine, target_table)
        print(f"✅ Nuova tabella '{target_table}' pronta con {n_new} righe.")
        created_new_table = True
    except Exception as e:
        print("❌ Creazione/replace della nuova tabella fallita. Motivo:")
        import traceback; traceback.print_exc()
        created_new_table = False

    if created_new_table:
        print("\n➡️ Aggiorna ora il tuo .env per usare la tabella nuova:")
        print(f"DB_TABLE={target_table}")
        print("Poi rilancia i tuoi script (peek_table/test) per leggere dalla nuova tabella.")
        return

    # 4) Fallback: svuota con DELETE la tabella corrente e APPEND a chunk
    #    (niente ALTER: evitiamo lock; se il tipo non è compatibile, l'errore sarà chiaro)
    print("\n➡️ Fallback: uso la tabella esistente (DELETE + APPEND).")

    exists = table_exists(engine, table_name)
    if not exists:
        print(f"Tabella `{table_name}` non esiste: verrà creata automaticamente da to_sql(append).")

    else:
        print(f"Svuoto la tabella esistente `{table_name}` con DELETE (no TRUNCATE).")
        try:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM `{table_name}`"))
            print(f"Tabella `{table_name}` svuotata.")
        except Exception as e:
            print("❌ DELETE fallito sulla tabella esistente. Motivo:")
            import traceback; traceback.print_exc()
            raise

    print(f"Inserimento in `{table_name}`: {len(df_clean)} righe x {df_clean.shape[1]} colonne...")
    try:
        append_dataframe_to_db(df_clean, table_name, engine, chunksize=2000)
        n = count_rows(engine, table_name)
        print(f"✅ Refresh completato (fallback). Righe in `{table_name}`: {n}")
    except Exception:
        print("❌ Inserimento fallito anche in fallback. Dettagli:")
        import traceback; traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

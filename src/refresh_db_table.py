import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from src.data_preprocessing import (
    create_db_engine,
    test_connection,
    load_data_from_csv,
    preprocess_data,       # pulizia + scaling
    append_dataframe_to_db,
    truncate_table        # <-- usa quella già nel modulo
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
CSV_PATH = PROJECT_ROOT / "data" / "diabete_data.csv"

def main():
    load_dotenv(ENV_PATH)

    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    host     = os.getenv("SQL_HOST")
    database = os.getenv("SQL_DATABASE")
    port     = int(os.getenv("SQL_PORT", 3306))
    table_name = os.getenv("DB_TABLE", "diabetes_data")  # <-- dopo load_dotenv

    engine = create_db_engine(username, password, host, port, database)
    test_connection(engine)

    df_raw = load_data_from_csv(str(CSV_PATH))
    df_clean = preprocess_data(df_raw)

    truncate_table(engine, table_name)
    append_dataframe_to_db(df_clean, table_name, engine)

    print(f"Ricaricata '{table_name}': {len(df_clean)} righe, {len(df_clean.columns)} colonne.")

if __name__ == "__main__":
    main()

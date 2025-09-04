import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.data_preprocessing import create_db_engine, test_connection

# Path .env dalla root del progetto (come fai negli altri script)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

def main():
    # 0) ENV
    load_dotenv(ENV_PATH)
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    host     = os.getenv("SQL_HOST")
    database = os.getenv("SQL_DATABASE")
    port     = int(os.getenv("SQL_PORT", 3306))
    table    = os.getenv("DB_TABLE", "diabetes_data")  # default

    # 1) Connessione
    engine = create_db_engine(username, password, host, port, database)
    test_connection(engine)

    # 2) Conta righe + anteprima
    n_rows = int(pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table}", con=engine)["n"][0])
    head   = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", con=engine)
    n_cols = head.shape[1]

    print(f"Tabella: {table}")
    print(f"Shape: ({n_rows}, {n_cols})")
    print(head)

if __name__ == "__main__":
    main()
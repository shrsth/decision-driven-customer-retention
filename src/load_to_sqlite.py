"""Load the cleaned, economics-enriched customer table into SQLite."""

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH


def load_to_sqlite(customers_df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        customers_df.to_sql("customers", conn, if_exists="replace", index=False)
    print(f"[sqlite] Loaded {len(customers_df)} customers into {db_path}")


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    from src.economics import add_economic_fields
    from src.ingest import clean_telco_data

    raw = pd.read_csv(RAW_DATA_PATH)
    load_to_sqlite(add_economic_fields(clean_telco_data(raw)))

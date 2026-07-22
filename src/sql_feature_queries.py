"""SQL access layer for the customers table in SQLite."""

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH


def load_customers_from_sql(db_path: Path = DB_PATH) -> pd.DataFrame:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path} — run `python -m src.pipeline` first."
        )
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM customers", conn)


def churn_summary_by_segment(db_path: Path = DB_PATH) -> pd.DataFrame:
    """In-database churn summary by contract and internet service."""
    query = """
        SELECT
            Contract,
            InternetService,
            COUNT(*)                    AS customers,
            ROUND(AVG(churned), 3)      AS churn_rate,
            ROUND(AVG(MRR), 2)          AS avg_mrr,
            ROUND(AVG(CLV), 2)          AS avg_clv
        FROM customers
        GROUP BY Contract, InternetService
        ORDER BY churn_rate DESC
    """
    with sqlite3.connect(Path(db_path)) as conn:
        return pd.read_sql(query, conn)

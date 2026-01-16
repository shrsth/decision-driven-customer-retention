import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "db" / "retention.db"


def build_features_from_sql():
    conn = sqlite3.connect(DB_PATH)

    # ---------------------------
    # Customer-level data
    # ---------------------------
    customers_query = """
    SELECT
        customer_id,
        CLV,
        MRR,
        retention_cost,
        churned
    FROM customers
    """
    customers_df = pd.read_sql_query(customers_query, conn)

    # ---------------------------
    # Weekly behavior data
    # ---------------------------
    behavior_query = """
    SELECT
        customer_id,
        week,
        weekly_usage,
        support_tickets
    FROM behavior_events
    """
    behavior_df = pd.read_sql_query(behavior_query, conn)

    conn.close()
    return customers_df, behavior_df

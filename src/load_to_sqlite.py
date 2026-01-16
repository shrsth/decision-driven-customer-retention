import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "db" / "retention.db"

customers = pd.read_csv(BASE_DIR / "data" / "raw" / "customers.csv")
behavior = pd.read_csv(BASE_DIR / "data" / "raw" / "behavior.csv")

conn = sqlite3.connect(DB_PATH)

customers.to_sql("customers", conn, if_exists="replace", index=False)
behavior.to_sql("behavior_events", conn, if_exists="replace", index=False)

conn.close()

print("✅ Data loaded into SQLite successfully")

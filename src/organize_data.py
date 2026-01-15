import os
import shutil

RAW_DATA_DIR = "data/raw"
FILES_TO_MOVE = ["customers.csv", "behavior.csv"]

os.makedirs(RAW_DATA_DIR, exist_ok=True)

for file in FILES_TO_MOVE:
    if os.path.exists(file):
        shutil.move(file, os.path.join(RAW_DATA_DIR, file))
        print(f"Moved {file} to {RAW_DATA_DIR}")
    else:
        print(f"{file} not found")

print("Data organization complete.")

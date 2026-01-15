import pandas as pd
from features.feature_builder import build_feature_table

customers = pd.read_csv("data/raw/customers.csv")
behavior = pd.read_csv("data/raw/behavior.csv")

features = build_feature_table(customers, behavior)

print(features.head())
print(features.columns)

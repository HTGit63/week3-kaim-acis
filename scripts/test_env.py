# scripts/test_env.py
import pandas as pd
import yaml
import sklearn
df = pd.read_parquet("data/processed/features.parquet")
print("Loaded features with shape:", df.shape)
with open("params.yaml") as f:
    params = yaml.safe_load(f)
print("Params loaded:", params)

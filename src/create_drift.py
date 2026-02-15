import pandas as pd
import yaml
import numpy as np
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

processed_dir = config["data"]["processed_dir"]
test_file = config["versions"]["test"]

test_path = os.path.join(processed_dir, test_file)

df = pd.read_csv(test_path)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
target_col = config["data"]["target_column"]
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
df[numeric_cols[0]] += 50
production_dir = config["data"]["production_dir"]
drift_file = config["versions"]["drift"]
drift_path = os.path.join(production_dir, drift_file)
df.to_csv(drift_path, index=False)

print("Drifted data created")

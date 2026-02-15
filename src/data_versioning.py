import pandas as pd
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_path = config["data"]["raw_path"]
processed_dir = config["data"]["processed_dir"]
raw_version = config["versions"]["raw"]
cleaned_version = config["versions"]["cleaned"]

df = pd.read_csv(raw_path)
raw_save_path = os.path.join(processed_dir, raw_version)
df.to_csv(raw_save_path, index=False)

df = df.drop(columns=["UDI", "Product ID"])
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

cleaned_save_path = os.path.join(processed_dir, cleaned_version)
df.to_csv(cleaned_save_path, index=False)

print("Data versioning done")

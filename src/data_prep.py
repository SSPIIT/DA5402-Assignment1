import pandas as pd
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_path = config["data"]["raw_path"]
processed_dir = config["data"]["processed_dir"]
version = config["data"]["version"]
train_file = config["data"]["train_file"]
test_file = config["data"]["test_file"]

df = pd.read_csv(raw_path)

df = df.drop(columns=["UDI", "Product ID"])

df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

train_df = df.iloc[:7000]
test_df = df.iloc[7000:]

train_path = os.path.join(processed_dir, train_file)
test_path = os.path.join(processed_dir, test_file)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Data preparation done")

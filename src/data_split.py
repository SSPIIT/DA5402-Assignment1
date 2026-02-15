import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

processed_dir = config["data"]["processed_dir"]
cleaned_version = config["versions"]["cleaned"]
train_version = config["versions"]["train"]
test_version = config["versions"]["test"]
train_size = config["split"]["train_size"]
random_state = config["split"]["random_state"]

cleaned_path = os.path.join(processed_dir, cleaned_version)
df = pd.read_csv(cleaned_path)

n_train = int(len(df) * train_size)

train_df = df.iloc[:n_train]
test_df = df.iloc[n_train:]
train_df.to_csv(os.path.join(processed_dir, train_version), index=False)
test_df.to_csv(os.path.join(processed_dir, test_version), index=False)

print("Train/Test split done")

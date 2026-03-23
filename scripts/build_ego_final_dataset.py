import os
import pandas as pd

base_path = os.path.expanduser("~/imptc_project/results/pedestrian_model_dataset.csv")
ego_path = os.path.expanduser("~/imptc_project/results/pedestrian_ego_vectors.csv")
out_path = os.path.expanduser("~/imptc_project/results/pedestrian_ego_final_dataset.csv")

base_df = pd.read_csv(base_path)
ego_df = pd.read_csv(ego_path)

# ενοποίηση με sample_id
df = base_df.merge(ego_df, on="sample_id", how="left")

df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print(list(df.columns))

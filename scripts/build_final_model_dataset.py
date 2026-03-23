import os
import pandas as pd

base_path = os.path.expanduser("~/imptc_project/results/pedestrian_model_dataset.csv")
nearestk_path = os.path.expanduser("~/imptc_project/results/pedestrian_social_nearestk.csv")
grid_path = os.path.expanduser("~/imptc_project/results/pedestrian_social_grid.csv")
out_path = os.path.expanduser("~/imptc_project/results/pedestrian_final_model_dataset.csv")

base_df = pd.read_csv(base_path)
nearestk_df = pd.read_csv(nearestk_path)
grid_df = pd.read_csv(grid_path)

df = base_df.merge(nearestk_df, on="sample_id", how="left")
df = df.merge(grid_df, on="sample_id", how="left", suffixes=("", "_grid"))

df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print(list(df.columns))

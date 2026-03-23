import os
import pandas as pd

context_path = os.path.expanduser("~/imptc_project/results/pedestrian_full_context.csv")
lights_path = os.path.expanduser("~/imptc_project/results/pedestrian_light_features.csv")
out_path = os.path.expanduser("~/imptc_project/results/pedestrian_model_dataset.csv")

context_df = pd.read_csv(context_path)
lights_df = pd.read_csv(lights_path)

df = context_df.merge(lights_df, on="sample_id", how="left")
df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(df))
print("Columns:", list(df.columns))

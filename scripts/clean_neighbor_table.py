import pandas as pd
import os

input_path = os.path.expanduser("~/imptc_project/results/pedestrian_all_neighbor_vectors.csv")
output_path = os.path.expanduser("~/imptc_project/results/pedestrian_neighbors_clean.csv")

df = pd.read_csv(input_path)

# κρατάμε μόνο τα σημαντικά columns
cols = [
    "sample_id",
    "target_id",
    "other_id",
    "other_class_name",
    "distance",
    "dx",
    "dy",
    "other_velocity",
    "relative_velocity",
]

df_clean = df[cols].copy()

# ταξινόμηση
df_clean = df_clean.sort_values(["sample_id", "distance"])

# save
df_clean.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Rows:", len(df_clean))

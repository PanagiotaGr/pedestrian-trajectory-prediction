import pandas as pd
import os

input_path = os.path.expanduser("~/imptc_project/results/pedestrian_ego_vectors_clean.csv")
output_path = os.path.expanduser("~/imptc_project/results/pedestrian_ego_vectors_table.csv")

df = pd.read_csv(input_path)

rows = []

TOP_K = 10

for _, row in df.iterrows():
    sample_id = row["sample_id"]

    for i in range(TOP_K):
        dx = row[f"nbr_{i}_dx"]
        dy = row[f"nbr_{i}_dy"]
        vel = row[f"nbr_{i}_vel"]

        # αγνοούμε empty neighbors
        if dx == 0 and dy == 0 and vel == 0:
            continue

        rows.append({
            "sample_id": sample_id,
            "neighbor_id": i,
            "dx": dx,
            "dy": dy,
            "velocity": vel
        })

out_df = pd.DataFrame(rows)

out_df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Rows:", len(out_df))

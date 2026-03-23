import os
import csv
import math
from collections import defaultdict

"""
EGO-CENTRIC representation:
Target pedestrian at (0,0)
Neighbors described by:
- dx, dy
- velocity
"""

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


input_path = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
output_path = os.path.expanduser("~/imptc_project/results/pedestrian_ego_vectors.csv")

RADIUS = 5.0
TOP_K = 10  # πόσους neighbors κρατάμε

grouped = defaultdict(list)

with open(input_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        grouped[row["sample_id"]].append(row)

rows_out = []

for sample_id, rows in grouped.items():

    neighbors = []

    first = rows[0]
    tx = safe_float(first["target_x"])
    ty = safe_float(first["target_y"])

    for r in rows:
        dist = safe_float(r["distance"], 9999)

        if dist > RADIUS or dist == 0:
            continue

        ox = safe_float(r["other_x"])
        oy = safe_float(r["other_y"])
        vel = safe_float(r["other_velocity"])

        dx = ox - tx
        dy = oy - ty

        neighbors.append((dist, dx, dy, vel))

    # sort by distance (πιο κοντινοί πρώτοι)
    neighbors.sort(key=lambda x: x[0])

    # κρατάμε TOP-K
    neighbors = neighbors[:TOP_K]

    out = {"sample_id": sample_id}

    for i in range(TOP_K):
        if i < len(neighbors):
            d, dx, dy, vel = neighbors[i]
            out[f"nbr_{i}_dx"] = dx
            out[f"nbr_{i}_dy"] = dy
            out[f"nbr_{i}_dist"] = d
            out[f"nbr_{i}_vel"] = vel
        else:
            out[f"nbr_{i}_dx"] = 0.0
            out[f"nbr_{i}_dy"] = 0.0
            out[f"nbr_{i}_dist"] = 0.0
            out[f"nbr_{i}_vel"] = 0.0

    rows_out.append(out)

# save
fieldnames = ["sample_id"]
for i in range(TOP_K):
    fieldnames += [
        f"nbr_{i}_dx",
        f"nbr_{i}_dy",
        f"nbr_{i}_dist",
        f"nbr_{i}_vel",
    ]

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", output_path)
print("Rows:", len(rows_out))

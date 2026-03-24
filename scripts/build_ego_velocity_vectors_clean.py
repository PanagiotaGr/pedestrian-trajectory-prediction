import os
import csv
from collections import defaultdict

"""
Clean ego-centric vector representation (like diagram)

Target at (0,0)
Neighbors represented ONLY by:
- dx
- dy
- velocity
"""

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


INPUT = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
OUTPUT = os.path.expanduser("~/imptc_project/results/pedestrian_ego_vectors_clean.csv")

RADIUS = 5.0
TOP_K = 10

grouped = defaultdict(list)

with open(INPUT, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        grouped[row["sample_id"]].append(row)


rows_out = []

for sample_id, rows in grouped.items():

    first = rows[0]
    tx = safe_float(first["target_x"])
    ty = safe_float(first["target_y"])

    neighbors = []

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

    # ταξινόμηση με βάση distance
    neighbors.sort(key=lambda x: x[0])

    neighbors = neighbors[:TOP_K]

    out = {"sample_id": sample_id}

    for i in range(TOP_K):
        if i < len(neighbors):
            _, dx, dy, vel = neighbors[i]

            out[f"nbr_{i}_dx"] = dx
            out[f"nbr_{i}_dy"] = dy
            out[f"nbr_{i}_vel"] = vel
        else:
            out[f"nbr_{i}_dx"] = 0.0
            out[f"nbr_{i}_dy"] = 0.0
            out[f"nbr_{i}_vel"] = 0.0

    rows_out.append(out)


# fields
fields = ["sample_id"]
for i in range(TOP_K):
    fields += [
        f"nbr_{i}_dx",
        f"nbr_{i}_dy",
        f"nbr_{i}_vel",
    ]


with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT)
print("Rows:", len(rows_out))

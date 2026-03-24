import os
import csv

MASTER_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_master_interactions.csv")
GRID_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_final_master.csv")
OUT_DIR = os.path.expanduser("~/imptc_project/results/sample_views")

os.makedirs(OUT_DIR, exist_ok=True)


def norm_sid(x):
    try:
        return f"{int(float(x)):04d}"
    except:
        return str(x).zfill(4)


# --------------------------------------------------
# load all neighbors grouped
# --------------------------------------------------
grouped_neighbors = {}

with open(MASTER_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sid = norm_sid(row["sample_id"])
        grouped_neighbors.setdefault(sid, []).append(row)


# --------------------------------------------------
# load all grid rows
# --------------------------------------------------
grid_rows = {}
with open(GRID_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sid = norm_sid(row["sample_id"])
        grid_rows[sid] = row


# --------------------------------------------------
# loop over all samples
# --------------------------------------------------
count = 0

for sid, neighbors in grouped_neighbors.items():

    # ---------- neighbors file ----------
    neighbor_out = os.path.join(OUT_DIR, f"{sid}_neighbors.csv")

    neighbor_fields = [
        "sample_id",
        "scene_path",
        "timestamp",
        "target_id",
        "target_class_name",
        "target_x_ego",
        "target_y_ego",
        "target_velocity",
        "map_label_name",
        "f1_state",
        "f2_state",
        "f3_state",
        "neighbor_rank",
        "other_type",
        "other_class_name",
        "other_velocity",
        "dx",
        "dy",
        "distance",
    ]

    with open(neighbor_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=neighbor_fields)
        writer.writeheader()
        for r in neighbors:
            writer.writerow({k: r.get(k, "") for k in neighbor_fields})

    # ---------- grid file ----------
    if sid not in grid_rows:
        continue

    grid_row = grid_rows[sid]

    grid_out = os.path.join(OUT_DIR, f"{sid}_grid.csv")

    grid_fields = ["row", "col", "count", "mean_vel", "min_dist"]

    rows_out = []
    GRID_SIZE = 5

    for cell_id in range(GRID_SIZE * GRID_SIZE):
        r = cell_id // GRID_SIZE
        c = cell_id % GRID_SIZE

        rows_out.append({
            "row": r,
            "col": c,
            "count": grid_row.get(f"cell_{cell_id}_count", ""),
            "mean_vel": grid_row.get(f"cell_{cell_id}_mean_vel", ""),
            "min_dist": grid_row.get(f"cell_{cell_id}_min_dist", ""),
        })

    with open(grid_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=grid_fields)
        writer.writeheader()
        writer.writerows(rows_out)

    count += 1

print("Done.")
print("Samples processed:", count)
print("Saved in:", OUT_DIR)

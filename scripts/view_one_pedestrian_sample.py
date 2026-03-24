import os
import csv
import sys

MASTER_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_master_interactions.csv")
GRID_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_final_master.csv")
OUT_DIR = os.path.expanduser("~/imptc_project/results")


def norm_sid(x):
    try:
        return f"{int(float(x)):04d}"
    except Exception:
        return str(x).zfill(4)


if len(sys.argv) < 2:
    print("Usage: python3 view_one_pedestrian_sample.py <sample_id>")
    sys.exit(1)

sample_id = norm_sid(sys.argv[1])

# --------------------------------------------------
# 1) detailed neighbor rows
# --------------------------------------------------
neighbor_rows = []
with open(MASTER_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if norm_sid(row["sample_id"]) == sample_id:
            neighbor_rows.append(row)

if not neighbor_rows:
    print("Sample not found:", sample_id)
    sys.exit(1)

neighbor_out = os.path.join(OUT_DIR, f"sample_{sample_id}_neighbors_view.csv")

neighbor_fields = [
    "sample_id",
    "scene_path",
    "timestamp",
    "target_id",
    "target_class_name",
    "target_x_world",
    "target_y_world",
    "target_x_ego",
    "target_y_ego",
    "target_velocity",
    "map_label_name",
    "is_crosswalk",
    "is_sidewalk",
    "is_road",
    "f1_state",
    "f2_state",
    "f3_state",
    "n_ped_green",
    "n_ped_red",
    "has_green",
    "red_only",
    "is_moving",
    "neighbor_rank",
    "other_type",
    "other_id",
    "other_class_name",
    "other_velocity",
    "dx",
    "dy",
    "dz",
    "distance",
]

with open(neighbor_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=neighbor_fields)
    writer.writeheader()
    for r in neighbor_rows:
        writer.writerow({k: r.get(k, "") for k in neighbor_fields})

# --------------------------------------------------
# 2) 5x5 grid row from final master
# --------------------------------------------------
grid_row = None
with open(GRID_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if norm_sid(row["sample_id"]) == sample_id:
            grid_row = row
            break

if grid_row is None:
    print("Grid row not found for:", sample_id)
    sys.exit(1)

grid_out = os.path.join(OUT_DIR, f"sample_{sample_id}_grid_view.csv")

# φτιάχνουμε πίνακα 5x5
grid_fields = [
    "row",
    "col",
    "cell_id",
    "count",
    "mean_vel",
    "min_dist",
]

grid_rows = []
grid_size = 5

for cell_id in range(grid_size * grid_size):
    row_idx = cell_id // grid_size
    col_idx = cell_id % grid_size

    grid_rows.append({
        "row": row_idx,
        "col": col_idx,
        "cell_id": cell_id,
        "count": grid_row.get(f"cell_{cell_id}_count", ""),
        "mean_vel": grid_row.get(f"cell_{cell_id}_mean_vel", ""),
        "min_dist": grid_row.get(f"cell_{cell_id}_min_dist", ""),
    })

with open(grid_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=grid_fields)
    writer.writeheader()
    writer.writerows(grid_rows)

# --------------------------------------------------
# 3) small text summary in terminal
# --------------------------------------------------
first = neighbor_rows[0]

print("Saved:", neighbor_out)
print("Saved:", grid_out)
print()
print("Sample:", sample_id)
print("Scene:", first["scene_path"])
print("Timestamp:", first["timestamp"])
print("Target:", first["target_class_name"], first["target_id"])
print("Map label:", first["map_label_name"])
print("Traffic lights: f1 =", first["f1_state"], ", f2 =", first["f2_state"], ", f3 =", first["f3_state"])
print("Green count:", first["n_ped_green"], "| Red count:", first["n_ped_red"])
print("Neighbors within 5m:", len(neighbor_rows))

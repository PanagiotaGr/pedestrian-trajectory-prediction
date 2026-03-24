import os
import csv
import math
from collections import defaultdict

INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
MAP_LABELS_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_map_labels.csv")
LIGHTS_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_crossing_behavior.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_ego_grid_velocity_dataset.csv")

RADIUS = 5.0
GRID_SIZE = 3  # 3x3
CELL_SIZE = (2 * RADIUS) / GRID_SIZE


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


# -------------------------------------------------------------------
# load map labels per sample_id
# -------------------------------------------------------------------
map_info = {}
with open(MAP_LABELS_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        map_info[row["sample_id"]] = {
            "map_label_id": row["map_label_id"],
            "map_label_name": row["map_label_name"],
            "is_crosswalk": row["is_crosswalk"],
            "is_sidewalk": row["is_sidewalk"],
            "is_road": row["is_road"],
            "is_bikelane": row["is_bikelane"],
            "nearest_map_dist": row["nearest_map_dist"],
        }

# -------------------------------------------------------------------
# load light / behavior info per sample_id
# -------------------------------------------------------------------
light_info = {}
with open(LIGHTS_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        light_info[row["sample_id"]] = {
            "displacement": row["displacement"],
            "avg_speed": row["avg_speed"],
            "f1_state": row["f1_state"],
            "f2_state": row["f2_state"],
            "f3_state": row["f3_state"],
            "n_ped_green": row["n_ped_green"],
            "n_ped_red": row["n_ped_red"],
            "has_green": row["has_green"],
            "red_only": row["red_only"],
            "is_moving": row["is_moving"],
            "nearest_dist": row["nearest_dist"],
            "n_neighbors": row["n_neighbors"],
        }

# -------------------------------------------------------------------
# group neighbors by sample_id
# -------------------------------------------------------------------
grouped = defaultdict(list)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        grouped[row["sample_id"]].append(row)

rows_out = []

for sample_id, rows in grouped.items():
    first = rows[0]

    tx = safe_float(first["target_x"])
    ty = safe_float(first["target_y"])
    target_velocity = safe_float(first["target_velocity"])

    # grid cells
    # for each cell keep stats
    cells = []
    for _ in range(GRID_SIZE * GRID_SIZE):
        cells.append({
            "count": 0,
            "sum_velocity": 0.0,
            "min_distance": None,
            "ped_count": 0,
            "vehicle_count": 0,
            "vru_count": 0,
        })

    total_neighbors_within_radius = 0

    for r in rows:
        dist = safe_float(r["distance"], 9999.0)
        if dist <= 0.0 or dist > RADIUS:
            continue

        ox = safe_float(r["other_x"])
        oy = safe_float(r["other_y"])
        other_velocity = safe_float(r["other_velocity"])

        dx = ox - tx
        dy = oy - ty

        # convert from ego coordinates to cell indices
        # x in [-RADIUS, +RADIUS], y in [-RADIUS, +RADIUS]
        gx = int((dx + RADIUS) / CELL_SIZE)
        gy = int((dy + RADIUS) / CELL_SIZE)

        # boundary correction
        if gx < 0 or gx >= GRID_SIZE or gy < 0 or gy >= GRID_SIZE:
            continue

        cell_idx = gy * GRID_SIZE + gx
        cell = cells[cell_idx]

        cell["count"] += 1
        cell["sum_velocity"] += other_velocity

        if cell["min_distance"] is None or dist < cell["min_distance"]:
            cell["min_distance"] = dist

        other_type = r["other_type"]
        other_class = r["other_class_name"]

        if other_class == "person":
            cell["ped_count"] += 1

        if other_type == "vehicles":
            cell["vehicle_count"] += 1

        if other_type == "vrus":
            cell["vru_count"] += 1

        total_neighbors_within_radius += 1

    out = {
        "sample_id": sample_id,
        "scene_path": first["scene_path"],
        "timestamp": first["timestamp"],
        "target_id": first["target_id"],
        "target_class_name": first["target_class_name"],
        "target_x": round(tx, 6),
        "target_y": round(ty, 6),
        "target_velocity": round(target_velocity, 6),
        "radius": RADIUS,
        "grid_size": GRID_SIZE,
        "cell_size": round(CELL_SIZE, 6),
        "total_neighbors_within_radius": total_neighbors_within_radius,
    }

    # add light info
    if sample_id in light_info:
        out.update(light_info[sample_id])

    # add map info
    if sample_id in map_info:
        out.update(map_info[sample_id])

    # add per-cell info
    for i, cell in enumerate(cells):
        mean_velocity = (
            cell["sum_velocity"] / cell["count"] if cell["count"] > 0 else 0.0
        )
        min_distance = cell["min_distance"] if cell["min_distance"] is not None else 0.0

        out[f"cell_{i}_count"] = cell["count"]
        out[f"cell_{i}_sum_velocity"] = round(cell["sum_velocity"], 6)
        out[f"cell_{i}_mean_velocity"] = round(mean_velocity, 6)
        out[f"cell_{i}_min_distance"] = round(min_distance, 6)
        out[f"cell_{i}_ped_count"] = cell["ped_count"]
        out[f"cell_{i}_vehicle_count"] = cell["vehicle_count"]
        out[f"cell_{i}_vru_count"] = cell["vru_count"]

    rows_out.append(out)

# -------------------------------------------------------------------
# write output
# -------------------------------------------------------------------
fieldnames = list(rows_out[0].keys())

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))
print("Grid:", f"{GRID_SIZE}x{GRID_SIZE}")
print("Radius:", RADIUS)
print("Cell size:", CELL_SIZE)

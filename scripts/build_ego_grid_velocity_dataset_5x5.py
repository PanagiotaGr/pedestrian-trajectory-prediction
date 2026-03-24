import os
import csv
from collections import defaultdict

INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
MAP_LABELS_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_map_labels.csv")
LIGHTS_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_crossing_behavior.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_ego_grid_velocity_dataset_5x5.csv")

RADIUS = 5.0
GRID_SIZE = 5
CELL_SIZE = (2 * RADIUS) / GRID_SIZE


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ---------------- load map info ----------------
map_info = {}
with open(MAP_LABELS_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        map_info[row["sample_id"]] = {
            "map_label_id": row.get("map_label_id", ""),
            "map_label_name": row.get("map_label_name", ""),
            "is_crosswalk": row.get("is_crosswalk", ""),
            "is_sidewalk": row.get("is_sidewalk", ""),
            "is_road": row.get("is_road", ""),
            "is_bikelane": row.get("is_bikelane", ""),
            "nearest_map_dist": row.get("nearest_map_dist", ""),
            "target_x": row.get("target_x", ""),
            "target_y": row.get("target_y", ""),
        }

# ---------------- load light info ----------------
light_info = {}
with open(LIGHTS_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        light_info[row["sample_id"]] = {
            "displacement": row.get("displacement", ""),
            "avg_speed": row.get("avg_speed", ""),
            "f1_state": row.get("f1_state", ""),
            "f2_state": row.get("f2_state", ""),
            "f3_state": row.get("f3_state", ""),
            "n_ped_green": row.get("n_ped_green", ""),
            "n_ped_red": row.get("n_ped_red", ""),
            "has_green": row.get("has_green", ""),
            "red_only": row.get("red_only", ""),
            "is_moving": row.get("is_moving", ""),
            "nearest_dist": row.get("nearest_dist", ""),
            "n_neighbors": row.get("n_neighbors", ""),
        }

# ---------------- group neighbors ----------------
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

    total_neighbors = 0

    for r in rows:
        dist = safe_float(r["distance"], 9999.0)
        if dist <= 0.0 or dist > RADIUS:
            continue

        ox = safe_float(r["other_x"])
        oy = safe_float(r["other_y"])
        vel = safe_float(r["other_velocity"])

        dx = ox - tx
        dy = oy - ty

        gx = int((dx + RADIUS) / CELL_SIZE)
        gy = int((dy + RADIUS) / CELL_SIZE)

        if gx < 0 or gx >= GRID_SIZE or gy < 0 or gy >= GRID_SIZE:
            continue

        idx = gy * GRID_SIZE + gx
        cell = cells[idx]

        cell["count"] += 1
        cell["sum_velocity"] += vel

        if cell["min_distance"] is None or dist < cell["min_distance"]:
            cell["min_distance"] = dist

        if r["other_class_name"] == "person":
            cell["ped_count"] += 1
        if r["other_type"] == "vehicles":
            cell["vehicle_count"] += 1
        if r["other_type"] == "vrus":
            cell["vru_count"] += 1

        total_neighbors += 1

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
        "total_neighbors_within_radius": total_neighbors,
    }

    if sample_id in light_info:
        out.update(light_info[sample_id])

    if sample_id in map_info:
        out["map_label_id"] = map_info[sample_id].get("map_label_id", "")
        out["map_label_name"] = map_info[sample_id].get("map_label_name", "")
        out["is_crosswalk"] = map_info[sample_id].get("is_crosswalk", "")
        out["is_sidewalk"] = map_info[sample_id].get("is_sidewalk", "")
        out["is_road"] = map_info[sample_id].get("is_road", "")
        out["is_bikelane"] = map_info[sample_id].get("is_bikelane", "")
        out["nearest_map_dist"] = map_info[sample_id].get("nearest_map_dist", "")

    for i, cell in enumerate(cells):
        mean_velocity = cell["sum_velocity"] / cell["count"] if cell["count"] > 0 else 0.0
        min_distance = cell["min_distance"] if cell["min_distance"] is not None else 0.0

        out[f"cell_{i}_count"] = cell["count"]
        out[f"cell_{i}_sum_velocity"] = round(cell["sum_velocity"], 6)
        out[f"cell_{i}_mean_velocity"] = round(mean_velocity, 6)
        out[f"cell_{i}_min_distance"] = round(min_distance, 6)
        out[f"cell_{i}_ped_count"] = cell["ped_count"]
        out[f"cell_{i}_vehicle_count"] = cell["vehicle_count"]
        out[f"cell_{i}_vru_count"] = cell["vru_count"]

    rows_out.append(out)

base_fields = [
    "sample_id",
    "scene_path",
    "timestamp",
    "target_id",
    "target_class_name",
    "target_x",
    "target_y",
    "target_velocity",
    "radius",
    "grid_size",
    "cell_size",
    "total_neighbors_within_radius",
    "displacement",
    "avg_speed",
    "f1_state",
    "f2_state",
    "f3_state",
    "n_ped_green",
    "n_ped_red",
    "has_green",
    "red_only",
    "is_moving",
    "nearest_dist",
    "n_neighbors",
    "map_label_id",
    "map_label_name",
    "is_crosswalk",
    "is_sidewalk",
    "is_road",
    "is_bikelane",
    "nearest_map_dist",
]

cell_fields = []
for i in range(GRID_SIZE * GRID_SIZE):
    cell_fields += [
        f"cell_{i}_count",
        f"cell_{i}_sum_velocity",
        f"cell_{i}_mean_velocity",
        f"cell_{i}_min_distance",
        f"cell_{i}_ped_count",
        f"cell_{i}_vehicle_count",
        f"cell_{i}_vru_count",
    ]

fieldnames = base_fields + cell_fields

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))
print("Grid:", f"{GRID_SIZE}x{GRID_SIZE}")
print("Radius:", RADIUS)
print("Cell size:", CELL_SIZE)

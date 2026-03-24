kkkkkimport os
import csv
from collections import defaultdict

NEIGHBOR_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
LIGHT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_crossing_behavior.csv")
MAP_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_map_labels.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_ego_detailed_dataset.csv")

RADIUS = 5.0


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# --------------------------------------------------
# load lights info per sample_id
# --------------------------------------------------
light_info = {}
with open(LIGHT_PATH, "r", encoding="utf-8") as f:
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

# --------------------------------------------------
# load map info per sample_id
# --------------------------------------------------
map_info = {}
with open(MAP_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        map_info[row["sample_id"]] = {
            "map_label_id": row.get("map_label_id", ""),
            "map_label_name": row.get("map_label_name", ""),
            "is_crosswalk": row.get("is_crosswalk", ""),
            "is_sidewalk": row.get("is_sidewalk", ""),
            "is_road": row.get("is_road", ""),
            "is_bikelane": row.get("is_bikelane", ""),
            "nearest_map_dist": row.get("nearest_map_dist", ""),
        }

# --------------------------------------------------
# group neighbors by sample_id
# --------------------------------------------------
grouped = defaultdict(list)

with open(NEIGHBOR_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        grouped[row["sample_id"]].append(row)

rows_out = []

for sample_id, rows in grouped.items():
    first = rows[0]

    target_x = safe_float(first["target_x"])
    target_y = safe_float(first["target_y"])
    target_z = safe_float(first["target_z"])
    target_velocity = safe_float(first["target_velocity"])

    lights = light_info.get(sample_id, {})
    semantic = map_info.get(sample_id, {})

    for r in rows:
        dist = safe_float(r["distance"], 9999.0)
        if dist <= 0.0 or dist > RADIUS:
            continue

        other_x = safe_float(r["other_x"])
        other_y = safe_float(r["other_y"])
        other_z = safe_float(r["other_z"])
        other_velocity = safe_float(r["other_velocity"])

        dx = other_x - target_x
        dy = other_y - target_y
        dz = other_z - target_z

        rows_out.append({
            # target pedestrian
            "sample_id": sample_id,
            "scene_path": first["scene_path"],
            "archive": first["archive"],
            "timestamp": first["timestamp"],
            "target_type": first["target_type"],
            "target_id": first["target_id"],
            "target_class_name": first["target_class_name"],
            "target_x_world": round(target_x, 6),
            "target_y_world": round(target_y, 6),
            "target_z_world": round(target_z, 6),
            "target_x_ego": 0.0,
            "target_y_ego": 0.0,
            "target_velocity": round(target_velocity, 6),

            # neighbor info
            "other_type": r["other_type"],
            "other_id": r["other_id"],
            "other_class_name": r["other_class_name"],
            "other_x_world": round(other_x, 6),
            "other_y_world": round(other_y, 6),
            "other_z_world": round(other_z, 6),
            "other_velocity": round(other_velocity, 6),
            "dx": round(dx, 6),
            "dy": round(dy, 6),
            "dz": round(dz, 6),
            "distance": round(dist, 6),

            # traffic lights
            "f1_state": lights.get("f1_state", ""),
            "f2_state": lights.get("f2_state", ""),
            "f3_state": lights.get("f3_state", ""),
            "n_ped_green": lights.get("n_ped_green", ""),
            "n_ped_red": lights.get("n_ped_red", ""),
            "has_green": lights.get("has_green", ""),
            "red_only": lights.get("red_only", ""),
            "is_moving": lights.get("is_moving", ""),
            "displacement": lights.get("displacement", ""),
            "avg_speed": lights.get("avg_speed", ""),

            # semantic map
            "map_label_id": semantic.get("map_label_id", ""),
            "map_label_name": semantic.get("map_label_name", ""),
            "is_crosswalk": semantic.get("is_crosswalk", ""),
            "is_sidewalk": semantic.get("is_sidewalk", ""),
            "is_road": semantic.get("is_road", ""),
            "is_bikelane": semantic.get("is_bikelane", ""),
            "nearest_map_dist": semantic.get("nearest_map_dist", ""),
        })

fieldnames = [
    "sample_id", "scene_path", "archive", "timestamp",
    "target_type", "target_id", "target_class_name",
    "target_x_world", "target_y_world", "target_z_world",
    "target_x_ego", "target_y_ego", "target_velocity",
    "other_type", "other_id", "other_class_name",
    "other_x_world", "other_y_world", "other_z_world",
    "other_velocity", "dx", "dy", "dz", "distance",
    "f1_state", "f2_state", "f3_state",
    "n_ped_green", "n_ped_red", "has_green", "red_only",
    "is_moving", "displacement", "avg_speed",
    "map_label_id", "map_label_name",
    "is_crosswalk", "is_sidewalk", "is_road", "is_bikelane",
    "nearest_map_dist",
]

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))
print("Radius:", RADIUS)

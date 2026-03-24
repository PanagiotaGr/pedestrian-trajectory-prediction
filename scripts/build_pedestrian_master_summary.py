import os
import csv
from collections import defaultdict

INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_master_interactions.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_master_summary.csv")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


grouped = defaultdict(list)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        grouped[row["sample_id"]].append(row)

rows_out = []

for sample_id, rows in grouped.items():
    first = rows[0]

    n_neighbors = len(rows)
    person_count = 0
    scooter_count = 0
    vehicle_count = 0

    neighbor_vel_sum = 0.0
    nearest_neighbor_dist = None
    nearest_neighbor_type = ""
    nearest_neighbor_class = ""

    for r in rows:
        dist = safe_float(r["distance"])
        vel = safe_float(r["other_velocity"])
        neighbor_vel_sum += vel

        if nearest_neighbor_dist is None or dist < nearest_neighbor_dist:
            nearest_neighbor_dist = dist
            nearest_neighbor_type = r["other_type"]
            nearest_neighbor_class = r["other_class_name"]

        if r["other_class_name"] == "person":
            person_count += 1
        if r["other_class_name"] == "scooter":
            scooter_count += 1
        if r["other_type"] == "vehicles":
            vehicle_count += 1

    avg_neighbor_velocity = neighbor_vel_sum / n_neighbors if n_neighbors > 0 else 0.0

    rows_out.append({
        "sample_id": sample_id,
        "scene_path": first["scene_path"],
        "timestamp": first["timestamp"],
        "target_id": first["target_id"],
        "target_class_name": first["target_class_name"],
        "target_x_world": first["target_x_world"],
        "target_y_world": first["target_y_world"],
        "target_velocity": first["target_velocity"],

        "map_label_name": first["map_label_name"],
        "is_crosswalk": first["is_crosswalk"],
        "is_sidewalk": first["is_sidewalk"],
        "is_road": first["is_road"],
        "nearest_map_dist": first["nearest_map_dist"],

        "f1_state": first["f1_state"],
        "f2_state": first["f2_state"],
        "f3_state": first["f3_state"],
        "n_ped_green": first["n_ped_green"],
        "n_ped_red": first["n_ped_red"],
        "has_green": first["has_green"],
        "red_only": first["red_only"],
        "is_moving": first["is_moving"],
        "displacement": first["displacement"],
        "avg_speed": first["avg_speed"],

        "neighbor_count_within_5m": n_neighbors,
        "person_neighbors": person_count,
        "scooter_neighbors": scooter_count,
        "vehicle_neighbors": vehicle_count,
        "avg_neighbor_velocity": round(avg_neighbor_velocity, 6),
        "nearest_neighbor_dist": round(nearest_neighbor_dist, 6) if nearest_neighbor_dist is not None else 0.0,
        "nearest_neighbor_type": nearest_neighbor_type,
        "nearest_neighbor_class": nearest_neighbor_class,
    })

fieldnames = list(rows_out[0].keys())

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))

import os
import csv
import math

INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_master_interactions.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_neighbors_within_5m_vectors.csv")

RADIUS = 5.0


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


rows_out = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # κρατάμε μόνο VRUs
        if row.get("other_type", "") != "vrus":
            continue

        dx = safe_float(row.get("dx", 0.0))
        dy = safe_float(row.get("dy", 0.0))
        dz = safe_float(row.get("dz", 0.0))
        other_velocity = safe_float(row.get("other_velocity", 0.0))

        distance = math.sqrt(dx * dx + dy * dy)

        # κρατάμε μόνο όσους είναι κοντά σε 5m
        if distance > RADIUS:
            continue

        rows_out.append({
            "sample_id": row["sample_id"],
            "scene_path": row["scene_path"],
            "timestamp": row["timestamp"],
            "target_id": row["target_id"],
            "target_class_name": row["target_class_name"],
            "target_x_ego": 0.0,
            "target_y_ego": 0.0,
            "target_velocity": row.get("target_velocity", ""),
            "other_type": row["other_type"],
            "other_id": row["other_id"],
            "other_class_name": row["other_class_name"],
            "other_velocity": round(other_velocity, 6),
            "dx": round(dx, 6),
            "dy": round(dy, 6),
            "dz": round(dz, 6),
            "distance": round(distance, 6),
        })

fieldnames = [
    "sample_id",
    "scene_path",
    "timestamp",
    "target_id",
    "target_class_name",
    "target_x_ego",
    "target_y_ego",
    "target_velocity",
    "other_type",
    "other_id",
    "other_class_name",
    "other_velocity",
    "dx",
    "dy",
    "dz",
    "distance",
]

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))
print("Radius:", RADIUS)

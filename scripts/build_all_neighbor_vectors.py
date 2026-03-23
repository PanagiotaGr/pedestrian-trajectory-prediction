import os
import csv

"""
Build detailed ego-centric neighbor table with relative velocity.

For each pedestrian sample:
- keep all neighbors within radius
- express them relative to target pedestrian
- compute relative velocity
- save one row per neighbor
"""

INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_all_neighbor_vectors.csv")

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
        dist = safe_float(row["distance"], 9999.0)

        if dist > RADIUS:
            continue

        # target
        tx = safe_float(row["target_x"])
        ty = safe_float(row["target_y"])
        target_vel = safe_float(row["target_velocity"])

        # other
        ox = safe_float(row["other_x"])
        oy = safe_float(row["other_y"])
        other_vel = safe_float(row["other_velocity"])

        # relative position
        dx = ox - tx
        dy = oy - ty

        # relative velocity
        rel_vel = other_vel - target_vel

        rows_out.append({
            "sample_id": row["sample_id"],
            "scene_path": row["scene_path"],
            "timestamp": row["timestamp"],
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "target_class_name": row["target_class_name"],
            "target_x": row["target_x"],
            "target_y": row["target_y"],
            "target_velocity": target_vel,
            "other_type": row["other_type"],
            "other_id": row["other_id"],
            "other_class_name": row["other_class_name"],
            "other_x": row["other_x"],
            "other_y": row["other_y"],
            "other_velocity": other_vel,
            "relative_velocity": round(rel_vel, 6),
            "dx": round(dx, 6),
            "dy": round(dy, 6),
            "distance": round(dist, 6),
            "other_member_path": row["other_member_path"],
        })


fieldnames = [
    "sample_id",
    "scene_path",
    "timestamp",
    "target_type",
    "target_id",
    "target_class_name",
    "target_x",
    "target_y",
    "target_velocity",
    "other_type",
    "other_id",
    "other_class_name",
    "other_x",
    "other_y",
    "other_velocity",
    "relative_velocity",
    "dx",
    "dy",
    "distance",
    "other_member_path",
]


with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))
print("Radius:", RADIUS)

import os
import csv
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

MAP_PATH = os.path.expanduser("~/imptc_project/data/map.ply")
INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_map_labels.csv")


def rgb_to_label_name(r, g, b):
    rgb = (int(r), int(g), int(b))

    if rgb == (128, 64, 128):
        return 0, "road"
    elif rgb == (244, 35, 232):
        return 1, "sidewalk"
    elif rgb == (81, 0, 81):
        return 2, "ground"
    elif rgb == (150, 100, 100):
        return 3, "curb"
    elif rgb == (157, 234, 50):
        return 4, "road_line"
    elif rgb == (229, 165, 10):
        return 5, "crosswalk"
    elif rgb == (98, 160, 234):
        return 6, "bikelane"
    elif rgb == (128, 128, 128):
        return 7, "unknown"
    else:
        return -1, "other"


print("Loading map...")
pcd = o3d.io.read_point_cloud(MAP_PATH)

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print("Loaded points:", len(points))

if len(points) == 0:
    raise RuntimeError("Map loaded with 0 points. Check MAP_PATH.")

colors255 = np.round(colors * 255).astype(int)

print("Building KDTree...")
tree = cKDTree(points[:, :2])

rows_out = []
seen = set()

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        sid = row["sample_id"]

        # κρατάμε μία φορά κάθε pedestrian sample
        if sid in seen:
            continue
        seen.add(sid)

        x = float(row["target_x"])
        y = float(row["target_y"])

        dist, idx = tree.query([x, y])

        r, g, b = colors255[idx]
        label_id, label_name = rgb_to_label_name(r, g, b)

        rows_out.append({
            "sample_id": sid,
            "scene_path": row["scene_path"],
            "timestamp": row["timestamp"],
            "target_id": row["target_id"],
            "target_class_name": row["target_class_name"],
            "target_x": round(x, 6),
            "target_y": round(y, 6),
            "nearest_map_dist": round(float(dist), 6),
            "map_label_id": label_id,
            "map_label_name": label_name,
            "is_crosswalk": 1 if label_id == 5 else 0,
            "is_sidewalk": 1 if label_id == 1 else 0,
            "is_road": 1 if label_id == 0 else 0,
            "is_bikelane": 1 if label_id == 6 else 0,
        })

fieldnames = list(rows_out[0].keys())

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))

import os
import csv
import numpy as np

BASE_SCENE_PATH = os.path.expanduser("~/imptc_project/raw_scene")
INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_crosswalk_status.csv")


def load_ply_points(path):
    pts = []

    with open(path, "r") as f:
        header = True
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            x, y, z = map(float, parts[:3])
            r, g, b = map(int, parts[3:6])

            pts.append((x, y, r, g, b))

    return np.array(pts)


def rgb_to_label(r, g, b):
    if (r, g, b) == (229,165,10):
        return 5  # crosswalk
    if (r, g, b) == (244,35,232):
        return 1  # sidewalk
    if (r, g, b) == (128,64,128):
        return 0  # road
    return -1


scene_cache = {}

rows_out = []

with open(INPUT_PATH, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        scene = row["scene_path"]

        # load scene map (cache για speed)
        if scene not in scene_cache:
            ply_path = os.path.join(BASE_SCENE_PATH, scene, "ground_plane.ply")

            if not os.path.exists(ply_path):
                print("Missing:", ply_path)
                scene_cache[scene] = None
                continue

            print("Loading:", scene)
            scene_cache[scene] = load_ply_points(ply_path)

        points = scene_cache[scene]

        if points is None:
            continue

        map_xy = points[:, :2]
        map_rgb = points[:, 2:]

        x = float(row["target_x"])
        y = float(row["target_y"])

        dists = np.linalg.norm(map_xy - [x, y], axis=1)
        idx = np.argmin(dists)

        r, g, b = map_rgb[idx]
        label = rgb_to_label(int(r), int(g), int(b))

        rows_out.append({
            "sample_id": row["sample_id"],
            "scene": scene,
            "x": x,
            "y": y,
            "map_label": label,
            "is_crosswalk": 1 if label == 5 else 0
        })


with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sample_id", "scene", "x", "y", "map_label", "is_crosswalk"]
    )
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)

import os
import math
import argparse
import pandas as pd
import numpy as np
from plyfile import PlyData

"""
Build semantic crosswalk features from segmentation .ply maps.

Label IDs:
0 road
1 sidewalk
2 ground
3 curb
4 road line
5 crosswalk
6 bikelane
7 unknown

Important:
- The .ply uses the same coordinate system as trajectories.
- We use pedestrian_moments_summary.csv for target pedestrian position at time t.
"""

CROSSWALK_LABEL = 5
SIDEWALK_LABEL = 1
ROAD_LABEL = 0


def load_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"].data

    cols = vertex.dtype.names

    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)

    # Try to infer label from common field names
    label_field_candidates = ["label", "class", "semantic", "segmentation", "scalar_label"]
    label_field = None
    for c in label_field_candidates:
        if c in cols:
            label_field = c
            break

    if label_field is None:
        raise ValueError(
            f"No semantic label field found in {ply_path}. Available fields: {cols}"
        )

    labels = np.asarray(vertex[label_field], dtype=np.int32)

    return x, y, z, labels


def nearest_distance_2d(px, py, points_x, points_y):
    if len(points_x) == 0:
        return None

    dx = points_x - px
    dy = points_y - py
    d2 = dx * dx + dy * dy
    return float(np.sqrt(np.min(d2)))


def point_near_label(px, py, points_x, points_y, radius):
    if len(points_x) == 0:
        return 0
    dx = points_x - px
    dy = points_y - py
    d2 = dx * dx + dy * dy
    return int(np.any(d2 <= radius * radius))


def nearest_label_of_point(px, py, all_x, all_y, all_labels):
    dx = all_x - px
    dy = all_y - py
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    return int(all_labels[idx]), float(np.sqrt(d2[idx]))


def find_scene_ply(scene_path, search_root, map_filename=None):
    """
    Search for the scene's .ply map using scene_path, e.g. 0001_20230322_083454
    """
    scene_dir = None
    for root, dirs, files in os.walk(search_root):
        if os.path.basename(root) == scene_path:
            scene_dir = root
            break

    if scene_dir is None:
        return None

    # If user gave explicit map filename, use it
    if map_filename:
        candidate = os.path.join(scene_dir, map_filename)
        if os.path.exists(candidate):
            return candidate

    # Otherwise pick first .ply inside scene tree
    for root, dirs, files in os.walk(scene_dir):
        for f in files:
            if f.lower().endswith(".ply"):
                return os.path.join(root, f)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moments",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_moments_summary.csv")
    )
    parser.add_argument(
        "--search-root",
        default=os.path.expanduser("~/imptc_project")
    )
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_crosswalk_features.csv")
    )
    parser.add_argument(
        "--map-filename",
        default=None,
        help="Optional exact .ply filename inside each scene folder"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.moments)

    # cache loaded maps by scene_path
    scene_cache = {}
    rows_out = []

    for i, row in df.iterrows():
        sample_id = row["sample_id"]
        scene_path = row["scene_path"]

        px = float(row["target_x"])
        py = float(row["target_y"])
        pz = float(row["target_z"])

        if scene_path not in scene_cache:
            ply_path = find_scene_ply(scene_path, args.search_root, args.map_filename)
            if ply_path is None:
                scene_cache[scene_path] = None
            else:
                try:
                    x, y, z, labels = load_ply_points(ply_path)
                    scene_cache[scene_path] = {
                        "ply_path": ply_path,
                        "x": x,
                        "y": y,
                        "z": z,
                        "labels": labels,
                        "crosswalk_x": x[labels == CROSSWALK_LABEL],
                        "crosswalk_y": y[labels == CROSSWALK_LABEL],
                        "sidewalk_x": x[labels == SIDEWALK_LABEL],
                        "sidewalk_y": y[labels == SIDEWALK_LABEL],
                        "road_x": x[labels == ROAD_LABEL],
                        "road_y": y[labels == ROAD_LABEL],
                    }
                except Exception as e:
                    print(f"[WARN] Failed loading {scene_path}: {e}")
                    scene_cache[scene_path] = None

        cached = scene_cache[scene_path]

        if cached is None:
            rows_out.append({
                "sample_id": sample_id,
                "scene_path": scene_path,
                "dist_to_crosswalk": "",
                "has_crosswalk_near_2m": 0,
                "has_crosswalk_near_5m": 0,
                "nearest_map_label": "",
                "nearest_map_label_dist": "",
                "on_crosswalk": 0,
                "on_sidewalk": 0,
                "on_road": 0,
                "status": "map_not_found",
            })
            continue

        dist_crosswalk = nearest_distance_2d(
            px, py,
            cached["crosswalk_x"],
            cached["crosswalk_y"]
        )

        nearest_label, nearest_label_dist = nearest_label_of_point(
            px, py,
            cached["x"], cached["y"], cached["labels"]
        )

        rows_out.append({
            "sample_id": sample_id,
            "scene_path": scene_path,
            "dist_to_crosswalk": round(dist_crosswalk, 6) if dist_crosswalk is not None else "",
            "has_crosswalk_near_2m": point_near_label(px, py, cached["crosswalk_x"], cached["crosswalk_y"], 2.0),
            "has_crosswalk_near_5m": point_near_label(px, py, cached["crosswalk_x"], cached["crosswalk_y"], 5.0),
            "nearest_map_label": nearest_label,
            "nearest_map_label_dist": round(nearest_label_dist, 6),
            "on_crosswalk": int(nearest_label == CROSSWALK_LABEL),
            "on_sidewalk": int(nearest_label == SIDEWALK_LABEL),
            "on_road": int(nearest_label == ROAD_LABEL),
            "status": "ok",
        })

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Rows:", len(out_df))


if __name__ == "__main__":
    main()

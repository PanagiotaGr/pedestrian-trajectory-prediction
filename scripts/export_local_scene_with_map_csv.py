import os
import csv
import math
import argparse

import numpy as np
import open3d as o3d


LABEL_MAP = {
    (128, 64, 128): "road",
    (244, 35, 232): "sidewalk",
    (81, 0, 81): "ground",
    (150, 100, 100): "curb",
    (157, 234, 50): "road_line",
    (229, 165, 10): "crosswalk",
    (98, 160, 234): "bikelane",
    (128, 128, 128): "unknown",
}


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def choose_sample(rows, sample_id):
    selected = [r for r in rows if r["sample_id"] == sample_id]
    if not selected:
        raise ValueError(f"Δεν βρέθηκε sample_id={sample_id}")
    return selected


def rotation_global_to_local(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=float)


def load_ply_points(ply_path):
    pcd = o3d.io.read_point_cloud(os.path.expanduser(ply_path))
    pts = np.asarray(pcd.points)
    cols = (np.asarray(pcd.colors) * 255).astype(int)
    return pts, cols


def rgb_to_label(rgb):
    return LABEL_MAP.get(tuple(int(v) for v in rgb), "unknown")


def main():
    parser = argparse.ArgumentParser(description="Export pedestrian local scene + semantic map to CSV")
    parser.add_argument(
        "--csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map.csv"),
    )
    parser.add_argument(
        "--ply",
        default=os.path.expanduser("~/imptc_project/ground_plane/xung_ground_plane_02.ply"),
    )
    parser.add_argument("--sample-id", default="0004")
    parser.add_argument("--map-radius", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/imptc_project/results/local_scene_with_map.csv"),
    )
    args = parser.parse_args()

    rows = load_csv_rows(args.csv)
    sample_rows = choose_sample(rows, args.sample_id)
    sample_rows.sort(key=lambda r: safe_float(r.get("dist_xy", 1e9)))
    sample_rows = sample_rows[:args.top_k]

    # Target pedestrian A
    ax_g = safe_float(sample_rows[0]["ax"])
    ay_g = safe_float(sample_rows[0]["ay"])
    az_g = safe_float(sample_rows[0].get("az", 0.0))
    theta_a = safe_float(sample_rows[0]["theta_a"])
    label_a = sample_rows[0].get("label_a", "unknown")
    scene_path = sample_rows[0].get("scene_path", "")
    target_id = sample_rows[0].get("target_id", "")

    R = rotation_global_to_local(theta_a)

    out_rows = []

    # 1) Target row
    out_rows.append({
        "sample_id": args.sample_id,
        "scene_path": scene_path,
        "target_id": target_id,
        "source": "target",
        "entity_id": target_id,
        "entity_class": "person",
        "map_label": label_a,
        "global_x": ax_g,
        "global_y": ay_g,
        "global_z": az_g,
        "local_x": 0.0,
        "local_y": 0.0,
        "local_vx": 0.0,
        "local_vy": 0.0,
        "theta_a": theta_a,
        "dist_to_target": 0.0,
        "interaction_zone": "self",
        "motion_relation": "self",
        "direction_relation": "self",
        "rgb_r": "",
        "rgb_g": "",
        "rgb_b": "",
    })

    # 2) Other agents
    for r in sample_rows:
        out_rows.append({
            "sample_id": r["sample_id"],
            "scene_path": r["scene_path"],
            "target_id": r["target_id"],
            "source": "agent",
            "entity_id": r.get("other_id", ""),
            "entity_class": r.get("other_class_name", "unknown"),
            "map_label": r.get("label_b", "unknown"),
            "global_x": safe_float(r.get("bx", 0.0)),
            "global_y": safe_float(r.get("by", 0.0)),
            "global_z": safe_float(r.get("bz", 0.0)),
            "local_x": safe_float(r.get("rel_x_local", 0.0)),
            "local_y": safe_float(r.get("rel_y_local", 0.0)),
            "local_vx": safe_float(r.get("rel_vx_local", 0.0)),
            "local_vy": safe_float(r.get("rel_vy_local", 0.0)),
            "theta_a": theta_a,
            "dist_to_target": safe_float(r.get("dist_xy", 0.0)),
            "interaction_zone": r.get("interaction_zone", ""),
            "motion_relation": r.get("motion_relation", ""),
            "direction_relation": r.get("direction_relation", ""),
            "rgb_r": "",
            "rgb_g": "",
            "rgb_b": "",
        })

    # 3) Map points from PLY
    print("Loading PLY...")
    pts, cols = load_ply_points(args.ply)
    print("PLY points:", len(pts))

    dx = pts[:, 0] - ax_g
    dy = pts[:, 1] - ay_g
    d2 = dx * dx + dy * dy
    keep = d2 <= args.map_radius * args.map_radius

    kept_pts = pts[keep]
    kept_cols = cols[keep]

    local_xy = np.stack([kept_pts[:, 0] - ax_g, kept_pts[:, 1] - ay_g], axis=1)
    local_xy = (R @ local_xy.T).T

    for p, rgb, lp in zip(kept_pts, kept_cols, local_xy):
        label = rgb_to_label(rgb)
        out_rows.append({
            "sample_id": args.sample_id,
            "scene_path": scene_path,
            "target_id": target_id,
            "source": "map",
            "entity_id": "",
            "entity_class": "",
            "map_label": label,
            "global_x": float(p[0]),
            "global_y": float(p[1]),
            "global_z": float(p[2]),
            "local_x": float(lp[0]),
            "local_y": float(lp[1]),
            "local_vx": "",
            "local_vy": "",
            "theta_a": theta_a,
            "dist_to_target": float(math.sqrt((p[0] - ax_g) ** 2 + (p[1] - ay_g) ** 2)),
            "interaction_zone": "",
            "motion_relation": "",
            "direction_relation": "",
            "rgb_r": int(rgb[0]),
            "rgb_g": int(rgb[1]),
            "rgb_b": int(rgb[2]),
        })

    fieldnames = [
        "sample_id",
        "scene_path",
        "target_id",
        "source",
        "entity_id",
        "entity_class",
        "map_label",
        "global_x",
        "global_y",
        "global_z",
        "local_x",
        "local_y",
        "local_vx",
        "local_vy",
        "theta_a",
        "dist_to_target",
        "interaction_zone",
        "motion_relation",
        "direction_relation",
        "rgb_r",
        "rgb_g",
        "rgb_b",
    ]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("Saved CSV:", args.out)
    print("Rows:", len(out_rows))


if __name__ == "__main__":
    main()

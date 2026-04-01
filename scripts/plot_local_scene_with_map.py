import os
import csv
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
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

LABEL_COLORS = {
    "road": "#808080",
    "sidewalk": "#f4b6d2",
    "ground": "#5a2a83",
    "curb": "#966464",
    "road_line": "#9dea32",
    "crosswalk": "#e5a50a",
    "bikelane": "#62a0ea",
    "unknown": "#b0b0b0",
}

CLASS_MARKERS = {
    "person": "o",
    "scooter": "^",
    "bicycle": "P",
    "motorcycle": "X",
    "stroller": "D",
    "car": "s",
    "truck": "s",
    "bus": "s",
    "unknown": "x",
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
    parser = argparse.ArgumentParser(description="Plot pedestrian local scene together with semantic map from PLY")
    parser.add_argument(
        "--csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map.csv"),
    )
    parser.add_argument(
        "--ply",
        default=os.path.expanduser("~/imptc_project/ground_plane/xung_ground_plane_02.ply"),
    )
    parser.add_argument("--sample-id", default="0004")
    parser.add_argument("--map-radius", type=float, default=12.0, help="meters around pedestrian in global frame")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--show-velocity", action="store_true")
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/imptc_project/results/local_scene_with_map.png"),
    )
    args = parser.parse_args()

    rows = load_csv_rows(args.csv)
    sample_rows = choose_sample(rows, args.sample_id)

    sample_rows.sort(key=lambda r: safe_float(r.get("dist_xy", 1e9)))
    sample_rows = sample_rows[:args.top_k]

    # pedestrian A
    ax_g = safe_float(sample_rows[0]["ax"])
    ay_g = safe_float(sample_rows[0]["ay"])
    theta_a = safe_float(sample_rows[0]["theta_a"])
    label_a = sample_rows[0].get("label_a", "unknown")
    scene_path = sample_rows[0].get("scene_path", "")
    target_id = sample_rows[0].get("target_id", "")

    R = rotation_global_to_local(theta_a)

    # Load map
    print("Loading PLY...")
    pts, cols = load_ply_points(args.ply)
    print("PLY points:", len(pts))

    # κρατάμε μόνο points κοντά στον pedestrian
    dx = pts[:, 0] - ax_g
    dy = pts[:, 1] - ay_g
    d2 = dx * dx + dy * dy
    keep = d2 <= args.map_radius * args.map_radius

    local_pts = np.stack([dx[keep], dy[keep]], axis=1)
    local_pts = (R @ local_pts.T).T
    local_cols = cols[keep]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # 1) semantic map background
    labels_present = {}
    for rgb in np.unique(local_cols, axis=0):
        rgb_t = tuple(int(v) for v in rgb)
        label = rgb_to_label(rgb_t)
        mask = np.all(local_cols == rgb, axis=1)
        pts_l = local_pts[mask]

        ax.scatter(
            pts_l[:, 0],
            pts_l[:, 1],
            s=4,
            c=LABEL_COLORS.get(label, "#b0b0b0"),
            alpha=0.35,
            linewidths=0,
            label=label if label not in labels_present else None,
            zorder=1,
        )
        labels_present[label] = True

    # 2) pedestrian target at origin
    ax.scatter(
        [0], [0],
        s=260,
        marker="*",
        c=["gold"],
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
        label=f"target pedestrian ({label_a})",
    )
    ax.text(0.2, 0.2, "A", fontsize=12, weight="bold", zorder=11)

    # 3) other agents
    used_agent_legend = set()
    for r in sample_rows:
        x = safe_float(r["rel_x_local"])
        y = safe_float(r["rel_y_local"])
        vx = safe_float(r.get("rel_vx_local", 0.0))
        vy = safe_float(r.get("rel_vy_local", 0.0))

        other_id = r.get("other_id", "?")
        other_class = r.get("other_class_name", "unknown") or "unknown"
        label_b = r.get("label_b", "unknown") or "unknown"

        marker = CLASS_MARKERS.get(other_class, "x")
        color = LABEL_COLORS.get(label_b, "#00cccc")

        leg_key = (other_class, label_b)
        leg_label = None
        if leg_key not in used_agent_legend:
            leg_label = f"{other_class} on {label_b}"
            used_agent_legend.add(leg_key)

        ax.scatter(
            [x], [y],
            s=110,
            marker=marker,
            c=[color],
            edgecolors="black",
            linewidths=0.6,
            zorder=20,
            label=leg_label,
        )

        ax.text(x + 0.10, y + 0.10, f"{other_id}", fontsize=8, zorder=21)

        if args.show_velocity:
            ax.arrow(
                x, y, vx, vy,
                length_includes_head=True,
                head_width=0.15,
                head_length=0.22,
                alpha=0.8,
                zorder=19,
            )

    # axes
    ax.axhline(0, linewidth=1.2, color="steelblue", zorder=5)
    ax.axvline(0, linewidth=1.2, color="steelblue", zorder=5)

    ax.annotate("front", xy=(2.0, 0.0), xytext=(3.0, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    ax.annotate("rear", xy=(-2.0, 0.0), xytext=(-3.0, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    ax.annotate("left", xy=(0.0, 2.0), xytext=(0.0, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    ax.annotate("right", xy=(0.0, -2.0), xytext=(0.0, -3.0),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)

    ax.set_xlabel("local x relative to pedestrian A")
    ax.set_ylabel("local y relative to pedestrian A")
    ax.set_title(
        f"Local scene with semantic map\nsample {args.sample_id}, scene {scene_path}, target {target_id}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

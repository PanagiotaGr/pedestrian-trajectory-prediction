import os
import csv
import json
import math
import tarfile
import argparse
from collections import defaultdict

EPS = 1e-9


# ════════════════════════════════════════
# ΒΟΗΘΗΤΙΚΑ
# ════════════════════════════════════════

def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def norm2(x, y):
    return math.sqrt(x * x + y * y)


def heading_angle(vx, vy):
    if abs(vx) < EPS and abs(vy) < EPS:
        return 0.0
    return math.atan2(vy, vx)


def rotation_global_to_local(theta):
    # Global -> local pedestrian frame
    c = math.cos(theta)
    s = math.sin(theta)
    return ((c, s), (-s, c))


def apply_rot(R, x, y):
    return (
        R[0][0] * x + R[0][1] * y,
        R[1][0] * x + R[1][1] * y,
    )


# ════════════════════════════════════════
# DERIVED FEATURES
# ════════════════════════════════════════

def interaction_zone(theta_local):
    if -math.pi / 4 <= theta_local <= math.pi / 4:
        return "front"
    if math.pi / 4 < theta_local < 3 * math.pi / 4:
        return "left_side"
    if -3 * math.pi / 4 < theta_local < -math.pi / 4:
        return "right_side"
    return "rear"


def motion_relation(closing_speed, eps=0.2):
    if closing_speed > eps:
        return "approaching"
    if closing_speed < -eps:
        return "receding"
    return "stable"


def direction_relation(heading_diff):
    a = abs(heading_diff)
    if a < math.pi / 4:
        return "same_direction"
    if a > 3 * math.pi / 4:
        return "opposite_direction"
    return "crossing_direction"


# ════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════

def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def open_tar_auto(path):
    return tarfile.open(path, "r:*")


def load_scene_tracks_from_archive(archive_path, scene_path):
    tracks = {}
    prefix = scene_path.rstrip("/") + "/"

    with open_tar_auto(archive_path) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.startswith(prefix):
                continue
            if not member.name.endswith("/track.json"):
                continue

            parts = member.name.split("/")
            if len(parts) < 4:
                continue

            # scene/track_type/track_id/track.json
            track_type = parts[1]
            track_id = parts[2]

            f = tar.extractfile(member)
            if f is None:
                continue

            try:
                obj = json.load(f)
            except Exception:
                continue

            data = obj.get("track_data", {})
            overview = obj.get("overview", {})

            points = []
            for _, v in data.items():
                ts = int(v["ts"])
                coords = v["coordinates"]
                points.append({
                    "ts": ts,
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]),
                })

            points.sort(key=lambda p: p["ts"])
            ts_to_idx = {p["ts"]: i for i, p in enumerate(points)}

            tracks[(track_type, track_id)] = {
                "points": points,
                "ts_to_idx": ts_to_idx,
                "overview": overview,
                "member_path": member.name,
            }

    return tracks


def estimate_velocity_xy(points, idx):
    n = len(points)
    if n < 2:
        return 0.0, 0.0

    if 0 < idx < n - 1:
        p0 = points[idx - 1]
        p1 = points[idx + 1]
    elif idx == 0:
        p0 = points[idx]
        p1 = points[idx + 1]
    else:
        p0 = points[idx - 1]
        p1 = points[idx]

    dt = (p1["ts"] - p0["ts"]) / 1_000_000.0
    if abs(dt) < EPS:
        return 0.0, 0.0

    vx = (p1["x"] - p0["x"]) / dt
    vy = (p1["y"] - p0["y"]) / dt
    return vx, vy


def find_min_distance_timestamp(track_a, track_b):
    common = sorted(set(track_a["ts_to_idx"]).intersection(track_b["ts_to_idx"]))
    if not common:
        return None

    best_ts = None
    best_d2 = None

    for ts in common:
        ia = track_a["ts_to_idx"][ts]
        ib = track_b["ts_to_idx"][ts]

        pa = track_a["points"][ia]
        pb = track_b["points"][ib]

        dx = pb["x"] - pa["x"]
        dy = pb["y"] - pa["y"]
        d2 = dx * dx + dy * dy

        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_ts = ts

    return best_ts


# ════════════════════════════════════════
# MAIN FEATURE BUILDER
# ════════════════════════════════════════

def build_row(base_row, track_a, track_b, ts):
    ia = track_a["ts_to_idx"][ts]
    ib = track_b["ts_to_idx"][ts]

    pa = track_a["points"][ia]
    pb = track_b["points"][ib]

    avx, avy = estimate_velocity_xy(track_a["points"], ia)
    bvx, bvy = estimate_velocity_xy(track_b["points"], ib)

    theta_a = heading_angle(avx, avy)
    theta_b = heading_angle(bvx, bvy)

    Rga = rotation_global_to_local(theta_a)

    dx = pb["x"] - pa["x"]
    dy = pb["y"] - pa["y"]

    dvx = bvx - avx
    dvy = bvy - avy

    rel_x_local, rel_y_local = apply_rot(Rga, dx, dy)
    rel_vx_local, rel_vy_local = apply_rot(Rga, dvx, dvy)

    dist_xy = norm2(dx, dy)
    theta_local = math.atan2(rel_y_local, rel_x_local)
    heading_diff = wrap_angle(theta_b - theta_a)

    if dist_xy > EPS:
        ux = dx / dist_xy
        uy = dy / dist_xy
        closing_speed = -(dvx * ux + dvy * uy)
    else:
        closing_speed = 0.0

    zone = interaction_zone(theta_local)
    motion_rel = motion_relation(closing_speed)
    dir_rel = direction_relation(heading_diff)

    return {
        # keep original identifiers
        "sample_id": base_row.get("sample_id", ""),
        "src_info": base_row.get("src_info", ""),
        "archive": base_row.get("archive", ""),
        "scene_path": base_row.get("scene_path", ""),
        "target_type": base_row.get("target_type", ""),
        "target_id": base_row.get("target_id", ""),
        "other_type": base_row.get("other_type", ""),
        "other_id": base_row.get("other_id", ""),
        "target_class_name": base_row.get("target_class_name", ""),
        "other_class_name": base_row.get("other_class_name", ""),
        "ts": str(ts),

        # global positions
        "ax": pa["x"],
        "ay": pa["y"],
        "az": pa["z"],
        "bx": pb["x"],
        "by": pb["y"],
        "bz": pb["z"],

        # global velocities
        "avx": avx,
        "avy": avy,
        "bvx": bvx,
        "bvy": bvy,

        # headings
        "theta_a": theta_a,
        "theta_b": theta_b,

        # relative geometry in local frame of A
        "rel_x_local": rel_x_local,
        "rel_y_local": rel_y_local,
        "rel_vx_local": rel_vx_local,
        "rel_vy_local": rel_vy_local,

        # metrics
        "dist_xy": dist_xy,
        "theta_local": theta_local,
        "heading_diff": heading_diff,
        "closing_speed": closing_speed,

        # derived labels
        "interaction_zone": zone,
        "motion_relation": motion_rel,
        "direction_relation": dir_rel,
    }


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build pedestrian mathematical interaction dataset")
    parser.add_argument(
        "--detailed-csv",
        default=os.path.expanduser("~/imptc_project/results/interactions_detailed.csv"),
    )
    parser.add_argument(
        "--archives-dir",
        default=os.path.expanduser("~/imptc_project/data"),
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_dataset_v2.csv"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = όλα")
    args = parser.parse_args()

    rows = load_csv_rows(args.detailed_csv)

    # κρατάμε μόνο pedestrian targets
    rows = [
        r for r in rows
        if r.get("target_type") == "vrus"
        and r.get("target_class_name") == "person"
    ]

    if args.limit > 0:
        rows = rows[:args.limit]

    print("Loaded pedestrian interaction rows:", len(rows))

    rows_by_scene = defaultdict(list)
    for r in rows:
        rows_by_scene[(r.get("archive", ""), r.get("scene_path", ""))].append(r)

    out_rows = []

    for (archive_name, scene_path), scene_rows in rows_by_scene.items():
        archive_path = os.path.join(args.archives_dir, archive_name)

        print(f"[SCENE] {scene_path} ({archive_name}) rows={len(scene_rows)}")

        if not os.path.exists(archive_path):
            print(f"  [WARN] Archive not found: {archive_path}")
            continue

        tracks = load_scene_tracks_from_archive(archive_path, scene_path)

        for r in scene_rows:
            key_a = (r.get("target_type", ""), r.get("target_id", ""))
            key_b = (r.get("other_type", ""), r.get("other_id", ""))

            if key_a not in tracks or key_b not in tracks:
                continue

            ts = find_min_distance_timestamp(tracks[key_a], tracks[key_b])
            if ts is None:
                continue

            out_rows.append(build_row(r, tracks[key_a], tracks[key_b], ts))

    if not out_rows:
        print("No rows produced.")
        return

    fieldnames = list(out_rows[0].keys())

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nDONE")
    print("Saved:", args.out_csv)
    print("Rows:", len(out_rows))


if __name__ == "__main__":
    main()

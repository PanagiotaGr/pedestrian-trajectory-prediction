import os
import csv
import json
import math
import tarfile
import argparse
from collections import defaultdict


EPS = 1e-9


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def open_tar_auto(path):
    # υποστηρίζει .tar.gz και .tar
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

            track_type = parts[1]
            track_id = parts[2]

            f = tar.extractfile(member)
            if f is None:
                continue

            try:
                obj = json.load(f)
            except Exception:
                continue

            overview = obj.get("overview", {})
            data = obj.get("track_data", {})

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
                "overview": overview,
                "points": points,
                "ts_to_idx": ts_to_idx,
                "member_path": member.name,
            }

    return tracks


def estimate_velocity_xy(points, idx):
    """
    Υπολογισμός vx, vy από γειτονικά trajectory points.
    dt σε δευτερόλεπτα, αφού τα ts είναι σε microseconds.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0.0

    if n == 1:
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


def norm2(x, y):
    return math.sqrt(x * x + y * y)


def rotation_global_to_local(heading):
    """
    Μετασχηματισμός από global frame -> local frame του agent A.
    Local x-axis = heading direction του A.
    """
    c = math.cos(heading)
    s = math.sin(heading)
    return ((c, s), (-s, c))


def rotation_local_to_global(heading):
    c = math.cos(heading)
    s = math.sin(heading)
    return ((c, -s), (s, c))


def matmul2(A, B):
    return (
        (
            A[0][0] * B[0][0] + A[0][1] * B[1][0],
            A[0][0] * B[0][1] + A[0][1] * B[1][1],
        ),
        (
            A[1][0] * B[0][0] + A[1][1] * B[1][0],
            A[1][0] * B[0][1] + A[1][1] * B[1][1],
        ),
    )


def apply_rot(R, x, y):
    return (
        R[0][0] * x + R[0][1] * y,
        R[1][0] * x + R[1][1] * y,
    )


def find_min_distance_timestamp(track_a, track_b):
    """
    Βρίσκει το κοινό ts με ελάχιστη απόσταση στο xy.
    """
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


def build_row(base_row, track_a, track_b, ts):
    ia = track_a["ts_to_idx"][ts]
    ib = track_b["ts_to_idx"][ts]

    pa = track_a["points"][ia]
    pb = track_b["points"][ib]

    avx, avy = estimate_velocity_xy(track_a["points"], ia)
    bvx, bvy = estimate_velocity_xy(track_b["points"], ib)

    dx = pb["x"] - pa["x"]
    dy = pb["y"] - pa["y"]
    dist_xy = norm2(dx, dy)

    dvx = bvx - avx
    dvy = bvy - avy

    speed_a = norm2(avx, avy)
    speed_b = norm2(bvx, bvy)

    if speed_a > EPS:
        heading_a = math.atan2(avy, avx)
        heading_a_source = "velocity"
    else:
        heading_a = 0.0
        heading_a_source = "fallback_zero"

    if speed_b > EPS:
        heading_b = math.atan2(bvy, bvx)
        heading_b_source = "velocity"
    else:
        heading_b = 0.0
        heading_b_source = "fallback_zero"

    theta_global = math.atan2(dy, dx)

    Rga = rotation_global_to_local(heading_a)
    rel_x_local, rel_y_local = apply_rot(Rga, dx, dy)
    rel_vx_local, rel_vy_local = apply_rot(Rga, dvx, dvy)
    theta_local = math.atan2(rel_y_local, rel_x_local)

    heading_diff = wrap_angle(heading_b - heading_a)

    Rbg = rotation_local_to_global(heading_b)
    Rba = matmul2(Rga, Rbg)

    # approaching / closing speed
    if dist_xy > EPS:
        ux = dx / dist_xy
        uy = dy / dist_xy
        closing_speed = -(dvx * ux + dvy * uy)
    else:
        closing_speed = 0.0

    return {
        "sample_id": base_row["sample_id"],
        "src_info": base_row["src_info"],
        "archive": base_row["archive"],
        "scene_path": base_row["scene_path"],
        "target_type": base_row["target_type"],
        "target_id": base_row["target_id"],
        "other_type": base_row["other_type"],
        "other_id": base_row["other_id"],
        "target_class_name": base_row.get("target_class_name", ""),
        "other_class_name": base_row.get("other_class_name", ""),
        "ts_min_dist": str(ts),

        "ax": pa["x"],
        "ay": pa["y"],
        "az": pa["z"],
        "bx": pb["x"],
        "by": pb["y"],
        "bz": pb["z"],

        "avx": avx,
        "avy": avy,
        "bvx": bvx,
        "bvy": bvy,

        "speed_a": speed_a,
        "speed_b": speed_b,

        "dx_global": dx,
        "dy_global": dy,
        "dvx_global": dvx,
        "dvy_global": dvy,
        "dist_xy": dist_xy,
        "theta_global": theta_global,

        "rel_x_local": rel_x_local,
        "rel_y_local": rel_y_local,
        "rel_vx_local": rel_vx_local,
        "rel_vy_local": rel_vy_local,
        "theta_local": theta_local,
        "closing_speed": closing_speed,

        "heading_a": heading_a,
        "heading_b": heading_b,
        "heading_diff": heading_diff,
        "heading_a_source": heading_a_source,
        "heading_b_source": heading_b_source,

        "Rga_11": Rga[0][0],
        "Rga_12": Rga[0][1],
        "Rga_21": Rga[1][0],
        "Rga_22": Rga[1][1],

        "Rba_11": Rba[0][0],
        "Rba_12": Rba[0][1],
        "Rba_21": Rba[1][0],
        "Rba_22": Rba[1][1],
    }


def main():
    parser = argparse.ArgumentParser(description="Build relative geometry dataset for all interactions")
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
        default=os.path.expanduser("~/imptc_project/results/relative_geometry_detailed.csv"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = όλα",
    )
    args = parser.parse_args()

    rows = load_csv_rows(args.detailed_csv)
    if args.limit > 0:
        rows = rows[:args.limit]

    print("Loaded detailed interaction rows:", len(rows))

    rows_by_scene = defaultdict(list)
    for r in rows:
        key = (r["archive"], r["scene_path"])
        rows_by_scene[key].append(r)

    out_rows = []

    for (archive_name, scene_path), scene_rows in rows_by_scene.items():
        archive_path = os.path.join(args.archives_dir, archive_name)
        print(f"[SCENE] {scene_path} ({archive_name}) rows={len(scene_rows)}")

        if not os.path.exists(archive_path):
            print("  archive missing:", archive_path)
            continue

        tracks = load_scene_tracks_from_archive(archive_path, scene_path)

        for r in scene_rows:
            key_a = (r["target_type"], r["target_id"])
            key_b = (r["other_type"], r["other_id"])

            if key_a not in tracks or key_b not in tracks:
                continue

            track_a = tracks[key_a]
            track_b = tracks[key_b]

            ts = find_min_distance_timestamp(track_a, track_b)
            if ts is None:
                continue

            out_rows.append(build_row(r, track_a, track_b, ts))

    fieldnames = [
        "sample_id", "src_info", "archive", "scene_path",
        "target_type", "target_id", "other_type", "other_id",
        "target_class_name", "other_class_name", "ts_min_dist",
        "ax", "ay", "az", "bx", "by", "bz",
        "avx", "avy", "bvx", "bvy",
        "speed_a", "speed_b",
        "dx_global", "dy_global", "dvx_global", "dvy_global",
        "dist_xy", "theta_global",
        "rel_x_local", "rel_y_local", "rel_vx_local", "rel_vy_local",
        "theta_local", "closing_speed",
        "heading_a", "heading_b", "heading_diff",
        "heading_a_source", "heading_b_source",
        "Rga_11", "Rga_12", "Rga_21", "Rga_22",
        "Rba_11", "Rba_12", "Rba_21", "Rba_22",
    ]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print()
    print("DONE")
    print("Saved:", args.out_csv)
    print("Rows:", len(out_rows))


if __name__ == "__main__":
    main()

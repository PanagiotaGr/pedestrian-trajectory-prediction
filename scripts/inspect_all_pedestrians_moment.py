import os
import csv
import json
import tarfile
import math
import argparse
from collections import defaultdict

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_scene_tracks_from_archive(archive_path, scene_path):
    tracks = {}
    prefix = scene_path.rstrip("/") + "/"

    with tarfile.open(archive_path, "r:gz") as tar:
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

            ts_map = {}
            for _, v in data.items():
                ts = str(v["ts"])
                coords = v["coordinates"]
                ts_map[ts] = {
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]),
                    "velocity": v.get("velocity"),
                    "class_prob": v.get("class_prob"),
                    "status": v.get("status"),
                    "source_type": v.get("source_type"),
                    "ground_type": v.get("ground_type"),
                }

            tracks[(track_type, track_id)] = {
                "overview": overview,
                "ts_map": ts_map,
                "member_path": member.name,
            }

    return tracks

def choose_timestamp(ts_sorted, mode):
    if not ts_sorted:
        return None
    if mode == "start":
        return ts_sorted[0]
    if mode == "end":
        return ts_sorted[-1]
    return ts_sorted[len(ts_sorted) // 2]  # middle

def main():
    parser = argparse.ArgumentParser(description="Inspect one moment for all pedestrian samples")
    parser.add_argument("--summary-csv", default=os.path.expanduser("~/imptc_project/results/interactions_summary.csv"))
    parser.add_argument("--archives-dir", default=os.path.expanduser("~/imptc_project/data"))
    parser.add_argument("--moment", choices=["start", "middle", "end"], default="middle")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="0 = όλα")
    parser.add_argument("--out-summary", default=os.path.expanduser("~/imptc_project/results/pedestrian_moments_summary.csv"))
    parser.add_argument("--out-neighbors", default=os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv"))
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)

    # κρατάμε μόνο matched pedestrians
    ped_rows = [
        r for r in rows
        if r.get("status") == "ok"
        and r.get("target_type") == "vrus"
        and r.get("target_class_name") == "person"
    ]

    if args.limit > 0:
        ped_rows = ped_rows[:args.limit]

    print(f"Pedestrian samples to inspect: {len(ped_rows)}")

    grouped = defaultdict(list)
    for r in ped_rows:
        grouped[(r["archive"], r["scene_path"])].append(r)

    moment_summary = []
    neighbor_rows = []

    for (archive_name, scene_path), target_rows in grouped.items():
        archive_path = os.path.join(args.archives_dir, archive_name)
        if not os.path.isfile(archive_path):
            continue

        print(f"[SCENE] {scene_path} ({archive_name})")
        scene_tracks = load_scene_tracks_from_archive(archive_path, scene_path)

        for row in target_rows:
            sample_id = row["sample_id"]
            target_key = (row["target_type"], row["target_id"])

            if target_key not in scene_tracks:
                moment_summary.append({
                    "sample_id": sample_id,
                    "scene_path": scene_path,
                    "archive": archive_name,
                    "target_type": row["target_type"],
                    "target_id": row["target_id"],
                    "target_class_name": row["target_class_name"],
                    "moment_mode": args.moment,
                    "timestamp": "",
                    "target_x": "",
                    "target_y": "",
                    "target_z": "",
                    "target_velocity": "",
                    "n_neighbors_found": 0,
                    "nearest_type": "",
                    "nearest_id": "",
                    "nearest_class": "",
                    "nearest_dist": "",
                    "status": "target_missing",
                })
                continue

            target = scene_tracks[target_key]
            ts_sorted = sorted(target["ts_map"].keys(), key=int)
            ts = choose_timestamp(ts_sorted, args.moment)
            if ts is None or ts not in target["ts_map"]:
                moment_summary.append({
                    "sample_id": sample_id,
                    "scene_path": scene_path,
                    "archive": archive_name,
                    "target_type": row["target_type"],
                    "target_id": row["target_id"],
                    "target_class_name": row["target_class_name"],
                    "moment_mode": args.moment,
                    "timestamp": "",
                    "target_x": "",
                    "target_y": "",
                    "target_z": "",
                    "target_velocity": "",
                    "n_neighbors_found": 0,
                    "nearest_type": "",
                    "nearest_id": "",
                    "nearest_class": "",
                    "nearest_dist": "",
                    "status": "timestamp_missing",
                })
                continue

            tx = target["ts_map"][ts]["x"]
            ty = target["ts_map"][ts]["y"]
            tz = target["ts_map"][ts]["z"]
            tv = target["ts_map"][ts].get("velocity")

            neighbors = []
            for (otype, oid), other in scene_tracks.items():
                if (otype, oid) == target_key:
                    continue
                if ts not in other["ts_map"]:
                    continue

                ox = other["ts_map"][ts]["x"]
                oy = other["ts_map"][ts]["y"]
                oz = other["ts_map"][ts]["z"]
                ov = other["ts_map"][ts].get("velocity")

                d = math.sqrt((ox - tx) ** 2 + (oy - ty) ** 2)

                neighbors.append({
                    "sample_id": sample_id,
                    "scene_path": scene_path,
                    "archive": archive_name,
                    "moment_mode": args.moment,
                    "timestamp": ts,
                    "target_type": row["target_type"],
                    "target_id": row["target_id"],
                    "target_class_name": row["target_class_name"],
                    "target_x": tx,
                    "target_y": ty,
                    "target_z": tz,
                    "target_velocity": tv,
                    "other_type": otype,
                    "other_id": oid,
                    "other_class_name": other["overview"].get("class_name", ""),
                    "other_x": ox,
                    "other_y": oy,
                    "other_z": oz,
                    "other_velocity": ov,
                    "distance": round(d, 6),
                    "other_member_path": other["member_path"],
                })

            neighbors.sort(key=lambda r: r["distance"])
            top_neighbors = neighbors[:args.top_k]

            nearest = top_neighbors[0] if top_neighbors else None

            moment_summary.append({
                "sample_id": sample_id,
                "scene_path": scene_path,
                "archive": archive_name,
                "target_type": row["target_type"],
                "target_id": row["target_id"],
                "target_class_name": row["target_class_name"],
                "moment_mode": args.moment,
                "timestamp": ts,
                "target_x": tx,
                "target_y": ty,
                "target_z": tz,
                "target_velocity": tv,
                "n_neighbors_found": len(top_neighbors),
                "nearest_type": nearest["other_type"] if nearest else "",
                "nearest_id": nearest["other_id"] if nearest else "",
                "nearest_class": nearest["other_class_name"] if nearest else "",
                "nearest_dist": nearest["distance"] if nearest else "",
                "status": "ok",
            })

            neighbor_rows.extend(top_neighbors)

    if moment_summary:
        with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(moment_summary[0].keys()))
            writer.writeheader()
            writer.writerows(moment_summary)

    if neighbor_rows:
        with open(args.out_neighbors, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(neighbor_rows[0].keys()))
            writer.writeheader()
            writer.writerows(neighbor_rows)

    print("DONE")
    print(f"Saved summary  : {args.out_summary}")
    print(f"Saved neighbors: {args.out_neighbors}")
    print(f"Rows summary   : {len(moment_summary)}")
    print(f"Rows neighbors : {len(neighbor_rows)}")

if __name__ == "__main__":
    main()

import os
import csv
import json
import tarfile
import math
import argparse
from collections import defaultdict

def load_matched_rows(matched_csv):
    rows = []
    with open(matched_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mc = int(row.get("match_count_for_sample", "0"))
            except ValueError:
                mc = 0
            if mc > 0:
                rows.append(row)
    return rows

def load_scene_tracks_from_archive(archive_path, scene_path):
    """
    Διαβάζει όλα τα track.json του scene κατευθείαν από το tar.gz χωρίς extract στο disk.
    Επιστρέφει dict:
      {
        ("vrus","000"): {"overview":..., "ts_map":..., "member_path":...},
        ("vehicles","1000"): {...}
      }
    """
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
            # scene_path / vrus|vehicles / track_id / track.json
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
                    "class_prob": v.get("class_prob"),
                    "velocity": v.get("velocity"),
                    "source_type": v.get("source_type"),
                    "ground_type": v.get("ground_type"),
                    "status": v.get("status"),
                }

            tracks[(track_type, track_id)] = {
                "overview": overview,
                "ts_map": ts_map,
                "member_path": member.name,
            }

    return tracks

def compute_interactions_for_target(scene_tracks, target_type, target_id, radius):
    target_key = (target_type, target_id)
    if target_key not in scene_tracks:
        return []

    target = scene_tracks[target_key]
    target_ts = target["ts_map"]

    results = []

    for other_key, other in scene_tracks.items():
        if other_key == target_key:
            continue

        common_ts = set(target_ts.keys()) & set(other["ts_map"].keys())
        if not common_ts:
            continue

        min_dist = float("inf")
        close_frames = 0

        for ts in common_ts:
            tx = target_ts[ts]["x"]
            ty = target_ts[ts]["y"]
            ox = other["ts_map"][ts]["x"]
            oy = other["ts_map"][ts]["y"]

            d = math.sqrt((ox - tx) ** 2 + (oy - ty) ** 2)

            if d < min_dist:
                min_dist = d
            if d < radius:
                close_frames += 1

        if close_frames > 0:
            results.append({
                "other_type": other_key[0],
                "other_id": other_key[1],
                "common_timestamps": len(common_ts),
                "close_frames": close_frames,
                "min_dist": round(min_dist, 6),
                "other_class_name": other["overview"].get("class_name", ""),
                "other_first_ts": other["overview"].get("first_ts", ""),
                "other_last_ts": other["overview"].get("last_ts", ""),
                "other_length": other["overview"].get("lenght", ""),
                "other_duration": other["overview"].get("duration", ""),
                "other_member_path": other["member_path"],
            })

    results.sort(key=lambda r: (-r["close_frames"], r["min_dist"]))
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch interaction analysis from matched_codes.csv")
    parser.add_argument("--matched-csv", default=os.path.expanduser("~/imptc_project/results/matched_codes.csv"))
    parser.add_argument("--archives-dir", default=os.path.expanduser("~/imptc_project/data"))
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0, help="0 = όλα τα matched samples")
    parser.add_argument("--out-detailed", default=os.path.expanduser("~/imptc_project/results/interactions_detailed.csv"))
    parser.add_argument("--out-summary", default=os.path.expanduser("~/imptc_project/results/interactions_summary.csv"))
    args = parser.parse_args()

    matched_rows = load_matched_rows(args.matched_csv)
    if args.limit > 0:
        matched_rows = matched_rows[:args.limit]

    print(f"Loaded matched samples: {len(matched_rows)}")

    # group by archive+scene_path για να ανοίγουμε κάθε scene μία φορά
    grouped = defaultdict(list)
    for row in matched_rows:
        key = (row["archive"], row["scene_path"])
        grouped[key].append(row)

    print(f"Unique matched scenes: {len(grouped)}")

    detailed_rows = []
    summary_rows = []

    processed_scenes = 0
    processed_targets = 0

    for (archive_name, scene_path), target_rows in grouped.items():
        archive_path = os.path.join(args.archives_dir, archive_name)
        if not os.path.isfile(archive_path):
            print(f"[WARN] Archive not found: {archive_path}")
            continue

        print(f"\n[SCENE] {scene_path}  ({archive_name})")
        scene_tracks = load_scene_tracks_from_archive(archive_path, scene_path)

        if not scene_tracks:
            print("  [WARN] No tracks loaded for scene.")
            continue

        processed_scenes += 1

        for row in target_rows:
            sample_id = row["sample_id"]
            target_type = row["track_type"]
            target_id = row["track_id"]

            processed_targets += 1

            target_key = (target_type, target_id)
            if target_key not in scene_tracks:
                print(f"  [WARN] Target missing in scene: sample={sample_id}, target={target_key}")
                summary_rows.append({
                    "sample_id": sample_id,
                    "src_info": row["src_info"],
                    "archive": archive_name,
                    "scene_path": scene_path,
                    "target_type": target_type,
                    "target_id": target_id,
                    "radius": args.radius,
                    "target_found": 0,
                    "n_interacting_tracks": 0,
                    "n_vru_interactions": 0,
                    "n_vehicle_interactions": 0,
                    "strongest_other_type": "",
                    "strongest_other_id": "",
                    "strongest_min_dist": "",
                    "strongest_close_frames": "",
                })
                continue

            target_overview = scene_tracks[target_key]["overview"]
            interactions = compute_interactions_for_target(scene_tracks, target_type, target_id, args.radius)

            n_vrus = sum(1 for x in interactions if x["other_type"] == "vrus")
            n_vehicles = sum(1 for x in interactions if x["other_type"] == "vehicles")

            strongest = interactions[0] if interactions else None

            summary_rows.append({
                "sample_id": sample_id,
                "src_info": row["src_info"],
                "archive": archive_name,
                "scene_path": scene_path,
                "target_type": target_type,
                "target_id": target_id,
                "radius": args.radius,
                "target_found": 1,
                "target_class_name": target_overview.get("class_name", ""),
                "target_first_ts": target_overview.get("first_ts", ""),
                "target_last_ts": target_overview.get("last_ts", ""),
                "target_length": target_overview.get("lenght", ""),
                "target_duration": target_overview.get("duration", ""),
                "n_interacting_tracks": len(interactions),
                "n_vru_interactions": n_vrus,
                "n_vehicle_interactions": n_vehicles,
                "strongest_other_type": strongest["other_type"] if strongest else "",
                "strongest_other_id": strongest["other_id"] if strongest else "",
                "strongest_min_dist": strongest["min_dist"] if strongest else "",
                "strongest_close_frames": strongest["close_frames"] if strongest else "",
            })

            for inter in interactions:
                detailed_rows.append({
                    "sample_id": sample_id,
                    "src_info": row["src_info"],
                    "archive": archive_name,
                    "scene_path": scene_path,
                    "radius": args.radius,
                    "target_type": target_type,
                    "target_id": target_id,
                    "target_class_name": target_overview.get("class_name", ""),
                    "target_first_ts": target_overview.get("first_ts", ""),
                    "target_last_ts": target_overview.get("last_ts", ""),
                    "target_length": target_overview.get("lenght", ""),
                    "target_duration": target_overview.get("duration", ""),
                    **inter
                })

            print(f"  target sample={sample_id} -> interactions={len(interactions)}")

    # save detailed
    if detailed_rows:
        with open(args.out_detailed, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detailed_rows)

    # save summary
    if summary_rows:
        with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\nDONE")
    print(f"Processed scenes   : {processed_scenes}")
    print(f"Processed targets  : {processed_targets}")
    print(f"Detailed rows      : {len(detailed_rows)}")
    print(f"Summary rows       : {len(summary_rows)}")
    print(f"Saved detailed CSV : {args.out_detailed}")
    print(f"Saved summary CSV  : {args.out_summary}")

if __name__ == "__main__":
    main()

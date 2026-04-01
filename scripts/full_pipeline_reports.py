import os
import csv
import json
import tarfile
import math
import argparse
from collections import defaultdict

def load_all_rows(matched_csv):
    rows = []
    with open(matched_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["match_count_for_sample"] = int(row.get("match_count_for_sample", "0"))
            except ValueError:
                row["match_count_for_sample"] = 0
            rows.append(row)
    return rows

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

def write_sample_report(report_dir, row, target_overview, interactions, radius, status):
    os.makedirs(report_dir, exist_ok=True)
    sample_id = row["sample_id"]
    report_path = os.path.join(report_dir, f"{sample_id}_report.txt")

    n_vrus = sum(1 for x in interactions if x["other_type"] == "vrus")
    n_vehicles = sum(1 for x in interactions if x["other_type"] == "vehicles")
    strongest = interactions[0] if interactions else None

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Source info: {row.get('src_info', '')}\n")
        f.write(f"Archive: {row.get('archive', '')}\n")
        f.write(f"Scene path: {row.get('scene_path', '')}\n")
        f.write(f"Target type: {row.get('track_type', '')}\n")
        f.write(f"Target ID: {row.get('track_id', '')}\n")
        f.write(f"Radius: {radius}\n")
        f.write(f"Status: {status}\n\n")

        if target_overview:
            f.write("Target overview\n")
            f.write(f"  class_name: {target_overview.get('class_name', '')}\n")
            f.write(f"  first_ts: {target_overview.get('first_ts', '')}\n")
            f.write(f"  last_ts: {target_overview.get('last_ts', '')}\n")
            f.write(f"  lenght: {target_overview.get('lenght', '')}\n")
            f.write(f"  duration: {target_overview.get('duration', '')}\n\n")

        f.write("Interaction summary\n")
        f.write(f"  total interacting tracks: {len(interactions)}\n")
        f.write(f"  VRU interactions: {n_vrus}\n")
        f.write(f"  vehicle interactions: {n_vehicles}\n")
        if strongest:
            f.write("  strongest neighbor:\n")
            f.write(f"    type: {strongest['other_type']}\n")
            f.write(f"    id: {strongest['other_id']}\n")
            f.write(f"    class: {strongest['other_class_name']}\n")
            f.write(f"    close_frames: {strongest['close_frames']}\n")
            f.write(f"    common_timestamps: {strongest['common_timestamps']}\n")
            f.write(f"    min_dist: {strongest['min_dist']}\n")
        f.write("\n")

        f.write("Detailed interactions\n")
        if not interactions:
            f.write("  No interactions found.\n")
        else:
            for i, inter in enumerate(interactions, start=1):
                f.write(f"{i}. other_type={inter['other_type']}, ")
                f.write(f"other_id={inter['other_id']}, ")
                f.write(f"class={inter['other_class_name']}, ")
                f.write(f"common_timestamps={inter['common_timestamps']}, ")
                f.write(f"close_frames={inter['close_frames']}, ")
                f.write(f"min_dist={inter['min_dist']}\n")

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: summary + detailed + per-sample reports")
    parser.add_argument("--matched-csv", default=os.path.expanduser("~/imptc_project/results/matched_codes.csv"))
    parser.add_argument("--archives-dir", default=os.path.expanduser("~/imptc_project/data"))
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="0 = όλα τα samples")
    parser.add_argument("--out-detailed", default=os.path.expanduser("~/imptc_project/results/interactions_detailed.csv"))
    parser.add_argument("--out-summary", default=os.path.expanduser("~/imptc_project/results/interactions_summary.csv"))
    parser.add_argument("--report-dir", default=os.path.expanduser("~/imptc_project/results/per_sample_reports"))
    args = parser.parse_args()

    all_rows = load_all_rows(args.matched_csv)
    if args.limit > 0:
        all_rows = all_rows[:args.limit]

    print(f"Loaded total samples: {len(all_rows)}")

    matched_rows = [r for r in all_rows if r["match_count_for_sample"] > 0]
    print(f"Matched samples: {len(matched_rows)}")
    print(f"Unmatched samples: {len(all_rows) - len(matched_rows)}")

    grouped = defaultdict(list)
    for row in matched_rows:
        grouped[(row["archive"], row["scene_path"])].append(row)

    detailed_rows = []
    summary_rows = []

    for (archive_name, scene_path), target_rows in grouped.items():
        archive_path = os.path.join(args.archives_dir, archive_name)

        if not os.path.isfile(archive_path):
            for row in target_rows:
                summary_rows.append({
                    "sample_id": row["sample_id"],
                    "src_info": row["src_info"],
                    "archive": archive_name,
                    "scene_path": scene_path,
                    "target_type": row["track_type"],
                    "target_id": row["track_id"],
                    "radius": args.radius,
                    "match_count_for_sample": row["match_count_for_sample"],
                    "target_found": 0,
                    "target_class_name": "",
                    "target_first_ts": "",
                    "target_last_ts": "",
                    "target_length": "",
                    "target_duration": "",
                    "n_interacting_tracks": 0,
                    "n_vru_interactions": 0,
                    "n_vehicle_interactions": 0,
                    "strongest_other_type": "",
                    "strongest_other_id": "",
                    "strongest_min_dist": "",
                    "strongest_close_frames": "",
                    "status": "archive_not_found",
                })
                write_sample_report(args.report_dir, row, {}, [], args.radius, "archive_not_found")
            continue

        print(f"[SCENE] {scene_path} ({archive_name})")
        scene_tracks = load_scene_tracks_from_archive(archive_path, scene_path)

        for row in target_rows:
            sample_id = row["sample_id"]
            target_type = row["track_type"]
            target_id = row["track_id"]
            target_key = (target_type, target_id)

            if target_key not in scene_tracks:
                summary_rows.append({
                    "sample_id": sample_id,
                    "src_info": row["src_info"],
                    "archive": archive_name,
                    "scene_path": scene_path,
                    "target_type": target_type,
                    "target_id": target_id,
                    "radius": args.radius,
                    "match_count_for_sample": row["match_count_for_sample"],
                    "target_found": 0,
                    "target_class_name": "",
                    "target_first_ts": "",
                    "target_last_ts": "",
                    "target_length": "",
                    "target_duration": "",
                    "n_interacting_tracks": 0,
                    "n_vru_interactions": 0,
                    "n_vehicle_interactions": 0,
                    "strongest_other_type": "",
                    "strongest_other_id": "",
                    "strongest_min_dist": "",
                    "strongest_close_frames": "",
                    "status": "target_missing_in_scene",
                })
                write_sample_report(args.report_dir, row, {}, [], args.radius, "target_missing_in_scene")
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
                "match_count_for_sample": row["match_count_for_sample"],
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
                "status": "ok",
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

            write_sample_report(args.report_dir, row, target_overview, interactions, args.radius, "ok")

    matched_ids = {r["sample_id"] for r in matched_rows}
    for row in all_rows:
        if row["sample_id"] in matched_ids:
            continue

        summary_rows.append({
            "sample_id": row["sample_id"],
            "src_info": row["src_info"],
            "archive": row.get("archive", ""),
            "scene_path": row.get("scene_path", ""),
            "target_type": row.get("track_type", ""),
            "target_id": row.get("track_id", ""),
            "radius": args.radius,
            "match_count_for_sample": row["match_count_for_sample"],
            "target_found": 0,
            "target_class_name": "",
            "target_first_ts": "",
            "target_last_ts": "",
            "target_length": "",
            "target_duration": "",
            "n_interacting_tracks": 0,
            "n_vru_interactions": 0,
            "n_vehicle_interactions": 0,
            "strongest_other_type": "",
            "strongest_other_id": "",
            "strongest_min_dist": "",
            "strongest_close_frames": "",
            "status": "unmatched_sample",
        })
        write_sample_report(args.report_dir, row, {}, [], args.radius, "unmatched_sample")

    summary_rows.sort(key=lambda r: int(r["sample_id"]) if r["sample_id"].isdigit() else r["sample_id"])
    detailed_rows.sort(key=lambda r: (int(r["sample_id"]) if r["sample_id"].isdigit() else r["sample_id"], -int(r["close_frames"])))

    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if detailed_rows:
        with open(args.out_detailed, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detailed_rows)

    print("DONE")
    print(f"Summary rows       : {len(summary_rows)}")
    print(f"Detailed rows      : {len(detailed_rows)}")
    print(f"Reports directory  : {args.report_dir}")
    print(f"Saved summary CSV  : {args.out_summary}")
    print(f"Saved detailed CSV : {args.out_detailed}")

if __name__ == "__main__":
    main()

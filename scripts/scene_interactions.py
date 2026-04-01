import os
import json
import math
import sys

if len(sys.argv) < 4:
    print("Usage: python scene_interactions.py <scene_dir> <target_type> <target_id> [radius]")
    print("Example: python scene_interactions.py ~/imptc_project/raw_scene/0000_20230322_081506 vrus 000 3.0")
    sys.exit(1)

scene_dir = os.path.expanduser(sys.argv[1])
target_type = sys.argv[2]          # vrus or vehicles
target_id = sys.argv[3]            # e.g. 000
radius = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0

def load_track(track_path):
    with open(track_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj["track_data"]

    ts_map = {}
    for _, v in data.items():
        ts = v["ts"]
        coords = v["coordinates"]
        ts_map[ts] = {
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
            "class_prob": v.get("class_prob"),
            "velocity": v.get("velocity"),
            "source_type": v.get("source_type"),
        }
    return obj.get("overview", {}), ts_map

def collect_tracks(base_dir, group_name):
    group_dir = os.path.join(base_dir, group_name)
    tracks = {}
    if not os.path.isdir(group_dir):
        return tracks

    for tid in sorted(os.listdir(group_dir)):
        track_json = os.path.join(group_dir, tid, "track.json")
        if os.path.isfile(track_json):
            overview, ts_map = load_track(track_json)
            tracks[(group_name, tid)] = {
                "overview": overview,
                "ts_map": ts_map,
                "path": track_json,
            }
    return tracks

all_tracks = {}
all_tracks.update(collect_tracks(scene_dir, "vrus"))
all_tracks.update(collect_tracks(scene_dir, "vehicles"))

target_key = (target_type, target_id)
if target_key not in all_tracks:
    print(f"Target not found: {target_key}")
    sys.exit(1)

target = all_tracks[target_key]
target_ts = target["ts_map"]

results = []

for other_key, other in all_tracks.items():
    if other_key == target_key:
        continue

    common_ts = set(target_ts.keys()) & set(other["ts_map"].keys())
    if not common_ts:
        continue

    min_dist = float("inf")
    count_close = 0

    for ts in common_ts:
        tx, ty = target_ts[ts]["x"], target_ts[ts]["y"]
        ox, oy = other["ts_map"][ts]["x"], other["ts_map"][ts]["y"]
        d = math.sqrt((ox - tx)**2 + (oy - ty)**2)

        if d < min_dist:
            min_dist = d
        if d < radius:
            count_close += 1

    if count_close > 0:
        results.append({
            "other_type": other_key[0],
            "other_id": other_key[1],
            "common_timestamps": len(common_ts),
            "close_frames": count_close,
            "min_dist": round(min_dist, 4),
            "class_name": other["overview"].get("class_name", ""),
        })

results.sort(key=lambda r: (-r["close_frames"], r["min_dist"]))

print(f"Scene: {scene_dir}")
print(f"Target: {target_type}/{target_id}")
print(f"Interaction radius: {radius} m")
print(f"Found {len(results)} interacting tracks\n")

for r in results[:50]:
    print(r)

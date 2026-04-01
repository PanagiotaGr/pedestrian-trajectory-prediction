import os
import json
import math
import sys

if len(sys.argv) < 4:
    print("Usage:")
    print("python inspect_pedestrian_moment.py <scene_dir> <target_id> <timestamp|'middle'> [top_k]")
    print("Example:")
    print("python inspect_pedestrian_moment.py ~/imptc_project/raw_scene/0001_20230322_083454 000 middle 10")
    sys.exit(1)

scene_dir = os.path.expanduser(sys.argv[1])
target_id = sys.argv[2]
timestamp_arg = sys.argv[3]
top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 10

def load_track(track_path):
    with open(track_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj

def collect_tracks(scene_dir):
    tracks = {}

    for group_name in ["vrus", "vehicles"]:
        group_dir = os.path.join(scene_dir, group_name)
        if not os.path.isdir(group_dir):
            continue

        for tid in sorted(os.listdir(group_dir)):
            track_json = os.path.join(group_dir, tid, "track.json")
            if not os.path.isfile(track_json):
                continue

            obj = load_track(track_json)
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

            tracks[(group_name, tid)] = {
                "overview": overview,
                "ts_map": ts_map,
                "path": track_json,
            }

    return tracks

tracks = collect_tracks(scene_dir)

target_key = ("vrus", target_id)
if target_key not in tracks:
    print(f"Target pedestrian vrus/{target_id} not found.")
    sys.exit(1)

target = tracks[target_key]
target_ts_sorted = sorted(target["ts_map"].keys(), key=int)

if not target_ts_sorted:
    print("Target has no timestamps.")
    sys.exit(1)

if timestamp_arg == "middle":
    timestamp = target_ts_sorted[len(target_ts_sorted) // 2]
else:
    timestamp = str(timestamp_arg)

if timestamp not in target["ts_map"]:
    print(f"Timestamp {timestamp} not found in target track.")
    print("Closest available timestamps around target:")
    print(target_ts_sorted[:5], "...", target_ts_sorted[-5:])
    sys.exit(1)

tx = target["ts_map"][timestamp]["x"]
ty = target["ts_map"][timestamp]["y"]
tz = target["ts_map"][timestamp]["z"]

neighbors = []

for (otype, oid), other in tracks.items():
    if (otype, oid) == target_key:
        continue
    if timestamp not in other["ts_map"]:
        continue

    ox = other["ts_map"][timestamp]["x"]
    oy = other["ts_map"][timestamp]["y"]
    oz = other["ts_map"][timestamp]["z"]

    d = math.sqrt((ox - tx) ** 2 + (oy - ty) ** 2)

    neighbors.append({
        "other_type": otype,
        "other_id": oid,
        "class_name": other["overview"].get("class_name", ""),
        "distance": round(d, 6),
        "x": ox,
        "y": oy,
        "z": oz,
        "velocity": other["ts_map"][timestamp].get("velocity"),
        "path": other["path"],
    })

neighbors.sort(key=lambda r: r["distance"])

print(f"Scene: {scene_dir}")
print(f"Target: vrus/{target_id}")
print(f"Timestamp: {timestamp}")
print(f"Target class: {target['overview'].get('class_name', '')}")
print(f"Target position: ({tx:.6f}, {ty:.6f}, {tz:.6f})")
print(f"Target velocity: {target['ts_map'][timestamp].get('velocity')}")
print()
print(f"Top {top_k} nearest agents at this moment:\n")

for i, n in enumerate(neighbors[:top_k], start=1):
    print(
        f"{i}. {n['other_type']}/{n['other_id']} "
        f"class={n['class_name']} "
        f"dist={n['distance']} "
        f"vel={n['velocity']}"
    )

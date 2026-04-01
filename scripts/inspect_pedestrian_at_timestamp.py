import os
import json
import math
import argparse


def load_track(track_path):
    with open(track_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    overview = obj.get("overview", {})
    data = obj.get("track_data", {})

    rows = []
    for _, v in data.items():
        coords = v["coordinates"]
        rows.append({
            "ts": int(v["ts"]),
            "x": float(coords[0]),
            "y": float(coords[1]),
            "z": float(coords[2]),
            "velocity": v.get("velocity", None),
            "class_name": overview.get("class_name", ""),
        })

    rows.sort(key=lambda r: r["ts"])
    return {
        "overview": overview,
        "points": rows,
    }


def nearest_point(points, target_ts):
    if not points:
        return None

    best = None
    best_dt = None
    for p in points:
        dt = abs(p["ts"] - target_ts)
        if best_dt is None or dt < best_dt:
            best = p
            best_dt = dt
    return best


def dist3(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def load_scene_tracks(scene_dir):
    tracks = {}

    for track_type in ["vrus", "vehicles"]:
        base = os.path.join(scene_dir, track_type)
        if not os.path.isdir(base):
            continue

        for track_id in sorted(os.listdir(base)):
            track_path = os.path.join(base, track_id, "track.json")
            if not os.path.exists(track_path):
                continue
            try:
                tracks[(track_type, track_id)] = load_track(track_path)
            except Exception:
                continue

    return tracks


def main():
    parser = argparse.ArgumentParser(description="Inspect one pedestrian at an exact timestamp")
    parser.add_argument("scene_dir")
    parser.add_argument("target_id")
    parser.add_argument("timestamp", type=int)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    tracks = load_scene_tracks(args.scene_dir)

    key = ("vrus", args.target_id)
    if key not in tracks:
        print(f"Target pedestrian vrus/{args.target_id} not found.")
        return

    target_track = tracks[key]
    target_pt = nearest_point(target_track["points"], args.timestamp)
    if target_pt is None:
        print("No target point found.")
        return

    print(f"Scene: {args.scene_dir}")
    print(f"Target: vrus/{args.target_id}")
    print(f"Requested timestamp: {args.timestamp}")
    print(f"Nearest target timestamp: {target_pt['ts']}")
    print(f"Target class: {target_track['overview'].get('class_name', '')}")
    print(f"Target position: ({target_pt['x']:.6f}, {target_pt['y']:.6f}, {target_pt['z']:.6f})")
    print(f"Target velocity: {target_pt.get('velocity', None)}")
    print()
    print(f"Top {args.top_k} nearest agents at this moment:")
    print()

    neighbors = []
    for (track_type, track_id), tr in tracks.items():
        if (track_type, track_id) == key:
            continue

        pt = nearest_point(tr["points"], args.timestamp)
        if pt is None:
            continue

        d = dist3(target_pt, pt)
        neighbors.append({
            "track_type": track_type,
            "track_id": track_id,
            "class_name": tr["overview"].get("class_name", ""),
            "dist": d,
            "velocity": pt.get("velocity", None),
            "ts": pt["ts"],
        })

    neighbors.sort(key=lambda r: r["dist"])

    for i, n in enumerate(neighbors[:args.top_k], start=1):
        print(
            f"{i}. {n['track_type']}/{n['track_id']} "
            f"class={n['class_name']} "
            f"dist={n['dist']:.6f} "
            f"vel={n['velocity']} "
            f"ts={n['ts']}"
        )


if __name__ == "__main__":
    main()

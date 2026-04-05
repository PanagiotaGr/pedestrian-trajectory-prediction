"""
enrich_crosswalk_cases.py
=========================
Παίρνει το crosswalk_approaching_vehicle_cases.csv και προσθέτει:
- Ολόκληρη την τροχιά του πεζού @ 10Hz
- Ground type σε κάθε θέση (από segmentation map)
- Traffic lights (f1, f2, f3) σε κάθε timestamp

Output: results/crosswalk_cases_enriched.csv

Χρήση:
    python enrich_crosswalk_cases.py
"""

import os
import csv
import json
import tarfile
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
DATA_DIR    = Path(os.path.expanduser("~/imptc_project/data"))
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))

INPUT_CSV   = RESULTS_DIR / "crosswalk_approaching_vehicle_cases.csv"
OUTPUT_CSV  = RESULTS_DIR / "crosswalk_cases_enriched.csv"
GROUND_CSV  = DATA_DIR / "ground_plane_map.csv"

STEP_US     = 100_000   # 10Hz downsample

GROUND_NAMES = {
    0: "road", 1: "sidewalk", 2: "ground", 3: "curb",
    4: "road_line", 5: "crosswalk", 6: "bikelane", 7: "unknown"
}

LIGHT_STATES = {
    4: "green", 10: "red", 20: "yellow",
    30: "red_yellow", 2: "yellow_blinking", 11: "disabled"
}


# ─── 1. Φόρτωση ground map ───────────────────────────────────────────────────
def load_ground_map():
    print("[→] Φόρτωση ground map...")
    points, labels = [], []
    with open(GROUND_CSV) as f:
        for row in csv.DictReader(f):
            points.append([float(row["x"]), float(row["y"])])
            labels.append(int(row["ground_type_id"]))
    tree = KDTree(np.array(points, dtype=np.float32))
    print(f"  {len(points):,} points ✓")
    return tree, np.array(labels, dtype=np.int8)


def get_ground_type(x, y, tree, labels):
    _, idx = tree.query([[x, y]], k=1)
    gid = int(labels[idx[0]])
    return gid, GROUND_NAMES.get(gid, "unknown")


# ─── 2. Φόρτωση track από archive ────────────────────────────────────────────
def load_track(tar, scene_path, track_type, track_id):
    path = f"{scene_path}/{track_type}/{track_id}/track.json"
    try:
        f = tar.extractfile(tar.getmember(path))
        obj = json.load(f)
    except Exception:
        return [], {}

    track_data = obj.get("track_data", {})
    points = []
    for _, v in track_data.items():
        try:
            points.append({
                "ts": int(v["ts"]),
                "x":  float(v["coordinates"][0]),
                "y":  float(v["coordinates"][1]),
            })
        except Exception:
            continue

    points.sort(key=lambda p: p["ts"])

    # Downsample 10Hz
    ds = []
    if points:
        ds = [points[0]]
        last_ts = points[0]["ts"]
        for p in points[1:]:
            if (p["ts"] - last_ts) >= STEP_US:
                ds.append(p)
                last_ts = p["ts"]

    return ds


# ─── 3. Φόρτωση traffic lights ───────────────────────────────────────────────
def load_traffic_lights(tar, scene_path):
    path = f"{scene_path}/context/traffic_light_signals.json"
    try:
        f = tar.extractfile(tar.getmember(path))
        data = json.load(f)
    except Exception:
        return {}

    result = {}
    for ts_str, signals in data.get("status_data", {}).items():
        result[int(ts_str)] = {
            "f1": signals.get("f1", 11),
            "f2": signals.get("f2", 11),
            "f3": signals.get("f3", 11),
        }
    return result


def nearest_tl(ts, tl_data):
    if not tl_data:
        return {"f1": 11, "f2": 11, "f3": 11}
    nearest = min(tl_data.keys(), key=lambda t: abs(t - ts))
    return tl_data[nearest]


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Ground map
    tree, labels = load_ground_map()

    # 2. Διάβασε cases
    with open(INPUT_CSV) as f:
        cases = list(csv.DictReader(f))
    print(f"\n[→] {len(cases)} crosswalk cases")

    # 3. Για κάθε case → φόρτωσε τροχιά
    out_rows = []

    for case in cases:
        archive_name = case["archive"]
        scene_path   = case["scene_path"]
        target_id    = case["target_id"]
        sample_id    = case["sample_id"]

        print(f"\n  Case {sample_id}: scene={scene_path}, pedestrian={target_id}")

        archive_path = DATA_DIR / archive_name
        if not archive_path.exists():
            print(f"  [!] Archive δεν βρέθηκε: {archive_name}")
            continue

        with tarfile.open(archive_path, "r:gz") as tar:
            # Φόρτωσε τροχιά πεζού
            track = load_track(tar, scene_path, "vrus", target_id)
            # Φόρτωσε traffic lights
            tl_data = load_traffic_lights(tar, scene_path)

        if not track:
            print(f"  [!] Δεν βρέθηκε τροχιά")
            continue

        print(f"  {len(track)} frames @ 10Hz")

        # 4. Για κάθε frame → ground type + traffic lights
        for frame in track:
            ts = frame["ts"]
            x  = frame["x"]
            y  = frame["y"]

            # Ground type
            gid, gname = get_ground_type(x, y, tree, labels)

            # Traffic lights
            tl = nearest_tl(ts, tl_data)
            f1, f2, f3 = tl["f1"], tl["f2"], tl["f3"]

            out_rows.append({
                # Ταυτότητα
                "sample_id":    sample_id,
                "scene_path":   scene_path,
                "target_id":    target_id,
                "class":        case["target_class_name"],
                # Timestamp
                "ts":           ts,
                # Θέση πεζού (global)
                "x":            round(x, 4),
                "y":            round(y, 4),
                # Ground type
                "ground_type_id":   gid,
                "ground_type_name": gname,
                # Traffic lights
                "f1":           f1,
                "f2":           f2,
                "f3":           f3,
                "f1_state":     LIGHT_STATES.get(f1, "unknown"),
                "f2_state":     LIGHT_STATES.get(f2, "unknown"),
                "f3_state":     LIGHT_STATES.get(f3, "unknown"),
                # Όχημα (από original case)
                "vehicle_id":   case["other_id"],
                "vehicle_class": case["other_class_name"],
                "dist_to_vehicle": round(float(case["dist_xy"]), 4),
                "interaction_zone": case["interaction_zone"],
                "motion_relation":  case["motion_relation"],
            })

    # 5. Αποθήκευση
    if not out_rows:
        print("[!] Κανένα row!")
        exit(1)

    fieldnames = list(out_rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n{'='*55}")
    print(f"[OK] Αποθηκεύτηκε: {OUTPUT_CSV}")
    print(f"  Γραμμές: {len(out_rows):,}")

    # Στατιστικά
    from collections import Counter
    ground_counts = Counter(r["ground_type_name"] for r in out_rows)
    f1_counts     = Counter(r["f1_state"] for r in out_rows)

    print(f"\n  Ground types:")
    for g, cnt in ground_counts.most_common():
        print(f"    {g:<12}: {cnt:>5}")

    print(f"\n  f1 (traffic light):")
    for s, cnt in f1_counts.most_common():
        print(f"    {s:<16}: {cnt:>5}")

    # Παράδειγμα
    print(f"\n  Παράδειγμα (πρώτες 3 γραμμές):")
    for r in out_rows[:3]:
        print(f"    ts={r['ts']}  x={r['x']}  y={r['y']}")
        print(f"    ground={r['ground_type_name']}  "
              f"f1={r['f1_state']}  f2={r['f2_state']}  f3={r['f3_state']}")

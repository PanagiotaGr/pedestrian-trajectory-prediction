"""
add_traffic_lights.py
=====================
Προσθέτει traffic light signals (f1, f2, f3) σε κάθε timestamp.
Χρησιμοποιεί το interactions_detailed.csv για να ξέρει
ποια scene ανήκει σε ποιο archive.

Χρήση:
    python add_traffic_lights.py
"""

import os
import csv
import json
import tarfile
from pathlib import Path

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
DATA_DIR         = Path(os.path.expanduser("~/imptc_project/data"))
RESULTS_DIR      = Path(os.path.expanduser("~/imptc_project/results"))
INPUT_JSON       = RESULTS_DIR / "grid_dataset_with_ground.json"
OUTPUT_JSON      = RESULTS_DIR / "grid_dataset_final.json"
INTERACTIONS_CSV = RESULTS_DIR / "interactions_detailed.csv"

LIGHT_STATES = {
    2: "yellow_blinking", 4: "green", 10: "red",
    11: "disabled", 20: "yellow", 30: "red_yellow",
}


# ─── 1. Φόρτωση scene→archive mapping ───────────────────────────────────────
def load_scene_archive_mapping(csv_path: Path) -> dict:
    """scene_id → archive_name"""
    mapping = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            scene  = row["scene_path"]
            archive = row["archive"]
            if scene not in mapping:
                mapping[scene] = archive
    print(f"[OK] {len(mapping)} scenes → archives mapped")
    return mapping


# ─── 2. Traffic lights από archive ───────────────────────────────────────────
def load_traffic_lights(tar, scene_path) -> dict:
    """ts(int) → {f1, f2, f3}"""
    tl_path = f"{scene_path}/context/traffic_light_signals.json"
    try:
        f = tar.extractfile(tar.getmember(tl_path))
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


def find_nearest_tl(ts: int, tl_data: dict) -> dict:
    if not tl_data:
        return {"f1": 11, "f2": 11, "f3": 11}
    nearest = min(tl_data.keys(), key=lambda t: abs(t - ts))
    return tl_data[nearest]


# ─── 3. Κύρια επεξεργασία ────────────────────────────────────────────────────
def process(data, scene_mapping):
    """
    Για κάθε scene, ανοίγει το σωστό archive και παίρνει τα traffic lights.
    """
    # scene → track indices
    scene_index = {}
    for i, track in enumerate(data):
        scene = track["scene"]
        scene_index.setdefault(scene, []).append(i)

    # archive → [scenes]
    archive_scenes = {}
    for scene, archive in scene_mapping.items():
        if scene in scene_index:  # μόνο scenes που έχουμε tracks
            archive_scenes.setdefault(archive, []).append(scene)

    # Scenes χωρίς mapping → disabled
    mapped_scenes = set(scene_mapping.keys())
    for scene, track_indices in scene_index.items():
        if scene not in mapped_scenes:
            print(f"  [!] Χωρίς mapping: {scene} → disabled")
            for ti in track_indices:
                for ts in data[ti]["timesteps"]:
                    ts["traffic_lights"] = {"f1": 11, "f2": 11, "f3": 11}

    total_scenes = sum(len(v) for v in archive_scenes.values())
    done = 0

    for archive_name, scenes in archive_scenes.items():
        archive_path = DATA_DIR / archive_name
        if not archive_path.exists():
            print(f"[!] Δεν βρέθηκε: {archive_name}")
            continue

        print(f"\n[→] {archive_name} ({len(scenes)} scenes)")
        with tarfile.open(archive_path, "r:gz") as tar:
            for scene in scenes:
                tl_data = load_traffic_lights(tar, scene)
                for ti in scene_index[scene]:
                    for ts in data[ti]["timesteps"]:
                        ts["traffic_lights"] = find_nearest_tl(ts["ts"], tl_data)
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{total_scenes} scenes...", end="\r")

    print(f"\n[OK] {done}/{total_scenes} scenes επεξεργάστηκαν")


# ─── 4. Αποθήκευση & στατιστικά ─────────────────────────────────────────────
def save_and_summarize(data):
    print(f"\n[→] Αποθήκευση...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f)
    print(f"[OK] {OUTPUT_JSON}")

    from collections import Counter
    f1 = Counter()
    f2 = Counter()
    f3 = Counter()
    for track in data:
        for ts in track["timesteps"]:
            tl = ts.get("traffic_lights", {})
            f1[tl.get("f1", 11)] += 1
            f2[tl.get("f2", 11)] += 1
            f3[tl.get("f3", 11)] += 1

    print(f"\nΣτατιστικά traffic lights:")
    for fname, counts in [("f1", f1), ("f2", f2), ("f3", f3)]:
        print(f"  {fname}:")
        for state, cnt in counts.most_common():
            print(f"    {state:2d} ({LIGHT_STATES.get(state,'?'):<16}): {cnt:>8,}")

    # Παράδειγμα
    ex = data[0]["timesteps"][0]
    print(f"\nΠαράδειγμα ts={ex['ts']}:")
    print(f"  traffic_lights: {ex.get('traffic_lights')}")
    print(f"\n[OK] Τελικά Grid layers:")
    print(f"  vrus           → 5x5 occupation (VRUs)")
    print(f"  vehicles       → 5x5 occupation (οχήματα)")
    print(f"  ground         → 5x5 ground type")
    print(f"  rel_x/y        → 5x5 relative position")
    print(f"  rel_vx/vy      → 5x5 relative velocity")
    print(f"  traffic_lights → {{f1, f2, f3}} per timestamp")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Scene→archive mapping
    print("[→] Φόρτωση scene→archive mapping...")
    scene_mapping = load_scene_archive_mapping(INTERACTIONS_CSV)

    # 2. Dataset
    print(f"\n[→] Φόρτωση dataset...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    total = sum(len(t["timesteps"]) for t in data)
    print(f"  {len(data):,} tracks, {total:,} frames")

    # 3. Πρόσθεσε traffic lights
    process(data, scene_mapping)

    # 4. Αποθήκευση
    save_and_summarize(data)
    print(f"\n[OK] Ολοκληρώθηκε!")

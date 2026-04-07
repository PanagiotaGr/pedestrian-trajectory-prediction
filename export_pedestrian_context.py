"""
export_pedestrian_context.py
============================
Εξάγει σε CSV για κάθε πεζό, σε κάθε timestamp @ 10Hz:
  - Θέση (x, y) στο χώρο
  - Ground type κάτω από τα πόδια του (κεντρικό κελί [2,2] του grid)
  - Traffic lights (f1, f2, f3) και χρώματα

Output: results/pedestrian_context.csv

Χρήση:
    python export_pedestrian_context.py
"""

import os
import json
import csv
from pathlib import Path

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))
INPUT_JSON  = RESULTS_DIR / "grid_dataset_final.json"
OUTPUT_CSV  = RESULTS_DIR / "pedestrian_context.csv"

GROUND_NAMES = {
    0: "road", 1: "sidewalk", 2: "ground", 3: "curb",
    4: "road_line", 5: "crosswalk", 6: "bikelane", 7: "unknown"
}

LIGHT_STATES = {
    4: "green", 10: "red", 20: "yellow",
    30: "red_yellow", 2: "yellow_blinking", 11: "disabled"
}

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[→] Φόρτωση grid_dataset_final.json...")
    with open(INPUT_JSON) as f:
        data = json.load(f)

    total = sum(len(t["timesteps"]) for t in data)
    print(f"  {len(data):,} tracks, {total:,} frames")

    print(f"\n[→] Εξαγωγή σε CSV...")

    fieldnames = [
        "track_id", "scene", "class",
        "ts", "x", "y",
        "ground_type_id", "ground_type_name",
        "f1", "f2", "f3",
        "f1_state", "f2_state", "f3_state",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, track in enumerate(data):
            if i % 200 == 0:
                print(f"  Track {i:,}/{len(data):,}...", end="\r")

            for ts in track["timesteps"]:
                # Ground type από το κεντρικό κελί [2,2] του grid
                # (αυτό είναι ακριβώς κάτω από τον πεζό)
                gid   = ts["grid"]["ground"][2][2]
                gname = GROUND_NAMES.get(gid, "unknown")

                # Traffic lights
                tl = ts.get("traffic_lights", {"f1": 11, "f2": 11, "f3": 11})
                f1 = tl.get("f1", 11)
                f2 = tl.get("f2", 11)
                f3 = tl.get("f3", 11)

                writer.writerow({
                    "track_id":        track["track_id"],
                    "scene":           track["scene"],
                    "class":           track["class"],
                    "ts":              ts["ts"],
                    "x":               round(ts["ax_global"], 4),
                    "y":               round(ts["ay_global"], 4),
                    "ground_type_id":  gid,
                    "ground_type_name": gname,
                    "f1":              f1,
                    "f2":              f2,
                    "f3":              f3,
                    "f1_state":        LIGHT_STATES.get(f1, "unknown"),
                    "f2_state":        LIGHT_STATES.get(f2, "unknown"),
                    "f3_state":        LIGHT_STATES.get(f3, "unknown"),
                })

    print(f"\n[OK] Αποθηκεύτηκε: {OUTPUT_CSV}")

    # Στατιστικά
    print(f"\n[→] Στατιστικά...")
    from collections import Counter
    ground_counts = Counter()
    f1_counts     = Counter()

    # Διάβασε το CSV για στατιστικά
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            ground_counts[row["ground_type_name"]] += 1
            f1_counts[row["f1_state"]] += 1

    total_rows = sum(ground_counts.values())
    print(f"  Σύνολο γραμμών: {total_rows:,}")

    print(f"\n  Πού βρίσκονται οι πεζοί (ground type):")
    for g, cnt in ground_counts.most_common():
        pct = cnt / total_rows * 100
        print(f"    {g:<12}: {cnt:>8,}  ({pct:.1f}%)")

    print(f"\n  Traffic light f1:")
    for s, cnt in f1_counts.most_common():
        pct = cnt / total_rows * 100
        print(f"    {s:<16}: {cnt:>8,}  ({pct:.1f}%)")

    print(f"\n  Παράδειγμα (πρώτες 3 γραμμές):")
    with open(OUTPUT_CSV) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"    track={row['track_id']}  ts={row['ts']}")
            print(f"    x={row['x']}  y={row['y']}")
            print(f"    ground={row['ground_type_name']}")
            print(f"    f1={row['f1_state']}  f2={row['f2_state']}  f3={row['f3_state']}")
            print()

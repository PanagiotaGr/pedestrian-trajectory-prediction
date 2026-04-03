"""
build_grid_with_ground.py
=========================
Προσθέτει ground plane segmentation στο 5x5 grid.

Το ground grid ΔΕΝ γυρίζει με τον πεζό — είναι global (North-Up).
Κάθε κελί [row,col] = ground type στη global θέση:
  col 0..4 → x: ax-2, ax-1, ax, ax+1, ax+2
  row 0..4 → y: ay+2, ay+1, ay, ay-1, ay-2

Ground types:
  0=road, 1=sidewalk, 2=ground, 3=curb,
  4=road_line, 5=crosswalk, 6=bikelane, 7=unknown

Χρήση:
    python build_grid_with_ground.py

Απαιτήσεις:
    pip install numpy scipy
"""

import os
import json
import csv
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
DATA_DIR    = Path(os.path.expanduser("~/imptc_project/data"))
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))

INPUT_JSON  = RESULTS_DIR / "grid_dataset.json"
OUTPUT_JSON = RESULTS_DIR / "grid_dataset_with_ground.json"
GROUND_CSV  = DATA_DIR / "ground_plane_map.csv"

GRID_SIZE = 5
CELL_SIZE = 1.0

GROUND_TYPES = {
    "road": 0, "sidewalk": 1, "ground": 2, "curb": 3,
    "road_line": 4, "crosswalk": 5, "bikelane": 6, "unknown": 7,
}


# ─── 1. Φόρτωση ground map ───────────────────────────────────────────────────
def load_ground_map(csv_path: Path):
    print(f"[→] Φόρτωση ground map...")
    points, labels = [], []

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            points.append([float(row["x"]), float(row["y"])])
            labels.append(int(row["ground_type_id"]))

    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int8)
    tree   = KDTree(points)

    print(f"  {len(points):,} points  KDTree OK")
    return tree, labels


# ─── 2. Ground grid (global, no rotation) ────────────────────────────────────
def build_ground_grid(ax, ay, tree, labels):
    """
    Κάθε κελί = global θέση του A + offset.
    ΔΕΝ γυρίζει με τον πεζό — είναι πάντα North-Up.
    col 0..4 → x: ax-2, ax-1, ax, ax+1, ax+2  (West→East)
    row 0..4 → y: ay+2, ay+1, ay, ay-1, ay-2  (North→South)
    """
    queries = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            gx = ax + (col - 2) * CELL_SIZE
            gy = ay + (2 - row) * CELL_SIZE
            queries.append([gx, gy])

    _, idxs = tree.query(np.array(queries, dtype=np.float32), k=1)
    grid = np.array([labels[i] for i in idxs], dtype=np.int8)
    return grid.reshape(GRID_SIZE, GRID_SIZE).tolist()


# ─── 3. Επεξεργασία dataset ──────────────────────────────────────────────────
def process_dataset(tree, labels):
    print(f"\n[→] Φόρτωση: {INPUT_JSON.name}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    total = sum(len(t["timesteps"]) for t in data)
    print(f"  {len(data):,} tracks, {total:,} frames")

    print(f"\n[→] Προσθήκη ground grid...")
    for i, track in enumerate(data):
        if i % 200 == 0:
            print(f"  {i:,}/{len(data):,}...", end="\r")
        for ts in track["timesteps"]:
            ts["grid"]["ground"] = build_ground_grid(
                ts["ax_global"], ts["ay_global"], tree, labels
            )

    print(f"\n[OK] Ground grid προστέθηκε!")

    print(f"\n[→] Αποθήκευση...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f)
    print(f"[OK] {OUTPUT_JSON}")

    # Παράδειγμα
    print(f"\nΠαράδειγμα:")
    ex = data[0]["timesteps"][0]
    g  = np.array(ex["grid"]["ground"])
    print(f"  A global = ({ex['ax_global']}, {ex['ay_global']})")
    print(f"  Ground grid (North-Up, no rotation):")
    print(g)
    names = {v: k for k, v in GROUND_TYPES.items()}
    print(f"\n  Legenda:")
    for gid in np.unique(g):
        cnt = (g == gid).sum()
        print(f"    {gid} = {names.get(gid,'?'):<12}: {cnt} κελιά")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tree, labels = load_ground_map(GROUND_CSV)
    process_dataset(tree, labels)
    print(f"\n[OK] Ολοκληρώθηκε!")
    print(f"  Επόμενο: traffic lights (f1, f2, f3)")

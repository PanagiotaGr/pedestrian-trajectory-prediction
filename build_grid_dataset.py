"""
build_grid_dataset.py
=====================
Για κάθε πεζό A, σε ΚΑΘΕ timestamp @ 10Hz της τροχιάς του:
  - A = (0,0) στο κέντρο
  - 5x5 grid, κάθε κελί 1m x 1m  → area: -2.5m..+2.5m
  - Βρίσκει γείτονες (VRUs + οχήματα) στο local frame του A
  - p_B|A = R_θA · (p_B - p_A)
  - V_rel = R_θA · (v_B - v_A)

Output: results/grid_dataset.json
  [
    {
      "track_id": "0000/000",
      "scene":    "0000_20230322_081506",
      "class":    "person",
      "timesteps": [
        {
          "ts": 1679472907040035,
          "ax_global": x, "ay_global": y,   ← global pos (για debugging)
          "theta_a": θ,                       ← heading
          "grid": {                           ← 5x5 grid
            "vrus":     [[0,0,0,0,0],...]    ← 1 αν υπάρχει VRU
            "vehicles": [[0,0,0,0,0],...]    ← 1 αν υπάρχει όχημα
            "rel_x":    [[...]]              ← relative x ανά κελί
            "rel_y":    [[...]]              ← relative y ανά κελί
            "rel_vx":   [[...]]              ← relative vx ανά κελί
            "rel_vy":   [[...]]              ← relative vy ανά κελί
          }
        },
        ...
      ]
    },
    ...
  ]

Χρήση:
    python build_grid_dataset.py

Απαιτήσεις:
    pip install numpy
"""

import os
import json
import math
import tarfile
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
DATA_DIR    = Path(os.path.expanduser("~/imptc_project/data"))
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = RESULTS_DIR / "grid_dataset.json"

# Grid parameters
GRID_SIZE   = 5          # 5x5
CELL_SIZE   = 1.0        # 1m x 1m
HALF        = GRID_SIZE * CELL_SIZE / 2.0   # 2.5m

# Downsample @ 10Hz
STEP_US     = 100_000    # 100ms

EPS         = 1e-9

# Archives
ARCHIVES = [
    "imptc_set_01.tar.gz",
    "imptc_set_02.tar.gz",
    "imptc_set_03.tar.gz",
    "imptc_set_04.tar.gz",
    "imptc_set_05.tar.gz",
]


# ─── Βοηθητικά ───────────────────────────────────────────────────────────────
def rotation_global_to_local(theta):
    """R_θA: global → local frame of A"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s], [-s, c]])


def estimate_velocity(points, idx):
    """vx, vy από διαδοχικά points (central difference)"""
    n = len(points)
    if n < 2:
        return 0.0, 0.0
    if 0 < idx < n - 1:
        p0, p1 = points[idx-1], points[idx+1]
    elif idx == 0:
        p0, p1 = points[0], points[1]
    else:
        p0, p1 = points[-2], points[-1]

    dt = (p1["ts"] - p0["ts"]) / 1_000_000.0
    if abs(dt) < EPS:
        return 0.0, 0.0
    return (p1["x"] - p0["x"]) / dt, (p1["y"] - p0["y"]) / dt


def world_to_cell(rel_x, rel_y):
    """
    Μετατρέπει relative position (rel_x, rel_y) σε grid cell (row, col).
    A=(0,0) → κελί (2,2) στο κέντρο.
    Επιστρέφει (row, col) ή None αν είναι εκτός grid.
    """
    # col: rel_x → 0..4  (αριστερά → δεξιά)
    # row: rel_y → 4..0  (κάτω → πάνω, flip για matrix notation)
    col = int(math.floor(rel_x + HALF))
    row = int(math.floor(-rel_y + HALF))

    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
        return row, col
    return None


def downsample(points):
    """Κρατάει 1 frame κάθε 100ms (10Hz)"""
    if not points:
        return []
    result  = [points[0]]
    last_ts = points[0]["ts"]
    for p in points[1:]:
        if (p["ts"] - last_ts) >= STEP_US:
            result.append(p)
            last_ts = p["ts"]
    return result


# ─── Φόρτωση scene από archive ───────────────────────────────────────────────
def load_scene_from_archive(tar, scene_path):
    """
    Φορτώνει VRUs και vehicles από μία scene μέσα στο tar.
    Επιστρέφει:
      vrus:     {track_id: {points, ts_to_idx, class_name}}
      vehicles: {track_id: {points, ts_to_idx}}
    """
    vrus     = {}
    vehicles = {}
    prefix   = scene_path.rstrip("/") + "/"

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

        track_type = parts[1]   # "vrus" or "vehicles"
        track_id   = parts[2]   # "000", "001", ...

        f = tar.extractfile(member)
        if f is None:
            continue

        try:
            obj = json.load(f)
        except Exception:
            continue

        overview   = obj.get("overview", {})
        track_data = obj.get("track_data", {})

        points = []
        for _, v in track_data.items():
            try:
                points.append({
                    "ts": int(v["ts"]),
                    "x":  float(v["coordinates"][0]),
                    "y":  float(v["coordinates"][1]),
                    "z":  float(v["coordinates"][2]),
                })
            except Exception:
                continue

        points.sort(key=lambda p: p["ts"])
        ts_to_idx = {p["ts"]: i for i, p in enumerate(points)}

        info = {
            "points":    points,
            "ts_to_idx": ts_to_idx,
            "class_name": overview.get("class_name", "unknown"),
        }

        if track_type == "vrus":
            vrus[track_id] = info
        elif track_type == "vehicles":
            vehicles[track_id] = info

    return vrus, vehicles


# ─── Φτιάξε 5x5 grid για ένα timestamp ──────────────────────────────────────
def build_grid(ax, ay, theta_a, avx, avy,
               vrus, vehicles, ts,
               target_id):
    """
    Φτιάχνει το 5x5 grid για τον πεζό A στο timestamp ts.
    """
    R = rotation_global_to_local(theta_a)

    # Αρχικοποίηση grid layers
    grid_vru_occ  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    grid_veh_occ  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    grid_rel_x    = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    grid_rel_y    = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    grid_rel_vx   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    grid_rel_vy   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    p_A = np.array([ax, ay])
    v_A = np.array([avx, avy])

    # ── VRUs ──
    for tid, info in vrus.items():
        if tid == target_id:
            continue   # skip ο ίδιος
        if ts not in info["ts_to_idx"]:
            continue

        idx  = info["ts_to_idx"][ts]
        pb   = info["points"][idx]
        bvx, bvy = estimate_velocity(info["points"], idx)

        p_B  = np.array([pb["x"], pb["y"]])
        v_B  = np.array([bvx, bvy])

        # p_B|A = R · (p_B - p_A)
        p_rel = R @ (p_B - p_A)
        # V_rel = R · (v_B - v_A)
        v_rel = R @ (v_B - v_A)

        cell = world_to_cell(p_rel[0], p_rel[1])
        if cell is None:
            continue   # εκτός grid

        row, col = cell
        grid_vru_occ[row, col] = 1
        grid_rel_x[row, col]   = round(float(p_rel[0]), 3)
        grid_rel_y[row, col]   = round(float(p_rel[1]), 3)
        grid_rel_vx[row, col]  = round(float(v_rel[0]), 3)
        grid_rel_vy[row, col]  = round(float(v_rel[1]), 3)

    # ── Vehicles ──
    for tid, info in vehicles.items():
        if ts not in info["ts_to_idx"]:
            continue

        idx  = info["ts_to_idx"][ts]
        pb   = info["points"][idx]
        bvx, bvy = estimate_velocity(info["points"], idx)

        p_B  = np.array([pb["x"], pb["y"]])
        v_B  = np.array([bvx, bvy])

        p_rel = R @ (p_B - p_A)
        v_rel = R @ (v_B - v_A)

        cell = world_to_cell(p_rel[0], p_rel[1])
        if cell is None:
            continue

        row, col = cell
        grid_veh_occ[row, col] = 1
        # Αν δεν υπάρχει ήδη VRU στο κελί, γράψε velocities
        if grid_vru_occ[row, col] == 0:
            grid_rel_x[row, col]  = round(float(p_rel[0]), 3)
            grid_rel_y[row, col]  = round(float(p_rel[1]), 3)
            grid_rel_vx[row, col] = round(float(v_rel[0]), 3)
            grid_rel_vy[row, col] = round(float(v_rel[1]), 3)

    return {
        "vrus":     grid_vru_occ.tolist(),
        "vehicles": grid_veh_occ.tolist(),
        "rel_x":    grid_rel_x.tolist(),
        "rel_y":    grid_rel_y.tolist(),
        "rel_vx":   grid_rel_vx.tolist(),
        "rel_vy":   grid_rel_vy.tolist(),
    }


# ─── Process μία scene ───────────────────────────────────────────────────────
def process_scene(tar, scene_path, scene_id):
    """
    Επεξεργάζεται όλους τους πεζούς μίας scene.
    Επιστρέφει λίστα από track records.
    """
    vrus, vehicles = load_scene_from_archive(tar, scene_path)

    records = []

    for target_id, info in vrus.items():
        # Μόνο πεζοί
        if info["class_name"] not in ("person", "pedestrian"):
            continue

        points_ds = downsample(info["points"])
        if len(points_ds) < 2:
            continue

        timesteps = []

        for i, pt in enumerate(points_ds):
            ts  = pt["ts"]
            ax  = pt["x"]
            ay  = pt["y"]

            avx, avy = estimate_velocity(points_ds, i)
            speed_a  = math.sqrt(avx**2 + avy**2)

            # Heading
            if speed_a > EPS:
                theta_a = math.atan2(avy, avx)
            else:
                theta_a = 0.0

            # 5x5 grid
            grid = build_grid(
                ax, ay, theta_a, avx, avy,
                vrus, vehicles, ts, target_id
            )

            timesteps.append({
                "ts":        ts,
                "ax_global": round(ax, 4),
                "ay_global": round(ay, 4),
                "theta_a":   round(theta_a, 4),
                "speed_a":   round(speed_a, 4),
                "grid":      grid,
            })

        records.append({
            "track_id":  f"{scene_id}/{target_id}",
            "scene":     scene_id,
            "class":     info["class_name"],
            "n_frames":  len(timesteps),
            "timesteps": timesteps,
        })

    return records


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_records = []
    total_frames = 0

    for archive_name in ARCHIVES:
        archive_path = DATA_DIR / archive_name
        if not archive_path.exists():
            print(f"[!] Δεν βρέθηκε: {archive_path}")
            continue

        print(f"\n[→] {archive_name}")
        with tarfile.open(archive_path, "r:gz") as tar:
            # Βρες όλες τις scenes
            scenes = set()
            for member in tar.getmembers():
                parts = member.name.split("/")
                if len(parts) >= 2 and parts[1] in ("vrus", "vehicles"):
                    scenes.add(parts[0])

            scenes = sorted(scenes)
            print(f"    {len(scenes)} scenes")

            for scene_path in scenes:
                records = process_scene(tar, scene_path, scene_path)
                all_records.extend(records)
                frames = sum(r["n_frames"] for r in records)
                total_frames += frames
                print(f"    [{scene_path}] {len(records)} πεζοί, {frames} frames")

    print(f"\n{'='*55}")
    print(f"  Σύνολο πεζών:  {len(all_records):,}")
    print(f"  Σύνολο frames: {total_frames:,}")
    print(f"  @ 10Hz")

    # Αποθήκευση
    print(f"\n[→] Αποθήκευση στο: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_records, f)

    print(f"[✓] Ολοκληρώθηκε!")
    print(f"\nΠαράδειγμα:")
    if all_records:
        ex = all_records[0]
        print(f"  Track: {ex['track_id']} ({ex['class']})")
        print(f"  Frames: {ex['n_frames']}")
        ts0 = ex["timesteps"][0]
        print(f"  t=0: A=({ts0['ax_global']}, {ts0['ay_global']})")
        print(f"       θ={ts0['theta_a']} rad")
        print(f"       VRUs grid:\n{np.array(ts0['grid']['vrus'])}")
        print(f"       Vehicles grid:\n{np.array(ts0['grid']['vehicles'])}")

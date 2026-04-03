"""
build_training_samples.py
=========================
Φτιάχνει training samples με sliding window.
Αποθηκεύει chunk by chunk για να μην ξεφύγει η RAM.

Input:  38 frames → 5x5 grid (7 channels) + traffic lights (3)
Output: 48 frames → (x, y) θέση πεζού

Χρήση:
    python build_training_samples.py
"""

import os
import json
import numpy as np
from pathlib import Path

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
RESULTS_DIR   = Path(os.path.expanduser("~/imptc_project/results"))
INPUT_JSON    = RESULTS_DIR / "grid_dataset_final.json"

INPUT_FRAMES  = 38
OUTPUT_FRAMES = 48
TOTAL_FRAMES  = INPUT_FRAMES + OUTPUT_FRAMES  # 86

GRID_SIZE     = 5
N_CHANNELS    = 7

TL_ENCODING   = {4: 1.0, 10: -1.0, 20: 0.5, 30: -0.5, 2: 0.0, 11: 0.0}

TRAIN_RATIO   = 0.80
EVAL_RATIO    = 0.10
CHUNK_SIZE    = 500  # tracks ανά chunk


# ─── Helpers ─────────────────────────────────────────────────────────────────
def grid_to_tensor(ts_data):
    g = ts_data["grid"]
    t = np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    t[0] = np.array(g["vrus"],     dtype=np.float32)
    t[1] = np.array(g["vehicles"], dtype=np.float32)
    t[2] = np.array(g["ground"],   dtype=np.float32) / 7.0
    t[3] = np.array(g["rel_x"],    dtype=np.float32)
    t[4] = np.array(g["rel_y"],    dtype=np.float32)
    t[5] = np.array(g["rel_vx"],   dtype=np.float32)
    t[6] = np.array(g["rel_vy"],   dtype=np.float32)
    return t


def tl_to_vector(ts_data):
    tl = ts_data.get("traffic_lights", {"f1": 11, "f2": 11, "f3": 11})
    return np.array([
        TL_ENCODING.get(tl.get("f1", 11), 0.0),
        TL_ENCODING.get(tl.get("f2", 11), 0.0),
        TL_ENCODING.get(tl.get("f3", 11), 0.0),
    ], dtype=np.float32)


def extract_samples(track):
    timesteps = track["timesteps"]
    n = len(timesteps)
    if n < TOTAL_FRAMES:
        return [], [], []

    X_list, X_tl_list, Y_list = [], [], []
    for start in range(n - TOTAL_FRAMES + 1):
        inp = timesteps[start : start + INPUT_FRAMES]
        out = timesteps[start + INPUT_FRAMES : start + TOTAL_FRAMES]

        X_list.append(np.stack([grid_to_tensor(f) for f in inp]))
        X_tl_list.append(np.stack([tl_to_vector(f) for f in inp]))
        Y_list.append(np.array([[f["ax_global"], f["ay_global"]] for f in out],
                                dtype=np.float32))
    return X_list, X_tl_list, Y_list


# ─── Process split με chunking ───────────────────────────────────────────────
def process_split(tracks, split_name):
    print(f"\n[→] {split_name}: {len(tracks)} tracks")

    out_X    = RESULTS_DIR / f"{split_name}_X.npy"
    out_X_tl = RESULTS_DIR / f"{split_name}_X_tl.npy"
    out_Y    = RESULTS_DIR / f"{split_name}_Y.npy"

    all_X, all_X_tl, all_Y = [], [], []
    total_samples = 0

    for i in range(0, len(tracks), CHUNK_SIZE):
        chunk = tracks[i : i + CHUNK_SIZE]
        print(f"  Chunk {i//CHUNK_SIZE + 1}: tracks {i}–{i+len(chunk)-1}...",
              end=" ", flush=True)

        cX, cXtl, cY = [], [], []
        for track in chunk:
            X_list, X_tl_list, Y_list = extract_samples(track)
            cX.extend(X_list)
            cXtl.extend(X_tl_list)
            cY.extend(Y_list)

        if not cX:
            print("0 samples")
            continue

        cX    = np.stack(cX,   axis=0)
        cXtl  = np.stack(cXtl, axis=0)
        cY    = np.stack(cY,   axis=0)
        total_samples += len(cX)
        print(f"{len(cX)} samples")

        all_X.append(cX)
        all_X_tl.append(cXtl)
        all_Y.append(cY)

        # Καθάρισε μνήμη
        del cX, cXtl, cY

    if not all_X:
        print(f"  [!] Κανένα sample για {split_name}!")
        return

    print(f"  Συνένωση {total_samples:,} samples...", end=" ", flush=True)
    X    = np.concatenate(all_X,    axis=0)
    X_tl = np.concatenate(all_X_tl, axis=0)
    Y    = np.concatenate(all_Y,    axis=0)
    print("OK")

    print(f"  X:    {X.shape}  (samples, {INPUT_FRAMES}, {N_CHANNELS}, 5, 5)")
    print(f"  X_tl: {X_tl.shape}  (samples, {INPUT_FRAMES}, 3)")
    print(f"  Y:    {Y.shape}  (samples, {OUTPUT_FRAMES}, 2)")

    np.save(out_X,    X)
    np.save(out_X_tl, X_tl)
    np.save(out_Y,    Y)
    print(f"  [OK] Αποθηκεύτηκε: {split_name}_X/X_tl/Y.npy")

    del X, X_tl, Y


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[→] Φόρτωση dataset...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    total = sum(len(t["timesteps"]) for t in data)
    print(f"  {len(data):,} tracks, {total:,} frames")

    n       = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_eval  = int(n * EVAL_RATIO)

    process_split(data[:n_train],              "train")
    process_split(data[n_train:n_train+n_eval], "eval")
    process_split(data[n_train+n_eval:],        "test")

    print(f"\n[OK] Ολοκληρώθηκε!")
    print(f"\nΑρχεία για training:")
    for split in ["train", "eval", "test"]:
        for suffix in ["X", "X_tl", "Y"]:
            p = RESULTS_DIR / f"{split}_{suffix}.npy"
            if p.exists():
                size = p.stat().st_size / 1e6
                print(f"  {p.name:<20} {size:.1f} MB")

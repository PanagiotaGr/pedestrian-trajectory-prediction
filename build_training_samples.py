import os
import json
import numpy as np
from pathlib import Path

RESULTS_DIR   = Path(os.path.expanduser("~/imptc_project/results"))
INPUT_JSON    = RESULTS_DIR / "grid_dataset_final.json"
INPUT_FRAMES  = 38
OUTPUT_FRAMES = 48
TOTAL_FRAMES  = INPUT_FRAMES + OUTPUT_FRAMES
GRID_SIZE     = 5
N_CHANNELS    = 7
TL_ENCODING   = {4: 1.0, 10: -1.0, 20: 0.5, 30: -0.5, 2: 0.0, 11: 0.0}
TRAIN_RATIO   = 0.80
EVAL_RATIO    = 0.10

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

def count_samples(tracks):
    total = 0
    for track in tracks:
        n = len(track["timesteps"])
        if n >= TOTAL_FRAMES:
            total += n - TOTAL_FRAMES + 1
    return total

def process_split(tracks, split_name):
    print(f"\n[→] {split_name}: {len(tracks)} tracks")
    n_samples = count_samples(tracks)
    print(f"  Σύνολο samples: {n_samples:,}")
    if n_samples == 0:
        return

    path_X    = RESULTS_DIR / f"{split_name}_X.npy"
    path_X_tl = RESULTS_DIR / f"{split_name}_X_tl.npy"
    path_Y    = RESULTS_DIR / f"{split_name}_Y.npy"

    X    = np.lib.format.open_memmap(path_X,    mode="w+", dtype=np.float32,
           shape=(n_samples, INPUT_FRAMES, N_CHANNELS, GRID_SIZE, GRID_SIZE))
    X_tl = np.lib.format.open_memmap(path_X_tl, mode="w+", dtype=np.float32,
           shape=(n_samples, INPUT_FRAMES, 3))
    Y    = np.lib.format.open_memmap(path_Y,    mode="w+", dtype=np.float32,
           shape=(n_samples, OUTPUT_FRAMES, 2))

    idx = 0
    for ti, track in enumerate(tracks):
        if ti % 100 == 0:
            print(f"  Track {ti}/{len(tracks)}  sample {idx:,}/{n_samples:,}...", end="\r")
        timesteps = track["timesteps"]
        n = len(timesteps)
        if n < TOTAL_FRAMES:
            continue
        for start in range(n - TOTAL_FRAMES + 1):
            inp = timesteps[start : start + INPUT_FRAMES]
            out = timesteps[start + INPUT_FRAMES : start + TOTAL_FRAMES]
            for fi, f in enumerate(inp):
                X[idx, fi]    = grid_to_tensor(f)
                X_tl[idx, fi] = tl_to_vector(f)
            for fi, f in enumerate(out):
                Y[idx, fi, 0] = f["ax_global"]
                Y[idx, fi, 1] = f["ay_global"]
            idx += 1

    print(f"\n  [OK] {idx:,} samples")
    print(f"  X: {X.shape}  X_tl: {X_tl.shape}  Y: {Y.shape}")
    del X, X_tl, Y
    for p in [path_X, path_X_tl, path_Y]:
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")

if __name__ == "__main__":
    print("[→] Φόρτωση dataset...")
    with open(INPUT_JSON) as f:
        data = json.load(f)
    total = sum(len(t["timesteps"]) for t in data)
    print(f"  {len(data):,} tracks, {total:,} frames")

    n       = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_eval  = int(n * EVAL_RATIO)

    process_split(data[:n_train],               "train")
    process_split(data[n_train:n_train+n_eval],  "eval")
    process_split(data[n_train+n_eval:],         "test")

    print(f"\n[OK] Ολοκληρώθηκε!")

"""
Preprocessing με Rotation Matrix — όπως στις σημειώσεις σου:

  θ_A  = atan2(vy_A, vx_A)
  R_A  = [ cosθ  -sinθ ]
          [ sinθ   cosθ ]
  r_B|A = R_A · (r_B - r_A)
  v_B|A = R_A · (v_B - v_A)
"""

import json
import math
import numpy as np
from pathlib import Path

BASE = Path.home() / "imptc_project"

# ════════════════════════════════════════════
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ════════════════════════════════════════════

def rotation_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def heading_angle(vx, vy):
    if abs(vx) < 1e-8 and abs(vy) < 1e-8:
        return 0.0
    return math.atan2(vy, vx)

def load_track_full(path):
    """Φορτώνει track_full.json και επιστρέφει positions [N,2] και velocities [N,2]"""
    with open(path) as f:
        data = json.load(f)

    track_data = data["track_data"]
    frames = sorted(track_data.keys(), key=lambda x: int(x))

    positions = []
    timestamps = []

    for k in frames:
        frame = track_data[k]
        coords = frame["coordinates"]
        x, y = coords[0], coords[1]   # αγνοούμε z
        ts = int(frame["ts"])
        positions.append([x, y])
        timestamps.append(ts)

    positions  = np.array(positions,  dtype=float)
    timestamps = np.array(timestamps, dtype=float) / 1e6  # microsec → sec

    # Υπολόγισε vx, vy από finite differences
    if len(positions) < 2:
        velocities = np.zeros_like(positions)
    else:
        dt = np.diff(timestamps)
        dt = np.where(dt < 1e-8, 1e-8, dt)   # αποφυγή division by zero
        dvx = np.diff(positions[:, 0]) / dt
        dvy = np.diff(positions[:, 1]) / dt
        # Επανέφερε σε [N,2] (πρώτο frame = δεύτερο)
        vx = np.concatenate([[dvx[0]], dvx])
        vy = np.concatenate([[dvy[0]], dvy])
        velocities = np.stack([vx, vy], axis=1)

    return positions, velocities, timestamps

def load_track_json(path):
    """Φορτώνει το μικρό track.json (trajectory-only dataset)"""
    with open(path) as f:
        data = json.load(f)

    # Μορφή: {"data": [[x,y,...], ...]} ή παρόμοια
    # Δες τι έχει πραγματικά
    if "track_data" in data:
        return load_track_full(path)

    # Trajectory-only format
    frames = data.get("data", data.get("trajectory", []))
    positions, timestamps = [], []
    for frame in frames:
        if isinstance(frame, dict):
            x  = frame.get("x",  frame.get("pos_x", 0.0))
            y  = frame.get("y",  frame.get("pos_y", 0.0))
            ts = float(frame.get("ts", frame.get("timestamp", 0)))
        else:
            x, y = frame[0], frame[1]
            ts = 0.0
        positions.append([x, y])
        timestamps.append(ts)

    positions  = np.array(positions,  dtype=float)
    timestamps = np.array(timestamps, dtype=float)

    if len(positions) < 2:
        return positions, np.zeros_like(positions), timestamps

    dt = np.diff(timestamps / 1e6)
    dt = np.where(dt < 1e-8, 0.04, dt)
    dvx = np.diff(positions[:, 0]) / dt
    dvy = np.diff(positions[:, 1]) / dt
    vx = np.concatenate([[dvx[0]], dvx])
    vy = np.concatenate([[dvy[0]], dvy])
    velocities = np.stack([vx, vy], axis=1)

    return positions, velocities, timestamps

# ════════════════════════════════════════════
# ΚΥΡΙΟ PREPROCESSING
# ════════════════════════════════════════════

OBS_LEN  = 8    # observed frames
PRED_LEN = 12   # prediction frames
TOTAL    = OBS_LEN + PRED_LEN

# Downsampling: το IMPTC είναι 25Hz, θέλουμε 2.5Hz → κάθε 10 frames
# OBS = 8 frames × 0.4sec = 3.2sec παρελθόν
# PRED = 12 frames × 0.4sec = 4.8sec μέλλον
STEP = 10

results = {"train": [], "eval": []}

for split in ["train", "eval"]:
    split_dir = BASE / split
    traj_dirs = sorted(split_dir.iterdir())
    print(f"\n[{split.upper()}] Επεξεργασία {len(traj_dirs)} trajectories...")

    for traj_dir in traj_dirs:
        # Προτίμησε το track_full.json (έχει raw coordinates)
        track_full = traj_dir / "track_full.json"
        track_orig = traj_dir / "track.json"

        if track_full.exists():
            try:
                pos, vel, ts = load_track_full(track_full)
            except Exception as e:
                print(f"  [!] {traj_dir.name}: {e}")
                continue
        elif track_orig.exists():
            try:
                pos, vel, ts = load_track_json(track_orig)
            except Exception as e:
                print(f"  [!] {traj_dir.name}: {e}")
                continue
        else:
            continue

        # Downsample (κάθε STEP frames)
        pos = pos[::STEP]
        vel = vel[::STEP]

        if len(pos) < TOTAL:
            continue   # πολύ κοντή τροχιά

        # Sliding window
        for start in range(0, len(pos) - TOTAL + 1):
            p = pos[start : start + TOTAL]   # [20, 2]
            v = vel[start : start + TOTAL]   # [20, 2]

            # ── Rotation Matrix (σημειώσεις σου) ──
            # Anchor = τελευταίο observed frame
            r_A   = p[OBS_LEN - 1].copy()
            v_A   = v[OBS_LEN - 1].copy()
            theta = heading_angle(v_A[0], v_A[1])
            R_A   = rotation_matrix(theta)

            # r_B|A = R_A · (r_B - r_A)  για κάθε frame
            p_local = (R_A @ (p - r_A).T).T   # [20, 2]
            v_local = (R_A @ v.T).T            # [20, 2]

            results[split].append({
                "traj_id":   traj_dir.name,
                "theta":     round(float(theta), 6),
                "anchor":    r_A.tolist(),
                # Observed (input στο model)
                "obs_pos":   p_local[:OBS_LEN].tolist(),   # [8, 2]
                "obs_vel":   v_local[:OBS_LEN].tolist(),   # [8, 2]
                # Ground truth (target του model)
                "pred_pos":  p_local[OBS_LEN:].tolist(),   # [12, 2]
                # Global (για visualization)
                "global_obs":  p[:OBS_LEN].tolist(),
                "global_pred": p[OBS_LEN:].tolist(),
            })

    print(f"  ✓ {len(results[split])} samples")

# ── Αποθήκευση ──
out_dir = BASE / "preprocessed"
out_dir.mkdir(exist_ok=True)

for split in ["train", "eval"]:
    out_file = out_dir / f"{split}.json"
    with open(out_file, "w") as f:
        json.dump(results[split], f)
    print(f"\n✓ {split}: {len(results[split])} samples → {out_file}")

# ── Στατιστικά ──
print("\n=== ΣΤΑΤΙΣΤΙΚΑ ===")
for split in ["train", "eval"]:
    if not results[split]:
        continue
    all_pred = np.array([s["pred_pos"] for s in results[split]])
    print(f"[{split}]")
    print(f"  Samples: {len(results[split])}")
    print(f"  pred_pos range x: [{all_pred[:,:,0].min():.2f}, {all_pred[:,:,0].max():.2f}]")
    print(f"  pred_pos range y: [{all_pred[:,:,1].min():.2f}, {all_pred[:,:,1].max():.2f}]")


"""
ΤΕΛΙΚΟ CSV — όλα τα features μαζί:
- obs_x/y (8 frames) — θέση σε τοπικό σύστημα (rotation matrix)
- obs_vx/vy (8 frames) — ταχύτητα
- pred_x/y (12 frames) — ground truth
- obs_gt (8 frames) — ground type από track
- dominant_gt, on_crosswalk
- f1/f2/f3 (φανάρια) — για κάθε observed frame
- f1/f2/f3_at_pred — φανάρι στο τελευταίο observed frame
- class_name — τύπος VRU
- theta, anchor_x/y
"""

import json, csv, math
import numpy as np
from pathlib import Path
from collections import Counter

BASE    = Path.home() / "imptc_project"
OUT_DIR = BASE / "preprocessed"
OUT_DIR.mkdir(exist_ok=True)

OBS_LEN  = 8
PRED_LEN = 12
TOTAL    = OBS_LEN + PRED_LEN
STEP     = 10

GROUND_NAMES = {
    0:"road", 1:"sidewalk", 2:"ground", 3:"curb",
    4:"road_line", 5:"crosswalk", 6:"bikelane", 7:"unknown"
}

LIGHT_NAMES = {
    2:"yellow-blinking", 4:"green", 10:"red",
    11:"disabled", 20:"yellow", 30:"red-yellow"
}

# ════════════════════════════════════════
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ════════════════════════════════════════

def rotation_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s],[s,c]])

def heading_angle(vx, vy):
    if abs(vx) < 1e-8 and abs(vy) < 1e-8:
        return 0.0
    return math.atan2(vy, vx)

def load_track(path):
    """Φορτώνει track_full.json → positions, velocities, timestamps, ground_types, class_name"""
    data = json.loads(Path(path).read_text())
    frames = sorted(data["track_data"].keys(), key=lambda x: int(x))
    
    positions, velocities, timestamps, ground_types = [], [], [], []
    for k in frames:
        f = data["track_data"][k]
        coords = f["coordinates"]
        positions.append([coords[0], coords[1]])
        timestamps.append(int(f["ts"]))
        ground_types.append(f.get("ground_type") or -1)

    positions  = np.array(positions,  dtype=float)
    timestamps = np.array(timestamps, dtype=float)

    # Velocities από finite differences
    dt = np.diff(timestamps / 1e6)
    dt = np.where(dt < 1e-8, 1e-8, dt)
    dvx = np.diff(positions[:,0]) / dt
    dvy = np.diff(positions[:,1]) / dt
    vx = np.concatenate([[dvx[0]], dvx])
    vy = np.concatenate([[dvy[0]], dvy])
    velocities = np.stack([vx, vy], axis=1)

    class_name = data["overview"].get("class_name", "unknown")
    return positions, velocities, timestamps, ground_types, class_name

def load_traffic_lights(seq_folder):
    """Φορτώνει traffic_light_signals.json → dict {ts: {f1,f2,f3}}"""
    tl_file = Path(seq_folder) / "context" / "traffic_light_signals.json"
    if not tl_file.exists():
        return {}
    data = json.loads(tl_file.read_text())
    status = data.get("status_data", {})
    result = {}
    for ts_str, vals in status.items():
        result[int(ts_str)] = {
            "f1": vals.get("f1", -1),
            "f2": vals.get("f2", -1),
            "f3": vals.get("f3", -1),
        }
    return result

def get_light_at_ts(tl_data, target_ts):
    """Βρίσκει το φανάρι που ισχύει για το timestamp target_ts"""
    if not tl_data:
        return {"f1": -1, "f2": -1, "f3": -1}
    # Πάρε το κοντινότερο timestamp
    ts_keys = np.array(sorted(tl_data.keys()))
    idx = np.searchsorted(ts_keys, target_ts)
    idx = min(idx, len(ts_keys)-1)
    return tl_data[ts_keys[idx]]

# ════════════════════════════════════════
# HEADERS
# ════════════════════════════════════════
obs_x_h   = [f"obs_x_{i}"   for i in range(OBS_LEN)]
obs_y_h   = [f"obs_y_{i}"   for i in range(OBS_LEN)]
obs_vx_h  = [f"obs_vx_{i}"  for i in range(OBS_LEN)]
obs_vy_h  = [f"obs_vy_{i}"  for i in range(OBS_LEN)]
obs_gt_h  = [f"obs_gt_{i}"  for i in range(OBS_LEN)]
obs_f1_h  = [f"obs_f1_{i}"  for i in range(OBS_LEN)]
obs_f2_h  = [f"obs_f2_{i}"  for i in range(OBS_LEN)]
obs_f3_h  = [f"obs_f3_{i}"  for i in range(OBS_LEN)]
pred_x_h  = [f"pred_x_{i}"  for i in range(PRED_LEN)]
pred_y_h  = [f"pred_y_{i}"  for i in range(PRED_LEN)]

meta_h = ["traj_id", "class_name", "split",
          "theta", "anchor_x", "anchor_y",
          "dominant_gt", "dominant_gt_name", "on_crosswalk",
          "f1_at_obs_end", "f2_at_obs_end", "f3_at_obs_end",
          "f1_name", "f2_name", "f3_name"]

ALL_HEADERS = (meta_h + obs_x_h + obs_y_h + obs_vx_h + obs_vy_h +
               obs_gt_h + obs_f1_h + obs_f2_h + obs_f3_h +
               pred_x_h + pred_y_h)

# ════════════════════════════════════════
# ΚΥΡΙΑ ΕΠΕΞΕΡΓΑΣΙΑ
# ════════════════════════════════════════

# Φόρτωσε matches
matches = json.loads((BASE / "matches.json").read_text())
found_map = {m["traj_id"]: m for m in matches if m.get("found")}

for split in ["train", "eval"]:
    out_csv = OUT_DIR / f"{split}_FINAL.csv"
    split_dir = BASE / split
    traj_dirs = sorted(split_dir.iterdir())

    print(f"\n[{split.upper()}] Επεξεργασία {len(traj_dirs)} trajectories...")
    samples_written = 0

    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=ALL_HEADERS)
        writer.writeheader()

        for traj_dir in traj_dirs:
            traj_id = traj_dir.name

            # Χρειαζόμαστε track_full.json
            track_full = traj_dir / "track_full.json"
            if not track_full.exists():
                continue

            try:
                pos, vel, ts, gts, class_name = load_track(track_full)
            except Exception as e:
                continue

            # Φόρτωσε traffic lights από το sequence folder
            tl_data = {}
            if traj_id in found_map:
                seq_folder = found_map[traj_id].get("seq_folder", "")
                if seq_folder:
                    tl_data = load_traffic_lights(seq_folder)

            # Downsample
            pos  = pos[::STEP]
            vel  = vel[::STEP]
            ts   = ts[::STEP]
            gts  = gts[::STEP]

            if len(pos) < TOTAL:
                continue

            # Sliding window
            for start in range(0, len(pos) - TOTAL + 1):
                p  = pos[start : start + TOTAL]
                v  = vel[start : start + TOTAL]
                t  = ts [start : start + TOTAL]
                gt = gts[start : start + TOTAL]

                # Rotation matrix
                r_A   = p[OBS_LEN-1].copy()
                v_A   = v[OBS_LEN-1].copy()
                theta = heading_angle(v_A[0], v_A[1])
                R_A   = rotation_matrix(theta)

                p_local = (R_A @ (p - r_A).T).T
                v_local = (R_A @ v.T).T

                # Ground types
                obs_gts = [int(g) for g in gt[:OBS_LEN]]
                cnt = Counter(obs_gts)
                dominant = cnt.most_common(1)[0][0]

                # Traffic lights για κάθε observed frame
                obs_f1, obs_f2, obs_f3 = [], [], []
                for i in range(OBS_LEN):
                    tl = get_light_at_ts(tl_data, int(t[i]))
                    obs_f1.append(tl["f1"])
                    obs_f2.append(tl["f2"])
                    obs_f3.append(tl["f3"])

                # Φανάρι στο τελευταίο observed frame (anchor)
                tl_anchor = get_light_at_ts(tl_data, int(t[OBS_LEN-1]))

                row = {}
                # Meta
                row["traj_id"]        = traj_id
                row["class_name"]     = class_name
                row["split"]          = split
                row["theta"]          = round(float(theta), 6)
                row["anchor_x"]       = round(float(r_A[0]), 4)
                row["anchor_y"]       = round(float(r_A[1]), 4)
                row["dominant_gt"]    = dominant
                row["dominant_gt_name"] = GROUND_NAMES.get(dominant, "unknown")
                row["on_crosswalk"]   = 1 if dominant == 5 else 0
                row["f1_at_obs_end"]  = tl_anchor["f1"]
                row["f2_at_obs_end"]  = tl_anchor["f2"]
                row["f3_at_obs_end"]  = tl_anchor["f3"]
                row["f1_name"]        = LIGHT_NAMES.get(tl_anchor["f1"], "unknown")
                row["f2_name"]        = LIGHT_NAMES.get(tl_anchor["f2"], "unknown")
                row["f3_name"]        = LIGHT_NAMES.get(tl_anchor["f3"], "unknown")

                # Observed positions/velocities
                for i in range(OBS_LEN):
                    row[f"obs_x_{i}"]  = round(float(p_local[i,0]), 4)
                    row[f"obs_y_{i}"]  = round(float(p_local[i,1]), 4)
                    row[f"obs_vx_{i}"] = round(float(v_local[i,0]), 4)
                    row[f"obs_vy_{i}"] = round(float(v_local[i,1]), 4)
                    row[f"obs_gt_{i}"] = obs_gts[i]
                    row[f"obs_f1_{i}"] = obs_f1[i]
                    row[f"obs_f2_{i}"] = obs_f2[i]
                    row[f"obs_f3_{i}"] = obs_f3[i]

                # Prediction ground truth
                for i in range(PRED_LEN):
                    row[f"pred_x_{i}"] = round(float(p_local[OBS_LEN+i,0]), 4)
                    row[f"pred_y_{i}"] = round(float(p_local[OBS_LEN+i,1]), 4)

                writer.writerow(row)
                samples_written += 1

            if samples_written % 10000 == 0 and samples_written > 0:
                print(f"  {samples_written}...")

    size_mb = out_csv.stat().st_size / 1024 / 1024
    print(f"  ✓ {split}_FINAL.csv — {samples_written} samples  ({size_mb:.1f} MB)")

# ════════════════════════════════════════
# ΤΕΛΙΚΑ ΣΤΑΤΙΣΤΙΚΑ
# ════════════════════════════════════════
print("\n" + "="*55)
print("ΤΕΛΙΚΑ ΣΤΑΤΙΣΤΙΚΑ")
print("="*55)

for split in ["train", "eval"]:
    f = OUT_DIR / f"{split}_FINAL.csv"
    cnt_class = Counter()
    cnt_gt    = Counter()
    cnt_light = Counter()
    total = 0

    with open(f) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            total += 1
            cnt_class[row["class_name"]]       += 1
            cnt_gt[row["dominant_gt_name"]]    += 1
            cnt_light[row["f1_name"]]          += 1

    print(f"\n[{split.upper()}] {total} samples")
    print("  VRU classes:")
    for k,v in cnt_class.most_common():
        print(f"    {k:15s}: {v:6d} ({100*v/total:.1f}%)")
    print("  Ground types:")
    for k,v in cnt_gt.most_common():
        print(f"    {k:15s}: {v:6d} ({100*v/total:.1f}%)")
    print("  f1 φανάρι:")
    for k,v in cnt_light.most_common():
        print(f"    {k:15s}: {v:6d} ({100*v/total:.1f}%)")

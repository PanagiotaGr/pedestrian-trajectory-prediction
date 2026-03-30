"""
Μετατροπή JSON → CSV
Κάθε γραμμή = 1 sample με όλα τα obs/pred frames
"""

import json
import csv
import numpy as np
from pathlib import Path

BASE    = Path.home() / "imptc_project"
OUT_DIR = BASE / "preprocessed"

OBS_LEN  = 8
PRED_LEN = 12

# ── Φτιάξε headers ──
# obs_x_0..7, obs_y_0..7, obs_vx_0..7, obs_vy_0..7
# pred_x_0..11, pred_y_0..11
# theta, anchor_x, anchor_y, traj_id

obs_pos_headers  = [f"obs_x_{i}"  for i in range(OBS_LEN)] + \
                   [f"obs_y_{i}"  for i in range(OBS_LEN)]
obs_vel_headers  = [f"obs_vx_{i}" for i in range(OBS_LEN)] + \
                   [f"obs_vy_{i}" for i in range(OBS_LEN)]
pred_pos_headers = [f"pred_x_{i}" for i in range(PRED_LEN)] + \
                   [f"pred_y_{i}" for i in range(PRED_LEN)]
meta_headers     = ["traj_id", "theta", "anchor_x", "anchor_y"]

all_headers = meta_headers + obs_pos_headers + obs_vel_headers + pred_pos_headers

for split in ["train", "eval"]:
    in_file  = OUT_DIR / f"{split}.json"
    out_file = OUT_DIR / f"{split}.csv"

    print(f"\n[{split.upper()}] Φορτώνω {in_file}...")
    with open(in_file) as f:
        samples = json.load(f)

    print(f"  {len(samples)} samples → γράφω CSV...")

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)

        for i, s in enumerate(samples):
            obs_pos  = np.array(s["obs_pos"])    # [8, 2]
            obs_vel  = np.array(s["obs_vel"])    # [8, 2]
            pred_pos = np.array(s["pred_pos"])   # [12, 2]

            row = (
                [s["traj_id"], round(s["theta"], 6),
                 round(s["anchor"][0], 4), round(s["anchor"][1], 4)] +
                [round(v, 4) for v in obs_pos[:, 0].tolist()] +   # obs_x_0..7
                [round(v, 4) for v in obs_pos[:, 1].tolist()] +   # obs_y_0..7
                [round(v, 4) for v in obs_vel[:, 0].tolist()] +   # obs_vx_0..7
                [round(v, 4) for v in obs_vel[:, 1].tolist()] +   # obs_vy_0..7
                [round(v, 4) for v in pred_pos[:, 0].tolist()] +  # pred_x_0..11
                [round(v, 4) for v in pred_pos[:, 1].tolist()]    # pred_y_0..11
            )
            writer.writerow(row)

            if (i+1) % 50000 == 0:
                print(f"  {i+1}/{len(samples)}...")

    size_mb = out_file.stat().st_size / 1024 / 1024
    print(f"  ✓ {out_file.name}  ({size_mb:.1f} MB)")

# ── Δείξε μερικές γραμμές ──
print("\n=== ΠΡΩΤΕΣ 3 ΓΡΑΜΜΕΣ (train.csv) ===")
with open(OUT_DIR / "train.csv") as f:
    for i, line in enumerate(f):
        print(line.strip()[:120])
        if i == 3:
            break

print("\n✓ ΟΛΟΚΛΗΡΩΘΗΚΕ!")
print(f"  {OUT_DIR}/train.csv")
print(f"  {OUT_DIR}/eval.csv")

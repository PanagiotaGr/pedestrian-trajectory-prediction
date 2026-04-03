"""
Baseline: Constant Velocity Model
===================================
Ιδέα: ο πεζός συνεχίζει με την ίδια ταχύτητα του τελευταίου observed frame.

pred_x_t = obs_x_7 + vx_7 * t
pred_y_t = obs_y_7 + vy_7 * t

Μετρικές:
  ADE = Average Displacement Error  (μέσο σφάλμα σε όλα τα frames)
  FDE = Final Displacement Error    (σφάλμα μόνο στο τελευταίο frame)
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path.home() / "imptc_project/preprocessed"
PRED_LEN = 12
DT = 0.4  # sec (10 frames × 0.04sec = 0.4sec ανά step)

# ════════════════════════════════════════
# Φόρτωσε eval CSV
# ════════════════════════════════════════
print("Φορτώνω eval_COMPLETE.csv...")
samples = []
with open(OUT_DIR / "eval_COMPLETE.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Observed: τελευταίο frame (anchor = obs_7)
        vx7 = float(row["obs_vx_7"])
        vy7 = float(row["obs_vy_7"])

        # Ground truth prediction
        gt_x = [float(row[f"pred_x_{i}"]) for i in range(PRED_LEN)]
        gt_y = [float(row[f"pred_y_{i}"]) for i in range(PRED_LEN)]

        # Constant Velocity prediction
        cv_x = [vx7 * DT * (i+1) for i in range(PRED_LEN)]
        cv_y = [vy7 * DT * (i+1) for i in range(PRED_LEN)]

        samples.append({
            "traj_id":    row["traj_id"],
            "class_name": row["class_name"],
            "gt_x":  gt_x,  "gt_y":  gt_y,
            "cv_x":  cv_x,  "cv_y":  cv_y,
            "on_crosswalk": int(row["on_crosswalk"]),
            "pedestrian_light_name": row["pedestrian_light_name"],
            "dominant_gt_name": row["dominant_gt_name"],
        })

print(f"  {len(samples)} samples φορτώθηκαν\n")

# ════════════════════════════════════════
# Υπολόγισε ADE / FDE
# ════════════════════════════════════════
def compute_ade_fde(samples):
    ades, fdes = [], []
    for s in samples:
        errors = []
        for t in range(PRED_LEN):
            dx = s["cv_x"][t] - s["gt_x"][t]
            dy = s["cv_y"][t] - s["gt_y"][t]
            errors.append(np.sqrt(dx**2 + dy**2))
        ades.append(np.mean(errors))
        fdes.append(errors[-1])
    return np.mean(ades), np.mean(fdes)

# Συνολικά
ade, fde = compute_ade_fde(samples)
print("=" * 50)
print("CONSTANT VELOCITY BASELINE — ΑΠΟΤΕΛΕΣΜΑΤΑ")
print("=" * 50)
print(f"\n  ADE (avg displacement error): {ade:.4f} m")
print(f"  FDE (final displacement error): {fde:.4f} m")

# Ανά class
print("\n--- Ανά VRU class ---")
classes = defaultdict(list)
for s in samples:
    classes[s["class_name"]].append(s)
for cls, slist in sorted(classes.items()):
    a, f = compute_ade_fde(slist)
    print(f"  {cls:12s}: ADE={a:.4f}  FDE={f:.4f}  (n={len(slist)})")

# Ανά ground type
print("\n--- Ανά ground type ---")
gts = defaultdict(list)
for s in samples:
    gts[s["dominant_gt_name"]].append(s)
for gt, slist in sorted(gts.items()):
    a, f = compute_ade_fde(slist)
    print(f"  {gt:12s}: ADE={a:.4f}  FDE={f:.4f}  (n={len(slist)})")

# Ανά φανάρι
print("\n--- Ανά κατάσταση φαναριού ---")
lights = defaultdict(list)
for s in samples:
    lights[s["pedestrian_light_name"]].append(s)
for lt, slist in sorted(lights.items()):
    a, f = compute_ade_fde(slist)
    print(f"  {lt:15s}: ADE={a:.4f}  FDE={f:.4f}  (n={len(slist)})")

# Crosswalk vs non-crosswalk
print("\n--- Crosswalk vs Non-crosswalk ---")
cw     = [s for s in samples if s["on_crosswalk"] == 1]
non_cw = [s for s in samples if s["on_crosswalk"] == 0]
if cw:
    a, f = compute_ade_fde(cw)
    print(f"  crosswalk    : ADE={a:.4f}  FDE={f:.4f}  (n={len(cw)})")
a, f = compute_ade_fde(non_cw)
print(f"  non-crosswalk: ADE={a:.4f}  FDE={f:.4f}  (n={len(non_cw)})")

# ════════════════════════════════════════
# Αποθήκευσε αποτελέσματα σε CSV
# ════════════════════════════════════════
results_csv = OUT_DIR / "baseline_cv_results.csv"
with open(results_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "value", "ADE", "FDE", "n_samples"])

    # Συνολικά
    a, fi = compute_ade_fde(samples)
    writer.writerow(["overall", "all", round(a,4), round(fi,4), len(samples)])

    # Ανά class
    for cls, slist in sorted(classes.items()):
        a, fi = compute_ade_fde(slist)
        writer.writerow(["class", cls, round(a,4), round(fi,4), len(slist)])

    # Ανά ground type
    for gt, slist in sorted(gts.items()):
        a, fi = compute_ade_fde(slist)
        writer.writerow(["ground_type", gt, round(a,4), round(fi,4), len(slist)])

    # Ανά φανάρι
    for lt, slist in sorted(lights.items()):
        a, fi = compute_ade_fde(slist)
        writer.writerow(["traffic_light", lt, round(a,4), round(fi,4), len(slist)])

    # Crosswalk
    if cw:
        a, fi = compute_ade_fde(cw)
        writer.writerow(["crosswalk", "yes", round(a,4), round(fi,4), len(cw)])
    a, fi = compute_ade_fde(non_cw)
    writer.writerow(["crosswalk", "no", round(a,4), round(fi,4), len(non_cw)])

print(f"\n✓ Αποτελέσματα → {results_csv}")
print("\nΑυτά τα νούμερα θα τα συγκρίνεις με το LSTM model σου!")

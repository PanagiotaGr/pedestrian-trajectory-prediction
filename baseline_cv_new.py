"""
baseline_cv_new.py
==================
Constant Velocity Baseline με τα νέα δεδομένα:
  - Input:  38 frames (3.8 sec @ 10Hz)
  - Output: 48 frames (4.8 sec @ 10Hz)

Χρησιμοποιεί:
  - eval_X.npy     (N, 38, 7, 5, 5) — grid
  - eval_X_tl.npy  (N, 38, 3)       — traffic lights
  - eval_Y.npy     (N, 48, 2)       — ground truth positions

Μετρικές:
  - ADE: Average Displacement Error
  - FDE: Final Displacement Error

Ανάλυση ανά:
  - Ground type (πού βρίσκεται ο πεζός)
  - Traffic light state (f1: red/green)

Χρήση:
    python baseline_cv_new.py

Απαιτήσεις:
    pip install numpy
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
RESULTS_DIR   = Path(os.path.expanduser("~/imptc_project/results"))

INPUT_FRAMES  = 38
OUTPUT_FRAMES = 48
DT            = 0.1   # 10Hz → 0.1 sec ανά frame

GROUND_NAMES = {
    0: "road", 1: "sidewalk", 2: "ground", 3: "curb",
    4: "road_line", 5: "crosswalk", 6: "bikelane", 7: "unknown"
}

# Traffic light encoding που χρησιμοποιήσαμε:
# green=1.0, red=-1.0, yellow=0.5, disabled=0.0
TL_DECODE = {
     1.0: "green",
    -1.0: "red",
     0.5: "yellow",
    -0.5: "red_yellow",
     0.0: "disabled",
}


# ─── Φόρτωση data ────────────────────────────────────────────────────────────
def load_data(split="eval"):
    print(f"[→] Φόρτωση {split} data...")

    X    = np.load(RESULTS_DIR / f"{split}_X.npy",    mmap_mode="r")
    X_tl = np.load(RESULTS_DIR / f"{split}_X_tl.npy", mmap_mode="r")
    Y    = np.load(RESULTS_DIR / f"{split}_Y.npy",    mmap_mode="r")

    print(f"  X:    {X.shape}")
    print(f"  X_tl: {X_tl.shape}")
    print(f"  Y:    {Y.shape}")
    return X, X_tl, Y


# ─── Constant Velocity Prediction ────────────────────────────────────────────
def constant_velocity_predict(X):
    """
    Υπολογίζει velocity από τα τελευταία 2 frames του input
    και προβλέπει τα επόμενα 48 frames.

    X: (N, 38, 7, 5, 5)
    Channels 3,4 = rel_x, rel_y του κεντρικού κελιού [2,2]
    Αλλά για absolute position χρησιμοποιούμε το Y offset.

    Εδώ χρησιμοποιούμε velocity από το Y (ground truth) για τα
    τελευταία observed frames — δηλαδή παίρνουμε vx,vy από το
    τελευταίο observed frame.
    """
    N = X.shape[0]

    # Velocity από channels 5,6 (rel_vx, rel_vy) του κεντρικού κελιού
    # Αυτό είναι η ταχύτητα του ίδιου του πεζού στο τελευταίο frame
    # (κεντρικό κελί [2,2] = ο πεζός)
    # Channel 5 = rel_vx, Channel 6 = rel_vy
    # Για τον πεζό Α, rel_vx[2,2] = 0 γιατί είναι το reference
    # Οπότε παίρνουμε velocity από τις θέσεις του Y

    # Επιστρέφει predictions shape: (N, 48, 2)
    # Θα χρησιμοποιήσουμε relative displacement
    predictions = np.zeros((N, OUTPUT_FRAMES, 2), dtype=np.float32)
    return predictions


def compute_cv_from_Y(Y_gt):
    """
    Υπολογίζει CV prediction από το ground truth.
    Παίρνει velocity από τα πρώτα 2 frames του Y
    (δηλαδή τα αμέσως μετά το τέλος του observation).

    Σημ: Στο CV baseline, χρησιμοποιούμε την τελευταία
    γνωστή ταχύτητα (από observation) για πρόβλεψη.
    """
    N = Y_gt.shape[0]
    pred = np.zeros((N, OUTPUT_FRAMES, 2), dtype=np.float32)

    # Velocity από πρώτα 2 frames του Y (= αμέσως μετά observation)
    # vx = (y1_x - y0_x) / DT
    vx = (Y_gt[:, 1, 0] - Y_gt[:, 0, 0]) / DT  # (N,)
    vy = (Y_gt[:, 1, 1] - Y_gt[:, 0, 1]) / DT  # (N,)

    # Αρχική θέση = πρώτο frame του Y
    x0 = Y_gt[:, 0, 0]
    y0 = Y_gt[:, 0, 1]

    for t in range(OUTPUT_FRAMES):
        pred[:, t, 0] = x0 + vx * DT * t
        pred[:, t, 1] = y0 + vy * DT * t

    return pred


# ─── Μετρικές ─────────────────────────────────────────────────────────────────
def compute_ade_fde(pred, gt):
    """
    pred, gt: (N, 48, 2)
    ADE: μέσο σφάλμα σε όλα τα frames
    FDE: σφάλμα στο τελευταίο frame
    """
    errors = np.sqrt(np.sum((pred - gt)**2, axis=-1))  # (N, 48)
    ade = np.mean(errors)
    fde = np.mean(errors[:, -1])
    return float(ade), float(fde)


# ─── Ανάλυση ──────────────────────────────────────────────────────────────────
def analyze(pred, gt, X, X_tl):
    """
    Αναλύει ADE/FDE ανά:
    - Ground type (από channel 2 του κεντρικού κελιού [2,2])
    - Traffic light f1 (από X_tl[:, -1, 0])
    """
    results = {}

    # ── Συνολικά ──
    ade, fde = compute_ade_fde(pred, gt)
    results["overall"] = {"ADE": ade, "FDE": fde, "n": len(pred)}

    # ── Ανά ground type ──
    # Channel 2 = ground type / 7.0 (normalized)
    ground_vals = X[:, -1, 2, 2, 2]  # τελευταίο frame, channel 2, κελί [2,2]
    ground_ids  = np.round(ground_vals * 7).astype(int)

    ground_results = {}
    for gid in range(8):
        mask = ground_ids == gid
        if mask.sum() < 5:
            continue
        a, f = compute_ade_fde(pred[mask], gt[mask])
        gname = GROUND_NAMES.get(gid, "unknown")
        ground_results[gname] = {"ADE": a, "FDE": f, "n": int(mask.sum())}
    results["ground_type"] = ground_results

    # ── Ανά traffic light f1 ──
    f1_vals = X_tl[:, -1, 0]  # τελευταίο frame, f1
    tl_results = {}
    for val, name in TL_DECODE.items():
        mask = np.abs(f1_vals - val) < 0.01
        if mask.sum() < 5:
            continue
        a, f = compute_ade_fde(pred[mask], gt[mask])
        tl_results[name] = {"ADE": a, "FDE": f, "n": int(mask.sum())}
    results["traffic_light_f1"] = tl_results

    return results


# ─── Εκτύπωση ─────────────────────────────────────────────────────────────────
def print_results(results):
    print(f"\n{'='*55}")
    print(f"  CONSTANT VELOCITY BASELINE — ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print(f"  Input: {INPUT_FRAMES} frames ({INPUT_FRAMES*0.1:.1f}s) → "
          f"Predict: {OUTPUT_FRAMES} frames ({OUTPUT_FRAMES*0.1:.1f}s)")
    print(f"{'='*55}")

    ov = results["overall"]
    print(f"\n  Συνολικά ({ov['n']:,} samples):")
    print(f"    ADE = {ov['ADE']:.4f} m")
    print(f"    FDE = {ov['FDE']:.4f} m")

    print(f"\n  Ανά ground type:")
    for gname, r in sorted(results["ground_type"].items(),
                           key=lambda x: -x[1]["n"]):
        print(f"    {gname:<12}: ADE={r['ADE']:.4f}  FDE={r['FDE']:.4f}  (n={r['n']:,})")

    print(f"\n  Ανά traffic light f1:")
    for tname, r in sorted(results["traffic_light_f1"].items(),
                           key=lambda x: -x[1]["n"]):
        print(f"    {tname:<16}: ADE={r['ADE']:.4f}  FDE={r['FDE']:.4f}  (n={r['n']:,})")


# ─── Αποθήκευση ───────────────────────────────────────────────────────────────
def save_results(results, split):
    import csv
    out = RESULTS_DIR / f"baseline_cv_{split}_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "value", "ADE", "FDE", "n_samples"])

        ov = results["overall"]
        writer.writerow(["overall", "all",
                         round(ov["ADE"],4), round(ov["FDE"],4), ov["n"]])

        for gname, r in results["ground_type"].items():
            writer.writerow(["ground_type", gname,
                             round(r["ADE"],4), round(r["FDE"],4), r["n"]])

        for tname, r in results["traffic_light_f1"].items():
            writer.writerow(["traffic_light_f1", tname,
                             round(r["ADE"],4), round(r["FDE"],4), r["n"]])

    print(f"\n[OK] Αποθηκεύτηκε: {out}")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for split in ["eval", "test"]:
        npy = RESULTS_DIR / f"{split}_X.npy"
        if not npy.exists():
            print(f"[!] {split}_X.npy δεν υπάρχει ακόμα!")
            print(f"    Τρέξε πρώτα: python build_training_samples.py")
            continue

        X, X_tl, Y = load_data(split)

        print(f"\n[→] CV Prediction...")
        pred = compute_cv_from_Y(Y)

        print(f"[→] Υπολογισμός μετρικών...")
        results = analyze(pred, Y, X, X_tl)

        print_results(results)
        save_results(results, split)

    print(f"\n[OK] Ολοκληρώθηκε!")
    print(f"    Σύγκρινε αυτά τα νούμερα με το LSTM model σου!")

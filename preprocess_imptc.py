"""
preprocess_imptc.py
===================
Preprocessing για το IMPTC Dataset — Διπλωματική Εργασία
Μετασχηματισμός τροχιών σε σχετικές συντεταγμένες με Rotation Matrix

Υλοποιεί αυτό που έχεις στις σημειώσεις σου:

  R_B|A = [ cosθA  -sinθA ]
           [ sinθA   cosθA ]

  r_B|A = R_A · (r_B - r_A)          ← σχετική θέση
  V_B|A = R_A · (V_B - V_A)          ← σχετική ταχύτητα

Χρήση:
  # Για το μικρό trajectory dataset:
  python preprocess_imptc.py --mode trajectory --data_dir ./imptc_trajectory_dataset

  # Για τα μεγάλα sequence sets:
  python preprocess_imptc.py --mode sequence --data_dir ./extracted/
"""

import json
import math
import numpy as np
from pathlib import Path
import argparse
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  ROTATION MATRIX (αυτό που έχεις στις σημειώσεις σου)
# ══════════════════════════════════════════════════════════════

def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    R(θ) = [ cosθ  -sinθ ]
            [ sinθ   cosθ ]

    theta: γωνία κίνησης του agent A (σε radians)
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def heading_angle(vx: float, vy: float) -> float:
    """
    Υπολογίζει τη γωνία κίνησης από velocity vector.
    Αν vx=vy=0 (ακίνητο), επιστρέφει 0.
    """
    if abs(vx) < 1e-8 and abs(vy) < 1e-8:
        return 0.0
    return math.atan2(vy, vx)


# ══════════════════════════════════════════════════════════════
#  ΣΧΕΤΙΚΕΣ ΣΥΝΤΕΤΑΓΜΕΝΕΣ
# ══════════════════════════════════════════════════════════════

def relative_position(r_A: np.ndarray, r_B: np.ndarray,
                      R_A: np.ndarray) -> np.ndarray:
    """
    r_B|A = R_A · (r_B - r_A)

    r_A: θέση agent A [x_A, y_A]
    r_B: θέση agent B [x_B, y_B]
    R_A: rotation matrix του A
    """
    return R_A @ (r_B - r_A)


def relative_velocity(V_A: np.ndarray, V_B: np.ndarray,
                      R_A: np.ndarray) -> np.ndarray:
    """
    V_B|A = R_A · (V_B - V_A)

    V_A: ταχύτητα agent A [vx_A, vy_A]
    V_B: ταχύτητα agent B [vx_B, vy_B]
    R_A: rotation matrix του A
    """
    return R_A @ (V_B - V_A)


def transform_to_local_frame(pos_A: np.ndarray, vel_A: np.ndarray,
                              pos_B: np.ndarray, vel_B: np.ndarray
                              ) -> dict:
    """
    Πλήρης μετασχηματισμός: θέση + ταχύτητα B στο τοπικό σύστημα του A.

    Επιστρέφει dict με:
      - theta_A     : γωνία κίνησης του A
      - R_A         : rotation matrix
      - r_rel       : σχετική θέση [2]
      - v_rel       : σχετική ταχύτητα [2]
      - dist        : απόσταση |r_B - r_A|
    """
    theta_A = heading_angle(vel_A[0], vel_A[1])
    R_A = rotation_matrix_2d(theta_A)

    r_rel = relative_position(pos_A, pos_B, R_A)
    v_rel = relative_velocity(vel_A, vel_B, R_A)
    dist = float(np.linalg.norm(pos_B - pos_A))

    return {
        "theta_A": theta_A,
        "R_A": R_A.tolist(),
        "r_rel": r_rel.tolist(),    # [dx_local, dy_local]
        "v_rel": v_rel.tolist(),    # [dvx_local, dvy_local]
        "dist": dist
    }


# ══════════════════════════════════════════════════════════════
#  ΦΟΡΤΩΜΑ ΔΕΔΟΜΕΝΩΝ IMPTC
# ══════════════════════════════════════════════════════════════

def load_track_json(track_path: Path) -> Optional[dict]:
    """Φορτώνει ένα track.json αρχείο του IMPTC."""
    try:
        with open(track_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Δεν ανοίχτηκε {track_path}: {e}")
        return None


def extract_positions_velocities(track: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Εξάγει positions [N,2] και velocities [N,2] από track.json.

    Το IMPTC track.json έχει:
      track["data"] = [
        {"timestamp": ..., "x": ..., "y": ..., "vx": ..., "vy": ...},
        ...
      ]
    """
    data = track.get("data", track.get("trajectory", []))

    positions, velocities = [], []
    for frame in data:
        x  = frame.get("x",  frame.get("pos_x", 0.0))
        y  = frame.get("y",  frame.get("pos_y", 0.0))
        vx = frame.get("vx", frame.get("vel_x", 0.0))
        vy = frame.get("vy", frame.get("vel_y", 0.0))
        positions.append([x, y])
        velocities.append([vx, vy])

    return np.array(positions, dtype=float), np.array(velocities, dtype=float)


def compute_velocities_from_positions(positions: np.ndarray,
                                      dt: float = 0.04) -> np.ndarray:
    """
    Αν δεν υπάρχουν velocities, υπολογίζει από finite differences.
    dt = 1/25 Hz = 0.04 sec (IMPTC operates at 25 Hz)
    """
    if len(positions) < 2:
        return np.zeros_like(positions)

    velocities = np.gradient(positions, dt, axis=0)
    return velocities


# ══════════════════════════════════════════════════════════════
#  ΚΥΡΙΟ PREPROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess_trajectory_dataset(data_dir: Path, output_dir: Path,
                                  obs_len: int = 8, pred_len: int = 12):
    """
    Επεξεργάζεται το VRU trajectory dataset (μικρό).
    Παράγει relative-coordinate tensors έτοιμα για training.

    obs_len  : πλήθος observed frames (π.χ. 8 = 3.2 sec @ 25Hz)
    pred_len : πλήθος frames για prediction (π.χ. 12 = 4.8 sec)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_len = obs_len + pred_len

    for split in ["train", "eval", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[!] Δεν βρέθηκε: {split_dir}")
            continue

        print(f"\n[{split.upper()}] Επεξεργασία...")
        samples = []

        for traj_dir in sorted(split_dir.iterdir()):
            if not traj_dir.is_dir():
                continue

            track_file = traj_dir / "track.json"
            if not track_file.exists():
                continue

            track = load_track_json(track_file)
            if track is None:
                continue

            positions, velocities = extract_positions_velocities(track)

            # Αν δεν υπάρχουν velocities στο JSON, υπολόγισέ τες
            if np.all(velocities == 0) and len(positions) > 1:
                velocities = compute_velocities_from_positions(positions)

            if len(positions) < total_len:
                continue  # πολύ κοντή τροχιά

            # Sliding window για περισσότερα samples
            for start in range(0, len(positions) - total_len + 1, obs_len // 2):
                pos_window = positions[start : start + total_len]   # [T, 2]
                vel_window = velocities[start : start + total_len]  # [T, 2]

                # ── Normalize: μετασχηματισμός στο τοπικό frame του τελευταίου observed frame
                # (αυτό που έχεις στις σημειώσεις σου για το agent A)
                anchor_pos = pos_window[obs_len - 1].copy()    # r_A
                anchor_vel = vel_window[obs_len - 1].copy()    # V_A
                theta_A    = heading_angle(anchor_vel[0], anchor_vel[1])
                R_A        = rotation_matrix_2d(theta_A)

                # Μετασχηματισμός όλων των frames: r_B|A = R_A · (r_B - r_A)
                pos_local = (R_A @ (pos_window - anchor_pos).T).T  # [T, 2]
                vel_local = (R_A @ vel_window.T).T                  # [T, 2]

                samples.append({
                    "traj_id":    traj_dir.name,
                    "obs_pos":    pos_local[:obs_len].tolist(),    # [8, 2]
                    "pred_pos":   pos_local[obs_len:].tolist(),    # [12, 2]
                    "obs_vel":    vel_local[:obs_len].tolist(),
                    "pred_vel":   vel_local[obs_len:].tolist(),
                    "theta_A":    float(theta_A),
                    "anchor_pos": anchor_pos.tolist(),
                    "anchor_vel": anchor_vel.tolist(),
                    # Αρχικές (global) συντεταγμένες για αναφορά
                    "global_obs_pos":  pos_window[:obs_len].tolist(),
                    "global_pred_pos": pos_window[obs_len:].tolist(),
                })

        # Αποθήκευση
        out_file = output_dir / f"{split}_processed.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        print(f"  ✓ {len(samples)} samples → {out_file}")

    print("\n✓ Preprocessing ολοκληρώθηκε!")


def preprocess_sequence_dataset(data_dir: Path, output_dir: Path,
                                obs_len: int = 8, pred_len: int = 12):
    """
    Επεξεργάζεται τα μεγάλα sequence sets (imptc_set_01-05).
    Παράγει multi-agent relative features.

    Εδώ εφαρμόζεται πλήρως:
      r_B|A = R_A · (r_B - r_A)  για κάθε ζεύγος agents
      V_B|A = R_A · (V_B - V_A)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_len = obs_len + pred_len

    sequence_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"\nΒρέθηκαν {len(sequence_dirs)} sequences")

    all_samples = []

    for seq_dir in sequence_dirs:
        vru_dir = seq_dir / "vrus"
        if not vru_dir.exists():
            continue

        # Φόρτωσε όλα τα VRU tracks της sequence
        agents = {}
        for agent_dir in sorted(vru_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            track_file = agent_dir / "track.json"
            if not track_file.exists():
                continue
            track = load_track_json(track_file)
            if track is None:
                continue
            pos, vel = extract_positions_velocities(track)
            if len(pos) < total_len:
                continue
            if np.all(vel == 0) and len(pos) > 1:
                vel = compute_velocities_from_positions(pos)
            agents[agent_dir.name] = {"pos": pos, "vel": vel}

        if len(agents) < 2:
            continue  # χρειαζόμαστε τουλάχιστον 2 agents για relative features

        # Για κάθε agent A, υπολόγισε relative features από όλους τους B ≠ A
        agent_ids = list(agents.keys())

        for a_id in agent_ids:
            pos_A = agents[a_id]["pos"]
            vel_A = agents[a_id]["vel"]
            n_frames = len(pos_A)

            for start in range(0, n_frames - total_len + 1, obs_len // 2):
                pA = pos_A[start : start + total_len]
                vA = vel_A[start : start + total_len]

                # Anchor frame: τελευταίο observed frame του A
                anchor_pos = pA[obs_len - 1].copy()
                anchor_vel = vA[obs_len - 1].copy()
                theta_A    = heading_angle(anchor_vel[0], anchor_vel[1])
                R_A        = rotation_matrix_2d(theta_A)

                # Τοπικές συντεταγμένες του A (αυτόματα: r_A|A = 0 στον anchor)
                pA_local = (R_A @ (pA - anchor_pos).T).T
                vA_local = (R_A @ vA.T).T

                # Relative features από κάθε neighbor B
                neighbors = []
                for b_id in agent_ids:
                    if b_id == a_id:
                        continue
                    pos_B = agents[b_id]["pos"]
                    vel_B = agents[b_id]["vel"]
                    if len(pos_B) < start + total_len:
                        continue

                    pB = pos_B[start : start + total_len]
                    vB = vel_B[start : start + total_len]

                    # r_B|A = R_A · (r_B - r_A)  ανά frame
                    pB_rel = (R_A @ (pB - anchor_pos).T).T
                    vB_rel = (R_A @ (vB - anchor_vel).T).T

                    # Απόσταση στο anchor frame
                    dist_at_obs = float(np.linalg.norm(pB[obs_len-1] - pA[obs_len-1]))

                    neighbors.append({
                        "agent_id":    b_id,
                        "dist":        dist_at_obs,
                        "obs_r_rel":   pB_rel[:obs_len].tolist(),  # [8, 2]
                        "obs_v_rel":   vB_rel[:obs_len].tolist(),  # [8, 2]
                    })

                # Ταξινόμηση neighbors κατά απόσταση (πιο κοντά πρώτα)
                neighbors.sort(key=lambda x: x["dist"])

                all_samples.append({
                    "seq_id":     seq_dir.name,
                    "agent_id":   a_id,
                    "theta_A":    float(theta_A),
                    "obs_pos":    pA_local[:obs_len].tolist(),
                    "pred_pos":   pA_local[obs_len:].tolist(),
                    "obs_vel":    vA_local[:obs_len].tolist(),
                    "anchor_pos": anchor_pos.tolist(),
                    "neighbors":  neighbors[:5],  # κρατάμε τους 5 πιο κοντά
                })

    out_file = output_dir / "sequences_processed.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    print(f"✓ {len(all_samples)} multi-agent samples → {out_file}")


# ══════════════════════════════════════════════════════════════
#  ΒΟΗΘΗΤΙΚΟ: Inverse transform (για visualization)
# ══════════════════════════════════════════════════════════════

def inverse_transform(pos_local: np.ndarray,
                      anchor_pos: np.ndarray,
                      theta_A: float) -> np.ndarray:
    """
    Αντίστροφος μετασχηματισμός: local → global coordinates.
    Χρήσιμο για να ζωγραφίσεις τις προβλέψεις στον αρχικό χάρτη.

    pos_global = R_A^T · pos_local + r_A
    """
    R_A = rotation_matrix_2d(theta_A)
    R_A_inv = R_A.T  # Για rotation matrix: R^-1 = R^T
    return (R_A_inv @ pos_local.T).T + anchor_pos


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IMPTC Preprocessing — Relative Coordinate Transform"
    )
    parser.add_argument(
        "--mode",
        choices=["trajectory", "sequence"],
        default="trajectory",
        help="'trajectory' για μικρό dataset, 'sequence' για μεγάλα sets"
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("./imptc_trajectory_dataset"),
        help="Φάκελος με τα δεδομένα"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./preprocessed"),
        help="Φάκελος εξόδου"
    )
    parser.add_argument(
        "--obs_len", type=int, default=8,
        help="Observed frames (default: 8 = 3.2 sec @ 25Hz)"
    )
    parser.add_argument(
        "--pred_len", type=int, default=12,
        help="Prediction frames (default: 12 = 4.8 sec @ 25Hz)"
    )
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  IMPTC Preprocessing — Mode: {args.mode.upper()}")
    print(f"  obs_len={args.obs_len}  pred_len={args.pred_len}")
    print(f"{'='*55}")

    if args.mode == "trajectory":
        preprocess_trajectory_dataset(
            args.data_dir, args.output_dir,
            args.obs_len, args.pred_len
        )
    else:
        preprocess_sequence_dataset(
            args.data_dir, args.output_dir,
            args.obs_len, args.pred_len
        )


# ══════════════════════════════════════════════════════════════
#  QUICK DEMO (τρέξε χωρίς arguments για να δεις πώς δουλεύει)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=" * 55)
        print("  DEMO: Rotation Matrix Transform")
        print("=" * 55)

        # Agent A: κινείται προς τα βορειοανατολικά
        r_A = np.array([10.0, 5.0])
        V_A = np.array([1.5, 1.5])   # ταχύτητα A: θ = 45°

        # Agent B: λίγο μπροστά και αριστερά
        r_B = np.array([12.0, 8.0])
        V_B = np.array([1.0, 0.5])

        theta_A = heading_angle(V_A[0], V_A[1])
        R_A     = rotation_matrix_2d(theta_A)

        print(f"\nAgent A:  r_A = {r_A},  V_A = {V_A}")
        print(f"          θ_A = {math.degrees(theta_A):.1f}°")
        print(f"\nAgent B:  r_B = {r_B},  V_B = {V_B}")

        print(f"\nRotation Matrix R_A:")
        print(f"  [ {R_A[0,0]:.4f}  {R_A[0,1]:.4f} ]")
        print(f"  [ {R_A[1,0]:.4f}  {R_A[1,1]:.4f} ]")

        r_rel = relative_position(r_A, r_B, R_A)
        v_rel = relative_velocity(V_A, V_B, R_A)

        print(f"\nr_B|A = R_A · (r_B - r_A) = {r_rel}")
        print(f"  → B είναι {r_rel[0]:.2f}m μπροστά, {r_rel[1]:.2f}m αριστερά του A")
        print(f"\nV_B|A = R_A · (V_B - V_A) = {v_rel}")

        # Αντίστροφος μετασχηματισμός
        r_B_recovered = inverse_transform(r_rel.reshape(1, 2), r_A, theta_A).flatten()
        print(f"\nInverse transform (έλεγχος): r_B = {r_B_recovered} (πρέπει = {r_B})")
        print("\n→ Τρέξε: python preprocess_imptc.py --help  για πλήρη χρήση")

    else:
        main()

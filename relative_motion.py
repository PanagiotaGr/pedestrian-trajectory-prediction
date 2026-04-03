"""
relative_motion_v2.py
=====================
Για κάθε track A, σε ΚΑΘΕ χρονική στιγμή της τροχιάς του:
  → Βρες όλους τους B που υπάρχουν την ίδια στιγμή
  → Υπολογισμός p_B|A = R_θA · (p_B - p_A)
  → Υπολογισμός V_rel  = R_θA · (v_B - v_A)

Output ανά track A:
{
  "id_A": "train/0000",
  "class_A": "pedestrian",
  "timesteps": [
      {
        "ts": 167947...,
        "p_A": [x, y],
        "neighbors": [
            {
              "id_B": "train/0001",
              "class_B": "pedestrian",
              "p_B|A": [px_rel, py_rel],
              "V_rel":  [vx_rel, vy_rel],
            },
            ...
        ]
      },
      ...
  ]
}

Χρήση:
    python relative_motion_v2.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
EXTRACT_DIR = Path("imptc_extracted_full")
OUTPUT_DIR  = Path("relative_motion_output")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_HZ = 10
STEP_MS   = 100        # downsample → 10 Hz
STEP_US   = STEP_MS * 1000  # microseconds

CLASS_NAMES = {
    0: "pedestrian", 2: "cyclist", 3: "motorcycle",
    4: "scooter", 5: "stroller", 6: "wheelchair", 10: "unknown"
}


# ─── 1. Φόρτωση ─────────────────────────────────────────────────────────────
def load_all_splits(extract_dir: Path) -> dict:
    candidates = list(extract_dir.rglob("train_tracks.json"))
    if not candidates:
        print("[!] Δεν βρέθηκε train_tracks.json!")
        return {}

    dataset_root = candidates[0].parent
    print(f"[✓] Dataset root: {dataset_root}")

    all_tracks = {}
    for split in ["train", "eval", "test"]:
        json_path = dataset_root / f"{split}_tracks.json"
        if not json_path.exists():
            continue
        print(f"  Φόρτωση {split}...", end=" ", flush=True)
        with open(json_path, "r") as f:
            raw = json.load(f)

        for track_id, data in raw.items():
            overview   = data.get("overview", {})
            track_data = data.get("track_data", {})
            class_id   = overview.get("class_id", 10)

            def sort_key(k):
                parts = k.split("_")
                try:
                    return int(parts[-1])
                except ValueError:
                    return int(k)

            time_series = []
            for ts_key in sorted(track_data.keys(), key=sort_key):
                entry  = track_data[ts_key]
                coords = entry.get("coordinates", [None, None, None])
                time_series.append((
                    entry.get("ts"),
                    coords[0], coords[1],
                    entry.get("velocity"),
                ))

            full_id = f"{split}/{track_id}"
            all_tracks[full_id] = {
                "class_name": CLASS_NAMES.get(class_id, "unknown"),
                "split":      split,
                "track":      time_series,
            }

        count = len([v for v in all_tracks.values() if v["split"] == split])
        print(f"{count} ✓")

    return all_tracks


# ─── 2. Downsample 25Hz → 10Hz ───────────────────────────────────────────────
def downsample_track(track: list) -> list:
    if not track:
        return []
    # Μετατροπή ts σε int αν είναι string
    def to_int_ts(frame):
        ts, x, y, vel = frame[0], frame[1], frame[2], frame[3]
        return (int(ts) if ts is not None else None, x, y, vel)

    track = [to_int_ts(f) for f in track]
    result  = [track[0]]
    last_ts = track[0][0]
    for frame in track[1:]:
        ts = frame[0]
        if ts is None:
            continue
        if (ts - last_ts) >= STEP_US:
            result.append(frame)
            last_ts = ts
    return result


def downsample_all(all_tracks: dict) -> dict:
    print(f"\n[→] Downsampling 25Hz → 10Hz ...")
    downsampled = {}
    total_before, total_after = 0, 0

    for tid, info in all_tracks.items():
        ds = downsample_track(info["track"])
        total_before += len(info["track"])
        total_after  += len(ds)
        downsampled[tid] = {
            **info,
            "track": ds,
        }

    print(f"  Frames πριν : {total_before:,}")
    print(f"  Frames μετά : {total_after:,}  "
          f"({total_after/total_before*100:.1f}% → ~10Hz ✓)")
    return downsampled


# ─── 3. Timestep index: ts → {track_id: frame} ───────────────────────────────
def build_global_ts_index(all_tracks_ds: dict, split: str) -> dict:
    """
    ts → { track_id: (ts, x, y, vel) }
    Μόνο για tracks του συγκεκριμένου split.
    """
    print(f"  Φτιάχνω global timestep index για {split}...", end=" ", flush=True)
    ts_index = defaultdict(dict)

    for tid, info in all_tracks_ds.items():
        if info["split"] != split:
            continue
        for frame in info["track"]:
            ts = frame[0]
            if ts is not None:
                ts_index[ts][tid] = frame

    print(f"✓  ({len(ts_index):,} μοναδικά timestamps)")
    return dict(ts_index)


# ─── 4. Heading & Rotation Matrix ────────────────────────────────────────────
def compute_heading(track: list, idx: int) -> float:
    """θ_A από διαδοχικές θέσεις (backward difference, forward αν idx=0)"""
    if len(track) < 2:
        return 0.0
    if idx == 0:
        p0 = np.array([track[0][1], track[0][2]], dtype=float)
        p1 = np.array([track[1][1], track[1][2]], dtype=float)
    else:
        p0 = np.array([track[idx-1][1], track[idx-1][2]], dtype=float)
        p1 = np.array([track[idx][1],   track[idx][2]],   dtype=float)

    diff = p1 - p0
    if np.linalg.norm(diff) < 1e-6:
        return 0.0
    return np.arctan2(diff[1], diff[0])


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ─── 5. Κύρια συνάρτηση ──────────────────────────────────────────────────────
def compute_all_relative_motion(all_tracks_ds: dict, split: str) -> list:
    """
    Για κάθε track A, σε κάθε timestamp της τροχιάς του,
    βρίσκει όλους τους B και υπολογίζει relative motion.
    """
    print(f"\n[→] Relative motion — {split} split")

    # Global index: ts → {tid: frame}
    ts_index = build_global_ts_index(all_tracks_ds, split)

    split_tracks = {
        tid: info for tid, info in all_tracks_ds.items()
        if info["split"] == split
    }

    all_results = []

    for id_a, info_a in split_tracks.items():
        track_a = info_a["track"]
        if len(track_a) < 2:
            continue

        # Φτιάξε ts → idx για το track A (για heading computation)
        ts_to_idx_a = {frame[0]: i for i, frame in enumerate(track_a)}

        timesteps = []

        for i, frame_a in enumerate(track_a):
            ts = frame_a[0]
            if ts is None:
                continue

            x_a  = frame_a[1]
            y_a  = frame_a[2]
            vel_a = (frame_a[3] or 0.0) / 3.6  # km/h → m/s

            if x_a is None or y_a is None:
                continue

            p_A     = np.array([x_a, y_a], dtype=float)
            theta_A = compute_heading(track_a, i)
            R_A     = rotation_matrix(theta_A)
            v_A_vec = vel_a * np.array([np.cos(theta_A), np.sin(theta_A)])

            # Βρες όλους τους B που υπάρχουν στο ίδιο ts
            neighbors_at_ts = ts_index.get(ts, {})
            neighbors = []

            for id_b, frame_b in neighbors_at_ts.items():
                if id_b == id_a:
                    continue  # skip ο ίδιος

                x_b   = frame_b[1]
                y_b   = frame_b[2]
                vel_b = (frame_b[3] or 0.0) / 3.6

                if x_b is None or y_b is None:
                    continue

                p_B = np.array([x_b, y_b], dtype=float)

                # Heading B — χρειαζόμαστε το track του B
                track_b   = all_tracks_ds[id_b]["track"]
                idx_b     = next(
                    (j for j, f in enumerate(track_b) if f[0] == ts), 0
                )
                theta_B   = compute_heading(track_b, idx_b)
                v_B_vec   = vel_b * np.array([np.cos(theta_B), np.sin(theta_B)])

                # p_B|A = R_θA · (p_B - p_A)
                p_rel = R_A @ (p_B - p_A)

                # V_rel = R_θA · (v_B - v_A)
                v_rel = R_A @ (v_B_vec - v_A_vec)

                # Απόσταση (χρήσιμο για φιλτράρισμα αργότερα)
                dist = float(np.linalg.norm(p_B - p_A))

                neighbors.append({
                    "id_B":    id_b,
                    "class_B": all_tracks_ds[id_b]["class_name"],
                    "p_B|A":   [round(p_rel[0], 4), round(p_rel[1], 4)],
                    "V_rel":   [round(v_rel[0], 4), round(v_rel[1], 4)],
                    "dist_m":  round(dist, 4),
                })

            timesteps.append({
                "ts":        ts,
                "p_A":       [round(x_a, 4), round(y_a, 4)],
                "theta_A":   round(float(theta_A), 4),
                "n_neighbors": len(neighbors),
                "neighbors": neighbors,
            })

        if timesteps:
            all_results.append({
                "id_A":       id_a,
                "class_A":    info_a["class_name"],
                "n_frames":   len(timesteps),
                "timesteps":  timesteps,
            })

    return all_results


# ─── 6. Στατιστικά & Αποθήκευση ─────────────────────────────────────────────
def print_and_save(results: list, split: str):
    total_ts        = sum(len(r["timesteps"]) for r in results)
    total_neighbors = sum(
        ts["n_neighbors"]
        for r in results
        for ts in r["timesteps"]
    )
    avg_neighbors = total_neighbors / total_ts if total_ts > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  Split: {split}")
    print(f"  Tracks A:              {len(results):,}")
    print(f"  Συνολικά timesteps:    {total_ts:,}")
    print(f"  Μέσοι γείτονες/frame: {avg_neighbors:.2f}")

    # Παράδειγμα
    ex = results[0]
    ts0 = ex["timesteps"][0]
    print(f"\n  Παράδειγμα: {ex['id_A']} ({ex['class_A']})")
    print(f"  Frame ts={ts0['ts']}  θ_A={ts0['theta_A']:.3f} rad")
    print(f"  Γείτονες: {ts0['n_neighbors']}")
    for nb in ts0["neighbors"][:3]:
        print(f"    {nb['id_B']} ({nb['class_B']})")
        print(f"      p_B|A = {nb['p_B|A']} m")
        print(f"      V_rel = {nb['V_rel']} m/s")
        print(f"      dist  = {nb['dist_m']} m")

    # Αποθήκευση
    out_path = OUTPUT_DIR / f"relative_motion_{split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\n  [✓] Αποθηκεύτηκε: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[→] Φόρτωση dataset...")
    all_tracks = load_all_splits(EXTRACT_DIR)
    if not all_tracks:
        exit(1)

    all_tracks_ds = downsample_all(all_tracks)

    for split in ["train", "eval", "test"]:
        results = compute_all_relative_motion(all_tracks_ds, split)
        print_and_save(results, split)

    print("\n[✓] Ολοκληρώθηκε!")
    print("    Output: relative_motion_output/relative_motion_{train,eval,test}.json")
    print("    Επόμενο βήμα: interaction matrix (SxS)")

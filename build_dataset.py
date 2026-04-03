"""
build_dataset.py
================
Συνδυάζει:
  - interactions_detailed.csv  (ποια ζεύγη αλληλεπιδρούν)
  - imptc_trajectory_dataset   (τροχιές @ 10Hz)

Matching μέσω first_ts + last_ts.

Output: dataset.csv με relative geometry για κάθε ζεύγος (A,B)
σε κάθε κοινό timestamp @ 10Hz.

Χρήση:
    python build_dataset.py
"""

import json
import csv
import math
from pathlib import Path
from collections import defaultdict

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
EXTRACT_DIR      = Path("imptc_extracted_full")
INTERACTIONS_CSV = Path("results/interactions_detailed.csv")
OUTPUT_CSV       = Path("results/dataset_relative_geometry.csv")
OUTPUT_CSV.parent.mkdir(exist_ok=True)

TARGET_HZ = 10
STEP_US   = 100_000   # 100ms = 10Hz

EPS = 1e-9

CLASS_NAMES = {
    0: "pedestrian", 2: "cyclist", 3: "motorcycle",
    4: "scooter", 5: "stroller", 6: "wheelchair", 10: "unknown"
}


# ─── 1. Φόρτωση trajectory dataset ──────────────────────────────────────────
def load_all_splits(extract_dir: Path) -> dict:
    """
    Επιστρέφει dict:
      (first_ts, last_ts) → {track_id, split, class_name, track: [(ts,x,y,vel),...]}
    Και επίσης dict:
      track_id_str → same info  (για backup lookup)
    """
    candidates = list(extract_dir.rglob("train_tracks.json"))
    if not candidates:
        print("[!] Δεν βρέθηκε train_tracks.json!")
        return {}, {}

    dataset_root = candidates[0].parent
    print(f"[✓] Dataset root: {dataset_root}")

    # ts_key_index: (first_ts, last_ts) → track info
    ts_key_index = {}
    # id_index: "train/0000" → track info
    id_index = {}

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
            first_ts   = int(overview.get("first_ts", 0))
            last_ts    = int(overview.get("last_ts", 0))

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
                ts_val = entry.get("ts")
                time_series.append((
                    int(ts_val) if ts_val is not None else None,
                    coords[0], coords[1],
                    entry.get("velocity"),
                ))

            info = {
                "track_id":   f"{split}/{track_id}",
                "split":      split,
                "class_name": CLASS_NAMES.get(class_id, "unknown"),
                "first_ts":   first_ts,
                "last_ts":    last_ts,
                "track":      time_series,
            }

            ts_key_index[(first_ts, last_ts)] = info
            id_index[f"{split}/{track_id}"] = info

        count = sum(1 for v in id_index.values() if v["split"] == split)
        print(f"{count} ✓")

    return ts_key_index, id_index


# ─── 2. Downsample 25Hz → 10Hz ───────────────────────────────────────────────
def downsample(track: list) -> list:
    if not track:
        return []
    result  = [track[0]]
    last_ts = track[0][0] or 0
    for frame in track[1:]:
        ts = frame[0]
        if ts is None:
            continue
        if (ts - last_ts) >= STEP_US:
            result.append(frame)
            last_ts = ts
    return result


# ─── 3. Velocity από θέσεις ──────────────────────────────────────────────────
def estimate_velocity(track: list, idx: int):
    n = len(track)
    if n < 2:
        return 0.0, 0.0
    if 0 < idx < n - 1:
        p0, p1 = track[idx-1], track[idx+1]
    elif idx == 0:
        p0, p1 = track[0], track[1]
    else:
        p0, p1 = track[-2], track[-1]

    dt = (p1[0] - p0[0]) / 1_000_000.0  # μs → sec
    if abs(dt) < EPS:
        return 0.0, 0.0
    vx = (p1[1] - p0[1]) / dt
    vy = (p1[2] - p0[2]) / dt
    return vx, vy


# ─── 4. Rotation matrix (global → local του A) ───────────────────────────────
def rotation_global_to_local(heading):
    c, s = math.cos(heading), math.sin(heading)
    return ((c, s), (-s, c))   # R_θA^T  (global→local)


def apply_rot(R, x, y):
    return R[0][0]*x + R[0][1]*y, R[1][0]*x + R[1][1]*y


def wrap_angle(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


# ─── 5. Κύρια λογική ─────────────────────────────────────────────────────────
def match_and_compute(ts_key_index: dict, interactions_csv: Path) -> list:
    print(f"\n[→] Φόρτωση interactions από: {interactions_csv}")

    with open(interactions_csv, "r", encoding="utf-8") as f:
        interactions = list(csv.DictReader(f))

    # Κρατάμε μόνο VRU-VRU interactions
    vru_pairs = [r for r in interactions if r["other_type"] == "vrus"]
    print(f"  Σύνολο interactions: {len(interactions):,}")
    print(f"  VRU-VRU pairs:       {len(vru_pairs):,}")

    # Φτιάξε index: (first_ts, last_ts) → track
    # Επίσης φτιάξε ts → frame για γρήγορο lookup
    print("\n[→] Downsample & index tracks...")
    track_ts_map = {}   # track_id → {ts: frame}
    track_ds_map = {}   # track_id → downsampled track list

    def get_or_load(first_ts, last_ts):
        key = (first_ts, last_ts)
        if key not in ts_key_index:
            return None
        info = ts_key_index[key]
        tid  = info["track_id"]
        if tid not in track_ds_map:
            ds = downsample(info["track"])
            track_ds_map[tid] = ds
            track_ts_map[tid] = {f[0]: (i, f) for i, f in enumerate(ds)}
        return info

    rows_out = []
    skipped  = 0

    print(f"[→] Υπολογισμός relative geometry για {len(vru_pairs):,} ζεύγη...")

    for i, row in enumerate(vru_pairs):
        if i % 1000 == 0:
            print(f"  {i:,}/{len(vru_pairs):,}...", end="\r")

        # Lookup track A
        a_first = int(row["target_first_ts"])
        a_last  = int(row["target_last_ts"])
        info_a  = get_or_load(a_first, a_last)

        # Lookup track B
        b_first = int(row["other_first_ts"])
        b_last  = int(row["other_last_ts"])
        info_b  = get_or_load(b_first, b_last)

        if info_a is None or info_b is None:
            skipped += 1
            continue

        tid_a = info_a["track_id"]
        tid_b = info_b["track_id"]

        ts_map_a = track_ts_map[tid_a]
        ts_map_b = track_ts_map[tid_b]
        track_a  = track_ds_map[tid_a]
        track_b  = track_ds_map[tid_b]

        # Κοινά timestamps @ 10Hz
        common_ts = sorted(set(ts_map_a.keys()) & set(ts_map_b.keys()))
        if not common_ts:
            skipped += 1
            continue

        # Relative geometry ανά timestamp
        for ts in common_ts:
            idx_a, frame_a = ts_map_a[ts]
            idx_b, frame_b = ts_map_b[ts]

            x_a, y_a = frame_a[1], frame_a[2]
            x_b, y_b = frame_b[1], frame_b[2]

            if None in (x_a, y_a, x_b, y_b):
                continue

            avx, avy = estimate_velocity(track_a, idx_a)
            bvx, bvy = estimate_velocity(track_b, idx_b)

            speed_a = math.sqrt(avx**2 + avy**2)
            speed_b = math.sqrt(bvx**2 + bvy**2)

            # Heading A
            heading_a = math.atan2(avy, avx) if speed_a > EPS else 0.0
            heading_b = math.atan2(bvy, bvx) if speed_b > EPS else 0.0

            # Rotation matrix global → local(A)
            R = rotation_global_to_local(heading_a)

            dx, dy   = x_b - x_a, y_b - y_a
            dvx, dvy = bvx - avx, bvy - avy

            # p_B|A = R · (p_B - p_A)
            rel_x, rel_y   = apply_rot(R, dx, dy)
            # V_rel = R · (v_B - v_A)
            rel_vx, rel_vy = apply_rot(R, dvx, dvy)

            dist_xy      = math.sqrt(dx**2 + dy**2)
            theta_global = math.atan2(dy, dx)
            theta_local  = math.atan2(rel_y, rel_x)
            heading_diff = wrap_angle(heading_b - heading_a)

            closing_speed = 0.0
            if dist_xy > EPS:
                ux = dx / dist_xy
                uy = dy / dist_xy
                closing_speed = -(dvx*ux + dvy*uy)

            rows_out.append({
                "sample_id":      row["sample_id"],
                "split":          info_a["split"],
                "track_id_A":     tid_a,
                "track_id_B":     tid_b,
                "class_A":        info_a["class_name"],
                "class_B":        info_b["class_name"],
                "ts":             ts,
                # Θέσεις global
                "ax": round(x_a, 4), "ay": round(y_a, 4),
                "bx": round(x_b, 4), "by": round(y_b, 4),
                # Ταχύτητες global
                "avx": round(avx, 4), "avy": round(avy, 4),
                "bvx": round(bvx, 4), "bvy": round(bvy, 4),
                "speed_a": round(speed_a, 4),
                "speed_b": round(speed_b, 4),
                # Relative geometry (local frame of A)
                "rel_x":  round(rel_x, 4),
                "rel_y":  round(rel_y, 4),
                "rel_vx": round(rel_vx, 4),
                "rel_vy": round(rel_vy, 4),
                # Άλλα
                "dist_xy":      round(dist_xy, 4),
                "theta_global": round(theta_global, 4),
                "theta_local":  round(theta_local, 4),
                "heading_a":    round(heading_a, 4),
                "heading_b":    round(heading_b, 4),
                "heading_diff": round(heading_diff, 4),
                "closing_speed": round(closing_speed, 4),
            })

    print(f"\n  Ζεύγη χωρίς match: {skipped}")
    return rows_out


# ─── 6. Αποθήκευση & στατιστικά ─────────────────────────────────────────────
def save_and_summarize(rows: list, out_path: Path):
    if not rows:
        print("[!] Κανένα row για αποθήκευση!")
        return

    fieldnames = [
        "sample_id", "split", "track_id_A", "track_id_B",
        "class_A", "class_B", "ts",
        "ax", "ay", "bx", "by",
        "avx", "avy", "bvx", "bvy",
        "speed_a", "speed_b",
        "rel_x", "rel_y", "rel_vx", "rel_vy",
        "dist_xy", "theta_global", "theta_local",
        "heading_a", "heading_b", "heading_diff", "closing_speed",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*55}")
    print(f"  Αποθηκεύτηκε: {out_path}")
    print(f"  Συνολικές γραμμές (frames): {len(rows):,}")

    from collections import Counter
    splits = Counter(r["split"] for r in rows)
    print(f"\n  Ανά split:")
    for sp in ["train", "eval", "test"]:
        print(f"    {sp:<8}: {splits.get(sp, 0):>8,} frames")

    classes = Counter(f"{r['class_A']}↔{r['class_B']}" for r in rows)
    print(f"\n  Top ζεύγη κλάσεων:")
    for cls, cnt in classes.most_common(5):
        print(f"    {cls:<30}: {cnt:>8,}")

    # Παράδειγμα
    ex = rows[0]
    print(f"\n  Παράδειγμα:")
    print(f"    {ex['track_id_A']} ({ex['class_A']}) ↔ {ex['track_id_B']} ({ex['class_B']})")
    print(f"    ts={ex['ts']}")
    print(f"    p_B|A = ({ex['rel_x']}, {ex['rel_y']}) m")
    print(f"    V_rel = ({ex['rel_vx']}, {ex['rel_vy']}) m/s")
    print(f"    dist  = {ex['dist_xy']} m")
    print(f"{'='*55}")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Φόρτωση trajectory dataset
    print("[→] Φόρτωση trajectory dataset...")
    ts_key_index, id_index = load_all_splits(EXTRACT_DIR)

    if not ts_key_index:
        print("[!] Τρέξε πρώτα: python load_full_dataset.py")
        exit(1)

    print(f"  Σύνολο tracks: {len(ts_key_index):,}")

    # 2. Match + relative geometry
    rows = match_and_compute(ts_key_index, INTERACTIONS_CSV)

    # 3. Αποθήκευση
    save_and_summarize(rows, OUTPUT_CSV)

    print("\n[✓] Ολοκληρώθηκε!")
    print(f"    Output: {OUTPUT_CSV}")

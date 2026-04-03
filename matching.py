"""
matching.py
===========
Για κάθε VRU track βρίσκει ποια άλλα tracks συνυπάρχουν χρονικά
(overlapping timestamps) — αυτά είναι τα candidate ζεύγη (A, B)
για υπολογισμό relative position/velocity.

Χρήση:
    python matching.py

Απαιτήσεις:
    pip install requests tqdm numpy
"""

import json
import tarfile
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
DOWNLOAD_DIR = Path("imptc_data")
EXTRACT_DIR  = Path("imptc_extracted_full")

CLASS_NAMES = {
    0: "pedestrian", 2: "cyclist", 3: "motorcycle",
    4: "scooter", 5: "stroller", 6: "wheelchair", 10: "unknown"
}


# ─── Φόρτωση dataset (ίδια λογική με load_full_dataset.py) ──────────────────
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
                    coords[0], coords[1], coords[2],
                    entry.get("velocity"),
                ))

            full_id = f"{split}/{track_id}"
            all_tracks[full_id] = {
                "class_name": CLASS_NAMES.get(class_id, "unknown"),
                "class_id":   class_id,
                "split":      split,
                "track_id":   track_id,
                "track":      time_series,
                # Αποθήκευσε το set των timestamps για γρήγορο lookup
                "ts_set":     set(row[0] for row in time_series),
                "ts_min":     time_series[0][0]  if time_series else None,
                "ts_max":     time_series[-1][0] if time_series else None,
            }
        print(f"{len([v for v in all_tracks.values() if v['split']==split])} ✓")

    return all_tracks


# ─── Temporal Matching ───────────────────────────────────────────────────────
def temporal_matching(all_tracks: dict, split: str = "train") -> list:
    """
    Βρίσκει όλα τα ζεύγη (A, B) που έχουν overlapping timestamps.
    
    Επιστρέφει λίστα από dicts:
    {
        "id_A": str,
        "id_B": str,
        "class_A": str,
        "class_B": str,
        "common_ts": list,      # κοινά timestamps
        "overlap_frames": int,  # πόσα κοινά frames
        "overlap_sec": float,   # διάρκεια overlap σε δευτερόλεπτα
    }
    """
    # Φιλτράρισμα ανά split
    split_tracks = {k: v for k, v in all_tracks.items() if v["split"] == split}
    track_ids    = list(split_tracks.keys())
    n            = len(track_ids)

    print(f"\n[→] Temporal matching για {n} tracks ({split} split)...")
    print(f"    Συνολικά πιθανά ζεύγη: {n*(n-1)//2:,}")

    # Βήμα 1: Ομαδοποίηση tracks ανά timestamp (timestep index)
    # ts → [track_ids που υπάρχουν σε αυτό το ts]
    print("    Φτιάχνω timestep index...", end=" ", flush=True)
    ts_to_tracks = defaultdict(list)
    for tid, info in split_tracks.items():
        for ts in info["ts_set"]:
            ts_to_tracks[ts].append(tid)
    print("✓")

    # Βήμα 2: Βρες ζεύγη με κοινά timestamps
    print("    Βρίσκω overlapping ζεύγη...", end=" ", flush=True)
    pair_common_ts = defaultdict(list)  # (A,B) → [common timestamps]

    for ts, tids in ts_to_tracks.items():
        if len(tids) < 2:
            continue
        # Όλοι οι VRUs που υπάρχουν σε αυτό το ts → φτιάξε ζεύγη
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                pair_key = (min(a, b), max(a, b))  # canonical order
                pair_common_ts[pair_key].append(ts)

    print(f"✓  ({len(pair_common_ts):,} ζεύγη βρέθηκαν)")

    # Βήμα 3: Φτιάξε αποτέλεσμα
    matches = []
    for (id_a, id_b), common_ts in pair_common_ts.items():
        common_ts_sorted = sorted(common_ts)
        overlap_frames   = len(common_ts_sorted)
        overlap_sec      = overlap_frames / 25.0  # 25 Hz

        matches.append({
            "id_A":           id_a,
            "id_B":           id_b,
            "class_A":        split_tracks[id_a]["class_name"],
            "class_B":        split_tracks[id_b]["class_name"],
            "common_ts":      common_ts_sorted,
            "overlap_frames": overlap_frames,
            "overlap_sec":    overlap_sec,
        })

    # Ταξινόμηση: πρώτα τα ζεύγη με τα περισσότερα κοινά frames
    matches.sort(key=lambda x: x["overlap_frames"], reverse=True)
    return matches


# ─── Στατιστικά matching ─────────────────────────────────────────────────────
def print_matching_summary(matches: list, split: str):
    print(f"\n{'='*60}")
    print(f"  Matching αποτελέσματα — {split} split")
    print(f"{'='*60}")
    print(f"  Συνολικά ζεύγη (A,B): {len(matches):,}")

    if not matches:
        return

    overlaps = [m["overlap_frames"] for m in matches]
    print(f"  Μέσο overlap:  {np.mean(overlaps):.1f} frames  "
          f"({np.mean(overlaps)/25:.1f} sec)")
    print(f"  Max overlap:   {max(overlaps)} frames  ({max(overlaps)/25:.1f} sec)")
    print(f"  Min overlap:   {min(overlaps)} frames")

    # Ανά συνδυασμό κλάσεων
    pair_classes = Counter(
        f"{m['class_A']} ↔ {m['class_B']}" for m in matches
    )
    print("\n  Ζεύγη ανά κλάση:")
    for cls_pair, cnt in pair_classes.most_common(8):
        print(f"    {cls_pair:<35}: {cnt:>5}")

    # Top 5 ζεύγη
    print("\n  Top 5 ζεύγη (μεγαλύτερο overlap):")
    for m in matches[:5]:
        print(f"    {m['id_A']} ↔ {m['id_B']}")
        print(f"      {m['class_A']} ↔ {m['class_B']}  |  "
              f"{m['overlap_frames']} frames  ({m['overlap_sec']:.1f} sec)")


# ─── Αποθήκευση αποτελεσμάτων ────────────────────────────────────────────────
def save_matches(matches: list, split: str):
    out_path = Path(f"matches_{split}.json")
    # Αποθηκεύουμε χωρίς τα common_ts (πολύ μεγάλα) για ελαφρύτερο αρχείο
    lightweight = []
    for m in matches:
        lightweight.append({
            "id_A":           m["id_A"],
            "id_B":           m["id_B"],
            "class_A":        m["class_A"],
            "class_B":        m["class_B"],
            "overlap_frames": m["overlap_frames"],
            "overlap_sec":    m["overlap_sec"],
            "ts_start":       m["common_ts"][0]  if m["common_ts"] else None,
            "ts_end":         m["common_ts"][-1] if m["common_ts"] else None,
        })

    with open(out_path, "w") as f:
        json.dump(lightweight, f, indent=2)
    print(f"\n[✓] Αποτελέσματα αποθηκεύτηκαν: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Φόρτωση dataset
    print("[→] Φόρτωση dataset...")
    all_tracks = load_all_splits(EXTRACT_DIR)

    if not all_tracks:
        print("[!] Κανένα track. Τρέξε πρώτα: python load_full_dataset.py")
        exit(1)

    # Matching για κάθε split
    for split in ["train", "eval", "test"]:
        matches = temporal_matching(all_tracks, split=split)
        print_matching_summary(matches, split)
        save_matches(matches, split)

    print("\n[✓] Matching ολοκληρώθηκε!")
    print("    Επόμενο βήμα: relative position/velocity (rotation matrix)")

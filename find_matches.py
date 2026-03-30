"""
Διαβάζει όλα τα src_info.txt από train/eval/test
και βρίσκει το αντίστοιχο track.json στα μεγάλα sets.

src_info.txt περιέχει: 20230322_081506/000
Μεγάλο set έχει:      0000_20230322_081506/vrus/000/track.json
"""

import os
import json
from pathlib import Path

BASE = Path.home() / "imptc_project"

# ── Βήμα 1: Διάβασε όλα τα src_info.txt ──
print("Διαβάζω src_info.txt από train/eval/test...")
trajectories = []

for split in ["train", "eval", "test"]:
    split_dir = BASE / split
    if not split_dir.exists():
        print(f"  [!] Δεν βρέθηκε: {split_dir}")
        continue

    for traj_dir in sorted(split_dir.iterdir()):
        src_file = traj_dir / "src_info.txt"
        if not src_file.exists():
            continue

        content = src_file.read_text().strip()
        # content = "20230322_081506/000"
        parts = content.split("/")
        if len(parts) != 2:
            print(f"  [!] Αδύνατη μορφή: {content}")
            continue

        seq_datetime = parts[0]   # π.χ. 20230322_081506
        track_id     = parts[1]   # π.χ. 000

        trajectories.append({
            "split":        split,
            "traj_id":      traj_dir.name,
            "traj_path":    str(traj_dir),
            "seq_datetime": seq_datetime,
            "track_id":     track_id,
            "src_info":     content,
        })

print(f"  Βρέθηκαν {len(trajectories)} trajectories συνολικά\n")

# ── Βήμα 2: Βρες τα sequence folders στα μεγάλα sets ──
print("Σαρώνω μεγάλα sequence folders...")

# Τα μεγάλα sets αποσυμπιέζονται ως: 0000_20230322_081506/
# Βρες όλα τα folders που ταιριάζουν στο pattern NNNN_YYYYMMDD_HHMMSS
seq_folders = {}
for item in BASE.iterdir():
    if item.is_dir() and "_" in item.name:
        parts = item.name.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            # parts[1] = "20230322_081506"
            datetime_key = parts[1]
            seq_folders[datetime_key] = item

print(f"  Βρέθηκαν {len(seq_folders)} sequences\n")

# ── Βήμα 3: Κάνε match ──
print("Κάνω matching...")
matches = []
not_found = []

for traj in trajectories:
    seq_dt  = traj["seq_datetime"]   # 20230322_081506
    track_id = traj["track_id"]       # 000

    if seq_dt not in seq_folders:
        not_found.append(traj)
        continue

    seq_folder = seq_folders[seq_dt]
    # track.json βρίσκεται στο: <seq_folder>/vrus/<track_id>/track.json
    track_json = seq_folder / "vrus" / track_id / "track.json"

    traj["seq_folder"]  = str(seq_folder)
    traj["track_json"]  = str(track_json)
    traj["found"]       = track_json.exists()
    matches.append(traj)

# ── Βήμα 4: Αποτελέσματα ──
found_ok  = [m for m in matches if m["found"]]
found_no  = [m for m in matches if not m["found"]]

print(f"✓ Matches που βρέθηκαν:       {len(found_ok)}")
print(f"✗ Matches χωρίς track.json:   {len(found_no)}")
print(f"✗ Sequences που λείπουν:      {len(not_found)}")

# ── Βήμα 5: Αποθήκευση ──
out_file = BASE / "matches.json"
with open(out_file, "w") as f:
    json.dump(matches + not_found, f, indent=2)
print(f"\n→ Αποθηκεύτηκε: {out_file}")

# ── Βήμα 6: Εμφάνισε μερικά παραδείγματα ──
print("\nΠρώτα 5 matches:")
for m in found_ok[:5]:
    print(f"  [{m['split']}] {m['traj_id']} → {m['track_json']}")

if not_found:
    print(f"\nΠρώτα 3 που ΔΕΝ βρέθηκαν:")
    for m in not_found[:3]:
        print(f"  [{m['split']}] {m['traj_id']} → seq: {m['seq_datetime']} (δεν έχει αποσυμπιεστεί ακόμα)")

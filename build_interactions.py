"""
Interactions CSV
1 γραμμή = 1 VRU + οι γείτονές του (ταυτόχρονοι VRUs στο ίδιο scene)
"""
import json, csv, math
import numpy as np
from pathlib import Path
from collections import Counter

BASE    = Path.home() / "imptc_project"
OUT_DIR = BASE / "preprocessed"
STEP    = 10
MAX_NEIGHBORS = 5  # κρατάμε τους 5 πιο κοντινούς

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Βρες όλα τα sequence folders
seq_folders = sorted([d for d in BASE.iterdir()
                      if d.is_dir() and d.name[0].isdigit() and "_" in d.name])
print(f"Sequences: {len(seq_folders)}")

# Φόρτωσε matches για να ξέρουμε traj_id → seq_folder
matches = {}
with open(str(OUT_DIR/"matches_all.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["found"] == "True":
            matches[row["seq_datetime"] + "/" + row["track_id"]] = row["traj_id"]

print(f"Matches: {len(matches)}")

# Headers
headers = [
    "traj_id", "split", "class_name", "seq_folder",
    "n_neighbors",           # πόσοι γείτονες βρέθηκαν
    # Γείτονας 1 (πιο κοντινός)
    *[f"n{i+1}_{x}" for i in range(MAX_NEIGHBORS)
      for x in ["id","class","min_dist","mean_dist","overlap_sec"]],
    # Στατιστικά αλληλεπίδρασης
    "min_dist_any",          # ελάχιστη απόσταση από οποιονδήποτε
    "mean_dist_all",         # μέση απόσταση
    "n_close_encounters",    # φορές που κάποιος ήταν <2m
    "classes_nearby",        # ποιες κλάσεις υπήρχαν γύρω
]

out_csv = OUT_DIR / "interactions.csv"
written = 0

with open(str(out_csv), "w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()

    for seq_folder in seq_folders:
        vrus_dir = seq_folder / "vrus"
        if not vrus_dir.exists():
            continue

        # Φόρτωσε όλα τα VRUs της sequence
        vru_data = {}
        for vru_dir in sorted(vrus_dir.iterdir()):
            track_f = vru_dir / "track.json"
            if not track_f.exists():
                continue
            try:
                track    = json.loads(track_f.read_text())
                frames   = sorted(track["track_data"].keys(), key=lambda x: int(x))
                pos, ts  = [], []
                for k in frames:
                    f = track["track_data"][k]
                    c = f["coordinates"]
                    pos.append([c[0], c[1]])
                    ts.append(int(f["ts"]))
                vru_data[vru_dir.name] = {
                    "class":    track["overview"]["class_name"],
                    "first_ts": int(track["overview"]["first_ts"]),
                    "last_ts":  int(track["overview"]["last_ts"]),
                    "pos":      pos[::STEP],
                    "ts":       ts[::STEP],
                }
            except:
                continue

        if len(vru_data) < 2:
            continue

        # Για κάθε VRU — βρες γείτονες
        for vru_id, target in vru_data.items():
            src_key  = seq_folder.name.split("_",1)[1] + "/" + vru_id
            traj_id  = matches.get(src_key, "")
            split    = ""
            if traj_id:
                # Βρες split
                for sp in ["train","eval"]:
                    if (BASE/sp/traj_id).exists():
                        split = sp
                        break

            neighbors_info = []

            for other_id, other in vru_data.items():
                if other_id == vru_id:
                    continue

                # Χρονική επικάλυψη
                overlap_start = max(target["first_ts"], other["first_ts"])
                overlap_end   = min(target["last_ts"],  other["last_ts"])
                if overlap_start >= overlap_end:
                    continue
                overlap_sec = round((overlap_end - overlap_start)/1e6, 2)

                # Υπολόγισε αποστάσεις στα κοινά frames
                dists = []
                t_ts  = np.array(target["ts"])
                o_ts  = np.array(other["ts"])

                for i, ts_t in enumerate(t_ts):
                    if ts_t < overlap_start or ts_t > overlap_end:
                        continue
                    # Βρες κοντινότερο frame του other
                    idx = np.argmin(np.abs(o_ts - ts_t))
                    if idx < len(other["pos"]) and i < len(target["pos"]):
                        d = dist(target["pos"][i], other["pos"][idx])
                        dists.append(d)

                if not dists:
                    continue

                neighbors_info.append({
                    "id":          other_id,
                    "class":       other["class"],
                    "min_dist":    round(min(dists), 3),
                    "mean_dist":   round(sum(dists)/len(dists), 3),
                    "overlap_sec": overlap_sec,
                })

            # Ταξινόμηση κατά min_dist
            neighbors_info.sort(key=lambda x: x["min_dist"])

            # Στατιστικά
            all_dists = [n["min_dist"] for n in neighbors_info]
            min_dist_any     = round(min(all_dists), 3) if all_dists else -1
            mean_dist_all    = round(sum(all_dists)/len(all_dists), 3) if all_dists else -1
            n_close          = sum(1 for d in all_dists if d < 2.0)
            classes_nearby   = ",".join(sorted(set(n["class"] for n in neighbors_info)))

            row = {
                "traj_id":          traj_id,
                "split":            split,
                "class_name":       target["class"],
                "seq_folder":       seq_folder.name,
                "n_neighbors":      len(neighbors_info),
                "min_dist_any":     min_dist_any,
                "mean_dist_all":    mean_dist_all,
                "n_close_encounters": n_close,
                "classes_nearby":   classes_nearby,
            }

            # Top 5 γείτονες
            for i in range(MAX_NEIGHBORS):
                if i < len(neighbors_info):
                    n = neighbors_info[i]
                    row[f"n{i+1}_id"]          = n["id"]
                    row[f"n{i+1}_class"]        = n["class"]
                    row[f"n{i+1}_min_dist"]     = n["min_dist"]
                    row[f"n{i+1}_mean_dist"]    = n["mean_dist"]
                    row[f"n{i+1}_overlap_sec"]  = n["overlap_sec"]
                else:
                    row[f"n{i+1}_id"]          = ""
                    row[f"n{i+1}_class"]        = ""
                    row[f"n{i+1}_min_dist"]     = ""
                    row[f"n{i+1}_mean_dist"]    = ""
                    row[f"n{i+1}_overlap_sec"]  = ""

            writer.writerow(row)
            written += 1

        if written % 500 == 0 and written > 0:
            print(f"  {written} VRUs επεξεργάστηκαν...")

size_mb = out_csv.stat().st_size/1024/1024
print(f"\n✓ interactions.csv")
print(f"  Γραμμές : {written} VRUs")
print(f"  Μέγεθος : {size_mb:.1f} MB")

# Στατιστικά
print(f"\n=== ΣΤΑΤΙΣΤΙΚΑ ΑΛΛΗΛΕΠΙΔΡΑΣΕΩΝ ===")
close_total = 0
total = 0
with open(str(out_csv), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        if row["min_dist_any"] and float(row["min_dist_any"]) < 2.0:
            close_total += 1

print(f"  VRUs με κοντινή επαφή (<2m): {close_total}/{total} ({100*close_total/total:.1f}%)")

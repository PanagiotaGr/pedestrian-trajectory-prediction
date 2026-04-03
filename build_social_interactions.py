"""
Social Interactions CSV
1 γραμμή = 1 VRU με:
- ποιοι άλλοι agents συνυπάρχουν
- πόσα κοινά timestamps
- πόσα frames < 3m
- ελάχιστη απόσταση
"""
import json, csv, math
import numpy as np
from pathlib import Path

BASE    = Path.home() / "imptc_project"
OUT_DIR = BASE / "preprocessed"
STEP    = 10
MAX_N   = 5
CLOSE_THRESH = 3.0  # μέτρα

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Φόρτωσε matches
matches_map = {}
with open(str(OUT_DIR/"matches_all.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["found"] == "True":
            key = row["seq_datetime"] + "/" + row["track_id"]
            matches_map[key] = {"traj_id": row["traj_id"],
                                "seq_folder": row["seq_folder"]}

seq_folders = sorted([d for d in BASE.iterdir()
                      if d.is_dir() and d.name[0].isdigit()])
print(f"Sequences: {len(seq_folders)}")

headers = [
    "traj_id", "split", "class_name", "seq_folder",
    # Γενικά
    "n_coexisting_agents",    # πόσοι agents συνυπάρχουν
    "coexisting_classes",     # ποιες κλάσεις
    # Στατιστικά αλληλεπίδρασης
    "min_dist_overall",       # ελάχιστη απόσταση από οποιονδήποτε
    "n_frames_below_3m",      # frames που κάποιος ήταν <3m
    "n_close_agents",         # πόσοι agents ήρθαν <3m
    # Top 5 γείτονες
    *[f"n{i+1}_{x}" for i in range(MAX_N)
      for x in ["traj_id","class",
                "common_timestamps",  # κοινά timestamps
                "frames_below_3m",    # frames <3m
                "min_dist",           # ελάχιστη απόσταση
                "mean_dist",          # μέση απόσταση
                "overlap_sec"]],      # διάρκεια συνύπαρξης
]

out_csv = OUT_DIR / "social_interactions.csv"
written = 0

with open(str(out_csv), "w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()

    for seq_folder in seq_folders:
        vrus_dir = seq_folder / "vrus"
        if not vrus_dir.exists(): continue

        # Φόρτωσε ΟΛΑ τα VRUs
        vru_data = {}
        for vru_dir in sorted(vrus_dir.iterdir()):
            track_f = vru_dir / "track.json"
            if not track_f.exists(): continue
            try:
                track  = json.loads(track_f.read_text())
                frames = sorted(track["track_data"].keys(), key=lambda x: int(x))
                pos, ts = [], []
                for k in frames:
                    f = track["track_data"][k]
                    c = f["coordinates"]
                    pos.append([c[0], c[1]])
                    ts.append(int(f["ts"]))
                # Downsample
                pos = pos[::STEP]
                ts  = ts[::STEP]
                # Dict: ts → pos για γρήγορο lookup
                ts_to_pos = {t: p for t, p in zip(ts, pos)}

                seq_dt   = seq_folder.name.split("_",1)[1]
                match_key = seq_dt + "/" + vru_dir.name
                match_info = matches_map.get(match_key, {})

                vru_data[vru_dir.name] = {
                    "class":     track["overview"]["class_name"],
                    "first_ts":  int(track["overview"]["first_ts"]),
                    "last_ts":   int(track["overview"]["last_ts"]),
                    "ts_to_pos": ts_to_pos,
                    "ts_set":    set(ts),
                    "traj_id":   match_info.get("traj_id",""),
                }
            except: continue

        if len(vru_data) < 2: continue

        # Για κάθε VRU
        for vru_id, target in vru_data.items():
            traj_id = target["traj_id"]
            split   = ""
            if traj_id:
                for sp in ["train","eval"]:
                    if (BASE/sp/traj_id).exists():
                        split = sp; break

            neighbors = []

            for other_id, other in vru_data.items():
                if other_id == vru_id: continue

                # Κοινά timestamps
                common_ts = target["ts_set"] & other["ts_set"]
                if not common_ts: continue

                common_timestamps = len(common_ts)
                overlap_sec = round(
                    (min(target["last_ts"], other["last_ts"]) -
                     max(target["first_ts"], other["first_ts"])) / 1e6, 2)

                # Αποστάσεις σε κοινά frames
                dists = []
                for ts_t in sorted(common_ts):
                    p1 = target["ts_to_pos"].get(ts_t)
                    p2 = other["ts_to_pos"].get(ts_t)
                    if p1 and p2:
                        dists.append(dist(p1, p2))

                if not dists: continue

                frames_below_3m = sum(1 for d in dists if d < CLOSE_THRESH)
                min_d  = round(min(dists), 3)
                mean_d = round(sum(dists)/len(dists), 3)

                neighbors.append({
                    "traj_id":          other["traj_id"],
                    "class":            other["class"],
                    "common_timestamps": common_timestamps,
                    "frames_below_3m":  frames_below_3m,
                    "min_dist":         min_d,
                    "mean_dist":        mean_d,
                    "overlap_sec":      overlap_sec,
                })

            # Ταξινόμηση κατά min_dist
            neighbors.sort(key=lambda x: x["min_dist"])

            # Συγκεντρωτικά
            all_min_dists = [n["min_dist"] for n in neighbors]
            min_dist_overall  = round(min(all_min_dists), 3) if all_min_dists else -1
            n_frames_below_3m = sum(n["frames_below_3m"] for n in neighbors)
            n_close_agents    = sum(1 for n in neighbors if n["min_dist"] < CLOSE_THRESH)
            coexisting_classes= ",".join(sorted(set(n["class"] for n in neighbors)))

            row = {
                "traj_id":             traj_id,
                "split":               split,
                "class_name":          target["class"],
                "seq_folder":          seq_folder.name,
                "n_coexisting_agents": len(neighbors),
                "coexisting_classes":  coexisting_classes,
                "min_dist_overall":    min_dist_overall,
                "n_frames_below_3m":   n_frames_below_3m,
                "n_close_agents":      n_close_agents,
            }

            # Top 5 γείτονες
            for i in range(MAX_N):
                if i < len(neighbors):
                    n = neighbors[i]
                    row[f"n{i+1}_traj_id"]          = n["traj_id"]
                    row[f"n{i+1}_class"]             = n["class"]
                    row[f"n{i+1}_common_timestamps"] = n["common_timestamps"]
                    row[f"n{i+1}_frames_below_3m"]   = n["frames_below_3m"]
                    row[f"n{i+1}_min_dist"]          = n["min_dist"]
                    row[f"n{i+1}_mean_dist"]         = n["mean_dist"]
                    row[f"n{i+1}_overlap_sec"]       = n["overlap_sec"]
                else:
                    row[f"n{i+1}_traj_id"]          = ""
                    row[f"n{i+1}_class"]             = ""
                    row[f"n{i+1}_common_timestamps"] = ""
                    row[f"n{i+1}_frames_below_3m"]   = ""
                    row[f"n{i+1}_min_dist"]          = ""
                    row[f"n{i+1}_mean_dist"]         = ""
                    row[f"n{i+1}_overlap_sec"]       = ""

            writer.writerow(row)
            written += 1

        if written % 500 == 0 and written > 0:
            print(f"  {written}...")

size_mb = out_csv.stat().st_size/1024/1024
print(f"\n✓ social_interactions.csv")
print(f"  Γραμμές : {written} VRUs")
print(f"  Μέγεθος : {size_mb:.1f} MB")

# Στατιστικά
print(f"\n=== ΣΤΑΤΙΣΤΙΚΑ ===")
total = close3m = isolated = 0
with open(str(out_csv), newline="") as f:
    for row in csv.DictReader(f):
        total += 1
        if row["min_dist_overall"] and float(row["min_dist_overall"]) < 3.0:
            close3m += 1
        if row["n_coexisting_agents"] == "0":
            isolated += 1

print(f"  Συνολικά VRUs        : {total}")
print(f"  Με γείτονα <3m       : {close3m} ({100*close3m/total:.1f}%)")
print(f"  Χωρίς γείτονες       : {isolated} ({100*isolated/total:.1f}%)")

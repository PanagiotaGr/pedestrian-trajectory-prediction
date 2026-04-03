"""
2 CSV αρχεία:
1. interactions_summary.csv   — 1 γραμμή ανά VRU
2. interactions_detailed.csv  — 1 γραμμή ανά ζεύγος VRU-VRU
"""
import json, csv, math
import numpy as np
from pathlib import Path

BASE    = Path.home() / "imptc_project"
OUT_DIR = BASE / "preprocessed"
STEP    = 10
CLOSE_THRESH = 3.0

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Φόρτωσε matches
matches_map = {}
with open(str(OUT_DIR/"matches_all.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["found"] == "True":
            key = row["seq_datetime"] + "/" + row["track_id"]
            matches_map[key] = row["traj_id"]

seq_folders = sorted([d for d in BASE.iterdir()
                      if d.is_dir() and d.name[0].isdigit()])
print(f"Sequences: {len(seq_folders)}")

# ════════════════════════════════════════
# HEADERS
# ════════════════════════════════════════
summary_headers = [
    "traj_id", "split", "class_name", "seq_folder",
    # Συνολικές αλληλεπιδράσεις
    "total_interactions",
    "interactions_with_vrus",
    "interactions_with_vehicles",
    # VRU αλληλεπιδράσεις
    "n_vru_coexisting",
    "n_vru_close",           # VRUs που ήρθαν <3m
    "min_dist_vru",          # ελάχιστη απόσταση από VRU
    "n_frames_below_3m_vru", # frames <3m με VRUs
    "vru_classes_nearby",    # ποιες κλάσεις VRU υπήρχαν
    # Vehicle αλληλεπιδράσεις
    "n_vehicle_coexisting",
    "n_vehicle_close",
    "min_dist_vehicle",
    "n_frames_below_3m_veh",
    # Ισχυρότερος γείτονας
    "closest_agent_id",
    "closest_agent_type",
    "closest_agent_class",
    "closest_agent_dist",
    "closest_agent_common_ts",
]

detailed_headers = [
    "target_traj_id", "target_split", "target_class", "seq_folder",
    "other_id", "other_traj_id", "other_type", "other_class",
    "common_timestamps",
    "frames_below_3m",
    "min_dist",
    "mean_dist",
    "max_dist",
    "overlap_sec",
    "is_close",              # 1 αν min_dist < 3m
]

sum_csv = OUT_DIR / "interactions_summary.csv"
det_csv = OUT_DIR / "interactions_detailed.csv"

sum_written = 0
det_written = 0

with open(str(sum_csv), "w", newline="") as fsum, \
     open(str(det_csv), "w", newline="") as fdet:

    sum_writer = csv.DictWriter(fsum, fieldnames=summary_headers, extrasaction="ignore")
    det_writer = csv.DictWriter(fdet, fieldnames=detailed_headers, extrasaction="ignore")
    sum_writer.writeheader()
    det_writer.writeheader()

    for seq_folder in seq_folders:
        # Φόρτωσε VRUs
        vrus_dir = seq_folder / "vrus"
        veh_dir  = seq_folder / "vehicles"

        def load_agents(agents_dir, agent_type):
            agents = {}
            if not agents_dir.exists(): return agents
            for agent_dir in sorted(agents_dir.iterdir()):
                track_f = agent_dir / "track.json"
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
                    pos = pos[::STEP]
                    ts  = ts[::STEP]
                    seq_dt    = seq_folder.name.split("_",1)[1]
                    match_key = seq_dt + "/" + agent_dir.name
                    traj_id   = matches_map.get(match_key, "")
                    agents[agent_dir.name] = {
                        "type":      agent_type,
                        "class":     track["overview"].get("class_name","unknown"),
                        "first_ts":  int(track["overview"]["first_ts"]),
                        "last_ts":   int(track["overview"]["last_ts"]),
                        "ts_to_pos": {t:p for t,p in zip(ts,pos)},
                        "ts_set":    set(ts),
                        "traj_id":   traj_id,
                    }
                except: continue
            return agents

        vru_agents = load_agents(vrus_dir, "vru")
        veh_agents = load_agents(veh_dir,  "vehicle")
        all_agents = {**vru_agents, **veh_agents}

        if not vru_agents: continue

        # Για κάθε VRU (target)
        for vru_id, target in vru_agents.items():
            traj_id = target["traj_id"]
            split   = ""
            if traj_id:
                for sp in ["train","eval"]:
                    if (BASE/sp/traj_id).exists():
                        split = sp; break

            vru_interactions = []
            veh_interactions = []

            # Vs όλοι οι άλλοι agents
            for other_id, other in all_agents.items():
                if other_id == vru_id: continue

                common_ts = target["ts_set"] & other["ts_set"]
                if not common_ts: continue

                overlap_sec = round(
                    (min(target["last_ts"], other["last_ts"]) -
                     max(target["first_ts"], other["first_ts"])) / 1e6, 2)

                dists = []
                for ts_t in sorted(common_ts):
                    p1 = target["ts_to_pos"].get(ts_t)
                    p2 = other["ts_to_pos"].get(ts_t)
                    if p1 and p2:
                        dists.append(dist(p1, p2))

                if not dists: continue

                frames_below = sum(1 for d in dists if d < CLOSE_THRESH)
                min_d   = round(min(dists), 3)
                mean_d  = round(sum(dists)/len(dists), 3)
                max_d   = round(max(dists), 3)

                interaction = {
                    "other_id":          other_id,
                    "other_traj_id":     other["traj_id"],
                    "other_type":        other["type"],
                    "other_class":       other["class"],
                    "common_timestamps": len(common_ts),
                    "frames_below_3m":   frames_below,
                    "min_dist":          min_d,
                    "mean_dist":         mean_d,
                    "max_dist":          max_d,
                    "overlap_sec":       overlap_sec,
                    "is_close":          1 if min_d < CLOSE_THRESH else 0,
                }

                if other["type"] == "vru":
                    vru_interactions.append(interaction)
                else:
                    veh_interactions.append(interaction)

                # Γράψε στο detailed CSV
                det_writer.writerow({
                    "target_traj_id":  traj_id,
                    "target_split":    split,
                    "target_class":    target["class"],
                    "seq_folder":      seq_folder.name,
                    "other_id":        other_id,
                    "other_traj_id":   other["traj_id"],
                    "other_type":      other["type"],
                    "other_class":     other["class"],
                    "common_timestamps": len(common_ts),
                    "frames_below_3m": frames_below,
                    "min_dist":        min_d,
                    "mean_dist":       mean_d,
                    "max_dist":        max_d,
                    "overlap_sec":     overlap_sec,
                    "is_close":        1 if min_d < CLOSE_THRESH else 0,
                })
                det_written += 1

            # Summary row
            all_inter = vru_interactions + veh_interactions
            all_inter.sort(key=lambda x: x["min_dist"])

            closest = all_inter[0] if all_inter else None

            vru_dists = [n["min_dist"] for n in vru_interactions]
            veh_dists = [n["min_dist"] for n in veh_interactions]

            sum_writer.writerow({
                "traj_id":      traj_id,
                "split":        split,
                "class_name":   target["class"],
                "seq_folder":   seq_folder.name,
                "total_interactions":        len(all_inter),
                "interactions_with_vrus":    len(vru_interactions),
                "interactions_with_vehicles": len(veh_interactions),
                "n_vru_coexisting":   len(vru_interactions),
                "n_vru_close":        sum(1 for n in vru_interactions if n["min_dist"]<CLOSE_THRESH),
                "min_dist_vru":       round(min(vru_dists),3) if vru_dists else -1,
                "n_frames_below_3m_vru": sum(n["frames_below_3m"] for n in vru_interactions),
                "vru_classes_nearby": ",".join(sorted(set(n["other_class"] for n in vru_interactions))),
                "n_vehicle_coexisting":   len(veh_interactions),
                "n_vehicle_close":        sum(1 for n in veh_interactions if n["min_dist"]<CLOSE_THRESH),
                "min_dist_vehicle":       round(min(veh_dists),3) if veh_dists else -1,
                "n_frames_below_3m_veh":  sum(n["frames_below_3m"] for n in veh_interactions),
                "closest_agent_id":    closest["other_id"]      if closest else "",
                "closest_agent_type":  closest["other_type"]    if closest else "",
                "closest_agent_class": closest["other_class"]   if closest else "",
                "closest_agent_dist":  closest["min_dist"]      if closest else -1,
                "closest_agent_common_ts": closest["common_timestamps"] if closest else 0,
            })
            sum_written += 1

        if sum_written % 200 == 0 and sum_written > 0:
            print(f"  {sum_written} VRUs | {det_written} interactions...")

print(f"\n✓ interactions_summary.csv  — {sum_written} γραμμές")
print(f"✓ interactions_detailed.csv — {det_written} γραμμές")

# Στατιστικά
print(f"\n=== SUMMARY ΣΤΑΤΙΣΤΙΚΑ ===")
from collections import Counter
cnt_close_vru = cnt_close_veh = cnt_isolated = total = 0
with open(str(sum_csv), newline="") as f:
    for row in csv.DictReader(f):
        total += 1
        if row["min_dist_vru"]     and float(row["min_dist_vru"]) < 3.0:     cnt_close_vru += 1
        if row["min_dist_vehicle"] and float(row["min_dist_vehicle"]) < 3.0: cnt_close_veh += 1
        if row["total_interactions"] == "0": cnt_isolated += 1

print(f"  Συνολικά VRUs          : {total}")
print(f"  Με VRU γείτονα <3m    : {cnt_close_vru} ({100*cnt_close_vru/total:.1f}%)")
print(f"  Με vehicle <3m         : {cnt_close_veh} ({100*cnt_close_veh/total:.1f}%)")
print(f"  Χωρίς καμία αλληλεπίδραση: {cnt_isolated} ({100*cnt_isolated/total:.1f}%)")

print(f"\n=== DETAILED ΣΤΑΤΙΣΤΙΚΑ ===")
cnt_type = Counter()
cnt_close = 0
with open(str(det_csv), newline="") as f:
    for row in csv.DictReader(f):
        cnt_type[row["other_type"]] += 1
        if row["is_close"] == "1": cnt_close += 1
print(f"  VRU-VRU pairs     : {cnt_type['vru']}")
print(f"  VRU-Vehicle pairs : {cnt_type['vehicle']}")
print(f"  Close pairs (<3m) : {cnt_close} ({100*cnt_close/det_written:.1f}%)")

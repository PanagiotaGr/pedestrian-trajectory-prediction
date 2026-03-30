import json
import csv
from pathlib import Path
from collections import Counter

BASE    = Path.home() / "imptc_project"
OUT_DIR = BASE / "preprocessed"

OBS_LEN = 8
STEP    = 10

GROUND_NAMES = {
    0: "road", 1: "sidewalk", 2: "ground", 3: "curb",
    4: "road_line", 5: "crosswalk", 6: "bikelane", 7: "unknown"
}

def get_ground_types(track_full_path, step=10):
    with open(track_full_path) as f:
        data = json.load(f)
    frames = sorted(data["track_data"].keys(), key=lambda x: int(x))
    # FIX: αντικατάστησε None με -1
    gts = [data["track_data"][k].get("ground_type") for k in frames]
    gts = [g if g is not None else -1 for g in gts]
    return gts[::step]

for split in ["train", "eval"]:
    in_csv  = OUT_DIR / f"{split}.csv"
    out_csv = OUT_DIR / f"{split}_with_gt.csv"

    print(f"\n[{split.upper()}] Φορτώνω...")

    matches = json.loads((BASE / "matches.json").read_text())
    found = {m["traj_id"]: m for m in matches if m.get("found")}

    # Cache ground types
    gt_cache = {}
    for traj_id, m in found.items():
        track_full = Path(m["traj_path"]) / "track_full.json"
        if track_full.exists():
            try:
                gt_cache[traj_id] = get_ground_types(track_full, STEP)
            except Exception as e:
                print(f"  [!] {traj_id}: {e}")

    print(f"  GT φορτώθηκαν για {len(gt_cache)} trajectories")

    traj_counters = {}
    written = 0
    skipped = 0

    with open(in_csv, newline="") as fin, \
         open(out_csv, "w", newline="") as fout:

        reader = csv.DictReader(fin)
        gt_headers    = [f"obs_gt_{i}" for i in range(OBS_LEN)]
        extra_headers = ["dominant_gt", "dominant_gt_name", "on_crosswalk"]
        new_headers   = reader.fieldnames + gt_headers + extra_headers
        writer = csv.DictWriter(fout, fieldnames=new_headers)
        writer.writeheader()

        for row in reader:
            traj_id = row["traj_id"]

            if traj_id not in traj_counters:
                traj_counters[traj_id] = 0
            start = traj_counters[traj_id]
            traj_counters[traj_id] += 1

            if traj_id in gt_cache:
                gts  = gt_cache[traj_id]
                total = OBS_LEN + 12
                window = gts[start : start + total]
                obs_gts = window[:OBS_LEN] if len(window) >= OBS_LEN else (window + [-1]*OBS_LEN)[:OBS_LEN]

                for i in range(OBS_LEN):
                    row[f"obs_gt_{i}"] = int(obs_gts[i])

                cnt      = Counter(obs_gts)
                dominant = cnt.most_common(1)[0][0]
                row["dominant_gt"]      = int(dominant)
                row["dominant_gt_name"] = GROUND_NAMES.get(dominant, "unknown")
                row["on_crosswalk"]     = 1 if dominant == 5 else 0
            else:
                for i in range(OBS_LEN):
                    row[f"obs_gt_{i}"] = -1
                row["dominant_gt"]      = -1
                row["dominant_gt_name"] = "unknown"
                row["on_crosswalk"]     = 0
                skipped += 1

            writer.writerow(row)
            written += 1
            if written % 50000 == 0:
                print(f"  {written}...")

    size_mb = out_csv.stat().st_size / 1024 / 1024
    print(f"  ✓ {out_csv.name}  ({size_mb:.1f} MB)")
    print(f"  ✓ Written: {written} | Χωρίς GT: {skipped}")

# ── Στατιστικά ──
print("\n=== ΣΤΑΤΙΣΤΙΚΑ GROUND TYPES (train) ===")
cnt = Counter()
with open(OUT_DIR / "train_with_gt.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cnt[row["dominant_gt_name"]] += 1

total = sum(cnt.values())
for name, count in cnt.most_common():
    pct = 100 * count / total
    print(f"  {name:12s}: {count:6d} samples  ({pct:.1f}%)")

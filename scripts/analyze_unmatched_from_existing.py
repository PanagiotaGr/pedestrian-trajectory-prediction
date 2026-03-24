import csv
import os
from collections import defaultdict, Counter

sample_csv = os.path.expanduser("~/imptc_project/results/sample_codes.csv")
archive_csv = os.path.expanduser("~/imptc_project/results/archive_index.csv")
matched_csv = os.path.expanduser("~/imptc_project/results/matched_codes.csv")
out_csv = os.path.expanduser("~/imptc_project/results/unmatched_analysis.csv")

# fortwse ola ta samples
samples = {}
with open(sample_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        samples[row["sample_id"]] = row

# fortwse ta matched sample_ids
matched_ids = set()
with open(matched_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        matched_ids.add(row["sample_id"])

# ftiaxe index apo archive
archive_rows = []
with open(archive_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        archive_rows.append(row)

by_datetime = defaultdict(list)
by_track = defaultdict(list)
by_scene_code = defaultdict(list)

for r in archive_rows:
    dt_key = (r["scene_date"], r["scene_time"])
    by_datetime[dt_key].append(r)

    by_track[r["track_id"]].append(r)

    scene_code = f'{r["scene_date"]}_{r["scene_time"]}'
    by_scene_code[scene_code].append(r)

# vre ta unmatched
unmatched_ids = [sid for sid in samples if sid not in matched_ids]

rows_out = []
reason_counter = Counter()

for sid in unmatched_ids:
    s = samples[sid]

    scene_code = s["src_scene_code"]
    track_id = s["src_track_id"]
    date_ = s["src_date"]
    time_ = s["src_time"]

    scene_matches = by_scene_code.get(scene_code, [])
    dt_matches = by_datetime.get((date_, time_), [])
    track_matches = by_track.get(track_id, [])

    if not scene_matches:
        reason = "scene_not_present_in_index"
    elif dt_matches and not any(r["track_id"] == track_id for r in dt_matches):
        reason = "datetime_found_but_track_id_missing"
    elif track_matches and not any(
        (r["scene_date"] == date_ and r["scene_time"] == time_) for r in track_matches
    ):
        reason = "track_id_exists_elsewhere_but_not_in_this_scene"
    else:
        reason = "other_or_ambiguous"

    reason_counter[reason] += 1

    rows_out.append({
        "sample_id": sid,
        "src_info": s["src_info"],
        "src_scene_code": scene_code,
        "src_track_id": track_id,
        "reason": reason,
        "same_scene_entries": len(scene_matches),
        "same_datetime_entries": len(dt_matches),
        "same_trackid_entries": len(track_matches),
    })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", out_csv)
print("Unmatched total:", len(rows_out))
print("\nReason counts:")
for k, v in reason_counter.items():
    print(f"{k}: {v}")

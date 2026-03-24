import csv
import os
from collections import defaultdict

sample_csv = os.path.expanduser("~/imptc_project/results/sample_codes.csv")
archive_csv = os.path.expanduser("~/imptc_project/results/archive_index.csv")
out_csv = os.path.expanduser("~/imptc_project/results/matched_codes_full.csv")

samples = []
with open(sample_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        samples.append(row)

archive_index = defaultdict(list)
with open(archive_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["scene_date"], row["scene_time"], row["track_id"])
        archive_index[key].append(row)

rows_out = []
matched_count = 0
unmatched_count = 0

for s in samples:
    key = (s["src_date"], s["src_time"], s["src_track_id"])
    matches = archive_index.get(key, [])

    if matches:
        for m in matches:
            rows_out.append({
                "sample_id": s["sample_id"],
                "src_info": s["src_info"],
                "src_scene_code": s["src_scene_code"],
                "src_track_id": s["src_track_id"],
                "archive": m["archive"],
                "scene_path": m["scene_path"],
                "track_type": m["track_type"],
                "track_id": m["track_id"],
                "member_path": m["member_path"],
                "match_count_for_sample": len(matches),
                "match_status": "matched",
            })
        matched_count += 1
    else:
        rows_out.append({
            "sample_id": s["sample_id"],
            "src_info": s["src_info"],
            "src_scene_code": s["src_scene_code"],
            "src_track_id": s["src_track_id"],
            "archive": "",
            "scene_path": "",
            "track_type": "",
            "track_id": "",
            "member_path": "",
            "match_count_for_sample": 0,
            "match_status": "unmatched",
        })
        unmatched_count += 1

fieldnames = [
    "sample_id",
    "src_info",
    "src_scene_code",
    "src_track_id",
    "archive",
    "scene_path",
    "track_type",
    "track_id",
    "member_path",
    "match_count_for_sample",
    "match_status",
]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", out_csv)
print("Total samples   :", len(samples))
print("Matched samples :", matched_count)
print("Unmatched       :", unmatched_count)

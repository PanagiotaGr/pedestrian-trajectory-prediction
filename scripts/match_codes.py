import os
import csv

samples_csv = os.path.expanduser("~/imptc_project/results/sample_codes.csv")
archive_csv = os.path.expanduser("~/imptc_project/results/archive_index.csv")
out_csv = os.path.expanduser("~/imptc_project/results/matched_codes.csv")

samples = []
with open(samples_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        samples.append(row)

archive_rows = []
with open(archive_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        archive_rows.append(row)

matches = []

for s in samples:
    src_date = s["src_date"]
    src_time = s["src_time"]
    src_scene_idx = s["src_scene_idx"]
    src_track_id = s["src_track_id"]

    # candidate matching:
    # same date/time and same track_id
    candidates = [
        a for a in archive_rows
        if a["scene_date"] == src_date
        and a["scene_time"] == src_time
        and a["track_id"] == src_track_id
    ]

    # προαιρετικά, αν το src_scene_idx = 00394 και το archive scene_num = 0394,
    # δοκίμασε numeric comparison
    if src_scene_idx:
        try:
            s_idx_int = int(src_scene_idx)
            narrowed = []
            for a in candidates:
                try:
                    a_idx_int = int(a["scene_num"])
                    if a_idx_int == s_idx_int:
                        narrowed.append(a)
                except:
                    pass
            if narrowed:
                candidates = narrowed
        except:
            pass

    if candidates:
        for c in candidates:
            matches.append({
                "sample_id": s["sample_id"],
                "src_info": s["src_info"],
                "src_scene_code": s["src_scene_code"],
                "src_track_id": s["src_track_id"],
                "archive": c["archive"],
                "scene_path": c["scene_path"],
                "track_type": c["track_type"],
                "track_id": c["track_id"],
                "member_path": c["member_path"],
                "match_count_for_sample": len(candidates),
            })
    else:
        matches.append({
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
        })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
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
        ],
    )
    writer.writeheader()
    writer.writerows(matches)


out_csv = os.path.expanduser("~/imptc_project/results/matched_codes_full.csv")
total_samples = len(set(m["sample_id"] for m in matches))
matched_samples = len(set(m["sample_id"] for m in matches if m["match_count_for_sample"] != 0))
unmatched_samples = total_samples - matched_samples

print("Total samples   :", total_samples)
print("Matched samples :", matched_samples)
print("Unmatched       :", unmatched_samples)

print("\nFirst 20 matched rows:")
shown = 0
for m in matches:
    if m["match_count_for_sample"] != 0:
        print(m)
        shown += 1
    if shown >= 20:
        break

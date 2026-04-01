import os
import csv

root = os.path.expanduser("~/imptc_project/data/train")
out_csv = os.path.expanduser("~/imptc_project/results/sample_codes.csv")

rows = []

for sample_id in sorted(os.listdir(root), key=lambda x: int(x) if x.isdigit() else x):
    sample_path = os.path.join(root, sample_id)
    if not os.path.isdir(sample_path):
        continue

    src_file = os.path.join(sample_path, "src_info.txt")
    if not os.path.exists(src_file):
        continue

    with open(src_file, "r", encoding="utf-8", errors="replace") as f:
        src_info = f.read().strip()

    src_scene_code = ""
    src_track_id = ""

    if "/" in src_info:
        src_scene_code, src_track_id = src_info.rsplit("/", 1)
    else:
        src_scene_code = src_info

    parts = src_scene_code.split("_")

    src_date = parts[0] if len(parts) > 0 else ""
    src_time = parts[1] if len(parts) > 1 else ""
    src_block = parts[2] if len(parts) > 2 else ""
    src_scene_idx = parts[3] if len(parts) > 3 else ""

    rows.append({
        "sample_id": sample_id,
        "src_info": src_info,
        "src_scene_code": src_scene_code,
        "src_track_id": src_track_id,
        "src_date": src_date,
        "src_time": src_time,
        "src_block": src_block,
        "src_scene_idx": src_scene_idx,
    })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_id",
            "src_info",
            "src_scene_code",
            "src_track_id",
            "src_date",
            "src_time",
            "src_block",
            "src_scene_idx",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out_csv}")
print("First 10 rows:")
for r in rows[:10]:
    print(r)

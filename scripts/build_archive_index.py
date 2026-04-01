import os
import csv
import tarfile
import re

archives = [
    os.path.expanduser("~/imptc_project/data/imptc_set_01.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_02.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_03.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_04.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_05.tar.gz"),
]

out_csv = os.path.expanduser("~/imptc_project/results/archive_index.csv")

pattern = re.compile(r'^([^/]+)/(vrus|vehicles)/([^/]+)/track\.json$')

rows = []

for archive_path in archives:
    archive_name = os.path.basename(archive_path)
    print(f"Indexing {archive_name} ...")

    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue

            match = pattern.match(m.name)
            if not match:
                continue

            scene_path, track_type, track_id = match.groups()

            # scene_path π.χ. 0394_20230323_104843
            parts = scene_path.split("_")
            scene_num = parts[0] if len(parts) > 0 else ""
            scene_date = parts[1] if len(parts) > 1 else ""
            scene_time = parts[2] if len(parts) > 2 else ""

            rows.append({
                "archive": archive_name,
                "scene_path": scene_path,
                "scene_num": scene_num,
                "scene_date": scene_date,
                "scene_time": scene_time,
                "track_type": track_type,
                "track_id": track_id,
                "member_path": m.name,
            })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "archive",
            "scene_path",
            "scene_num",
            "scene_date",
            "scene_time",
            "track_type",
            "track_id",
            "member_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out_csv}")
print("First 10 rows:")
for r in rows[:10]:
    print(r)

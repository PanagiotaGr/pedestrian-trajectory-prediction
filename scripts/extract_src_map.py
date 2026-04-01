import os
import csv

root = os.path.expanduser("~/imptc_project/data/train")
out_csv = os.path.expanduser("~/imptc_project/results/src_mapping.csv")

rows = []

for sample_id in sorted(os.listdir(root)):
    sample_path = os.path.join(root, sample_id)
    if not os.path.isdir(sample_path):
        continue

    src_file = os.path.join(sample_path, "src_info.txt")
    track_file = os.path.join(sample_path, "track.json")

    src_info = ""
    if os.path.exists(src_file):
        with open(src_file, "rb") as f:
            raw = f.read()
        src_info = raw.decode("utf-8", errors="replace").strip()

    rows.append({
        "sample_id": sample_id,
        "src_info": src_info,
        "has_track_json": os.path.exists(track_file),
    })

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "src_info", "has_track_json"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {out_csv}")
print(f"Total rows: {len(rows)}")
print("First 20 non-empty src_info values:\n")

count = 0
for r in rows:
    if r["src_info"]:
        print(r["sample_id"], "->", r["src_info"])
        count += 1
    if count >= 20:
        break

import os
import json
import tarfile

sample_path = os.path.expanduser("~/imptc_project/data/train/3558/track.json")
archives = [
    os.path.expanduser("~/imptc_project/data/imptc_set_01.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_02.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_03.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_04.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_05.tar.gz"),
]

with open(sample_path, "r", encoding="utf-8") as f:
    sample = json.load(f)

s_over = sample["overview"]
s_data = sample["track_data"]

# signature του sample
sample_sig = (
    s_over.get("first_ts"),
    s_over.get("last_ts"),
    s_over.get("lenght"),
    s_over.get("class_name"),
)

# πρώτα 5 σημεία για ισχυρότερο match
sample_keys = sorted(s_data.keys(), key=lambda x: int(x))[:5]
sample_points = [
    (
        s_data[k]["ts"],
        tuple(round(v, 6) for v in s_data[k]["coordinates"])
    )
    for k in sample_keys
]

print("Sample signature:", sample_sig)
print("Sample first points:", sample_points)

found = False

for archive_path in archives:
    print(f"\nSearching in {os.path.basename(archive_path)} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith("/track.json"):
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                obj = json.load(f)

                o = obj.get("overview", {})
                d = obj.get("track_data", {})

                sig = (
                    o.get("first_ts"),
                    o.get("last_ts"),
                    o.get("lenght"),
                    o.get("class_name"),
                )

                if sig != sample_sig:
                    continue

                keys = sorted(d.keys(), key=lambda x: int(x))[:5]
                points = [
                    (
                        d[k]["ts"],
                        tuple(round(v, 6) for v in d[k]["coordinates"])
                    )
                    for k in keys
                ]

                if points == sample_points:
                    print("\nFOUND MATCH")
                    print("Archive:", archive_path)
                    print("Member :", member.name)
                    found = True
                    raise SystemExit

            except Exception:
                continue

if not found:
    print("\nNo exact match found.")

import tarfile
import os

target = "20230323_104843_0tb0000_00394"
archives = [
    os.path.expanduser("~/imptc_project/data/imptc_set_01.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_02.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_03.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_04.tar.gz"),
    os.path.expanduser("~/imptc_project/data/imptc_set_05.tar.gz"),
]

for archive_path in archives:
    print(f"\n=== Searching in {os.path.basename(archive_path)} ===")
    found = False
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            if not m.isfile():
                continue

            # ψάξε και στο όνομα
            if target in m.name:
                print("FOUND IN PATH:", m.name)
                found = True
                break

            # ψάξε μόνο σε text-like αρχεία
            if not any(m.name.endswith(ext) for ext in [".json", ".txt", ".csv"]):
                continue

            try:
                f = tar.extractfile(m)
                if f is None:
                    continue
                data = f.read(1024 * 1024)  # διαβάζει το πρώτο 1MB
                text = data.decode("utf-8", errors="ignore")
                if target in text:
                    print("FOUND IN CONTENT:", m.name)
                    found = True
                    break
            except Exception:
                pass

    if found:
        print(f"Scene likely belongs to: {os.path.basename(archive_path)}")
        break

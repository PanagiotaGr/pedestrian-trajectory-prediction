"""
load_full_dataset.py
====================
Κατεβάζει το full IMPTC trajectory dataset (2.9GB) από το Zenodo
και φορτώνει train / eval / test splits.

Κάθε track:
    track_id → {
        "class_name": str,
        "split":      "train" | "eval" | "test",
        "track":      [(ts, x, y, z, velocity), ...]
    }

Χρήση:
    python load_full_dataset.py

Απαιτήσεις:
    pip install requests tqdm
"""

import json
import tarfile
import requests
from tqdm import tqdm
from pathlib import Path
from collections import Counter

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
ZENODO_FULL_URL = (
    "https://zenodo.org/records/14811016/files/"
    "imptc_trajectory_dataset.tar.gz?download=1"
)
DOWNLOAD_DIR = Path("imptc_data")
EXTRACT_DIR  = Path("imptc_extracted_full")

# Class id → όνομα
CLASS_NAMES = {
    0: "pedestrian", 2: "cyclist", 3: "motorcycle",
    4: "scooter", 5: "stroller", 6: "wheelchair", 10: "unknown"
}


# ─── 1. Λήψη ────────────────────────────────────────────────────────────────
def download_file(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = dest_dir / "imptc_trajectory_dataset.tar.gz"

    if filename.exists():
        print(f"[✓] Αρχείο ήδη υπάρχει: {filename}")
        return filename

    print("[↓] Λήψη full trajectory dataset (~2.9 GB)...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc="imptc_trajectory_dataset.tar.gz"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[✓] Λήψη ολοκληρώθηκε: {filename}")
    return filename


# ─── 2. Αποσυμπίεση ─────────────────────────────────────────────────────────
def extract_archive(archive_path: Path, extract_to: Path):
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[✓] Dataset ήδη αποσυμπιεσμένο στο: {extract_to}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[↗] Αποσυμπίεση σε: {extract_to} (μπορεί να πάρει λίγο...)") 
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print("[✓] Αποσυμπίεση ολοκληρώθηκε.")


# ─── 3. Φόρτωση από train/eval/test_tracks.json ─────────────────────────────
def load_split_from_json(json_path: Path, split_name: str) -> dict:
    """
    Φορτώνει ένα train_tracks.json / eval_tracks.json / test_tracks.json.
    Επιστρέφει dict: track_id → {class_name, split, track: [(ts,x,y,z,vel),...]}
    """
    print(f"  Φόρτωση {split_name}: {json_path.name} ...", end=" ", flush=True)

    with open(json_path, "r") as f:
        raw = json.load(f)

    tracks = {}

    for track_id, data in raw.items():
        overview   = data.get("overview", {})
        track_data = data.get("track_data", {})

        class_id   = overview.get("class_id", 10)
        class_name = CLASS_NAMES.get(class_id, "unknown")

        # Ταξινόμηση timestamps
        def sort_key(k):
            parts = k.split("_")
            try:
                return int(parts[-1])
            except ValueError:
                return int(k)

        time_series = []
        for ts_key in sorted(track_data.keys(), key=sort_key):
            entry  = track_data[ts_key]
            coords = entry.get("coordinates", [None, None, None])
            time_series.append((
                entry.get("ts"),        # timestamp UTC ms
                coords[0],              # x (m)
                coords[1],              # y (m)
                coords[2],              # z (m)
                entry.get("velocity"),  # km/h
            ))

        full_id = f"{split_name}/{track_id}"
        tracks[full_id] = {
            "class_name": class_name,
            "class_id":   class_id,
            "split":      split_name,
            "track_id":   track_id,
            "n_frames":   len(time_series),
            "track":      time_series,
        }

    print(f"{len(tracks)} tracks φορτώθηκαν ✓")
    return tracks


# ─── 4. Φόρτωση όλων των splits ─────────────────────────────────────────────
def load_all_splits(extract_dir: Path) -> dict:
    """
    Βρίσκει τα train/eval/test_tracks.json και τα φορτώνει όλα.
    Επιστρέφει ένα ενιαίο dict με όλα τα tracks.
    """
    # Βρες το root folder μέσα στο extracted dir
    candidates = list(extract_dir.rglob("train_tracks.json"))
    if not candidates:
        print("[!] Δεν βρέθηκε train_tracks.json!")
        print(f"    Περιεχόμενο του {extract_dir}:")
        for p in extract_dir.rglob("*"):
            if p.is_file():
                print(f"      {p}")
        return {}

    dataset_root = candidates[0].parent
    print(f"[✓] Dataset root: {dataset_root}")

    all_tracks = {}
    print("\nΦόρτωση splits:")
    for split in ["train", "eval", "test"]:
        json_path = dataset_root / f"{split}_tracks.json"
        if json_path.exists():
            split_tracks = load_split_from_json(json_path, split)
            all_tracks.update(split_tracks)
        else:
            print(f"  [!] Δεν βρέθηκε: {json_path}")

    return all_tracks


# ─── 5. Στατιστικά ──────────────────────────────────────────────────────────
def print_summary(all_tracks: dict):
    print("\n" + "="*60)
    print(f"  Σύνολο tracks: {len(all_tracks)}")
    print("="*60)

    # Ανά split
    split_counts = Counter(v["split"] for v in all_tracks.values())
    print("\nΑνά split:")
    for split in ["train", "eval", "test"]:
        print(f"  {split:<8}: {split_counts.get(split, 0):>5} tracks")

    # Ανά κλάση
    class_counts = Counter(v["class_name"] for v in all_tracks.values())
    print("\nΑνά κλάση VRU:")
    for cls, cnt in class_counts.most_common():
        print(f"  {cls:<15}: {cnt:>5}")

    # Μέσος αριθμός frames
    avg_frames = sum(v["n_frames"] for v in all_tracks.values()) / len(all_tracks)
    print(f"\nΜέσος αριθμός frames ανά track: {avg_frames:.1f}")
    print(f"  → {avg_frames/25:.1f} sec @ 25Hz")

    # Παράδειγμα
    print("\nΠαράδειγμα (πρώτα 2 tracks):")
    for tid, info in list(all_tracks.items())[:2]:
        print(f"\n  ID     : {tid}")
        print(f"  Κλάση  : {info['class_name']}  |  Split: {info['split']}")
        print(f"  Frames : {info['n_frames']}")
        for row in info["track"][:2]:
            ts, x, y, z, vel = row
            print(f"    ts={ts}  x={x:.2f}m  y={y:.2f}m  vel={vel:.1f}km/h")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Κατέβασε
    archive = download_file(ZENODO_FULL_URL, DOWNLOAD_DIR)

    # 2. Αποσυμπίεσε
    extract_archive(archive, EXTRACT_DIR)

    # 3. Φόρτωσε
    all_tracks = load_all_splits(EXTRACT_DIR)

    if not all_tracks:
        print("[!] Κανένα track δεν φορτώθηκε. Έλεγξε το extracted folder.")
        exit(1)

    # 4. Στατιστικά
    print_summary(all_tracks)

    print("\n[✓] Dataset έτοιμο! Μεταβλητή: all_tracks")
    print("    Χρήση: all_tracks['train/0000']['track'] → [(ts,x,y,z,vel), ...]")

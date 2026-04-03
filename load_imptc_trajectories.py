"""
load_imptc_trajectories.py
==========================
Κατεβάζει το IMPTC sample dataset από το Zenodo και φορτώνει
για κάθε VRU track_id → λίστα από (timestamp, x, y, velocity)
σε όλες τις χρονικές στιγμές που διαρκεί η τροχιά του.

Χρήση:
    python load_imptc_trajectories.py

Απαιτήσεις:
    pip install requests tqdm
"""

import os
import json
import tarfile
import requests
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
ZENODO_SAMPLE_URL = (
    "https://zenodo.org/records/14811016/files/imptc_samples.tar.gz?download=1"
)
DOWNLOAD_DIR = Path("imptc_data")          # όπου αποθηκεύεται το αρχείο
EXTRACT_DIR  = Path("imptc_extracted")     # όπου αποσυμπιέζεται


# ─── 1. Λήψη αρχείου ────────────────────────────────────────────────────────
def download_file(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    filename = dest / "imptc_samples.tar.gz"

    if filename.exists():
        print(f"[✓] Αρχείο ήδη υπάρχει: {filename}")
        return filename

    print(f"[↓] Λήψη dataset από Zenodo (~347 MB)...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="imptc_samples.tar.gz"
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
    print(f"[↗] Αποσυμπίεση σε: {extract_to} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print("[✓] Αποσυμπίεση ολοκληρώθηκε.")


# ─── 3. Φόρτωση VRU trajectories ────────────────────────────────────────────
def load_vru_trajectories(extract_dir: Path) -> dict:
    """
    Επιστρέφει dict:
        trajectories[global_track_id] = {
            "class_name": str,
            "class_id":   int,
            "sequence":   str,
            "track":      [(timestamp_ms, x, y, z, velocity_kmh), ...]
        }
    """
    trajectories = {}

    # Ψάχνει όλα τα track.json μέσα στους φακέλους vrus/
    track_files = sorted(extract_dir.rglob("vrus/*/track.json"))

    if not track_files:
        print("[!] Δεν βρέθηκαν VRU track.json αρχεία.")
        print(f"    Ψάξε μέσα στο: {extract_dir}")
        return {}

    print(f"[✓] Βρέθηκαν {len(track_files)} VRU tracks.")

    for track_path in track_files:
        # global_id = sequence_id / track_id  π.χ. "002/000"
        track_id   = track_path.parent.name          # π.χ. "000"
        seq_id     = track_path.parent.parent.parent.name  # π.χ. "002"
        global_id  = f"{seq_id}/{track_id}"

        with open(track_path, "r") as f:
            data = json.load(f)

        overview   = data.get("overview", {})
        track_data = data.get("track_data", {})

        # Κατασκευή χρονοσειράς: ταξινόμηση κατά timestamp key
        time_series = []
        def sort_key(k):
            parts = k.split("_")
            try:
                return int(parts[-1])
            except ValueError:
                return int(k)

        for ts_key in sorted(track_data.keys(), key=sort_key):
            entry = track_data[ts_key]
            coords = entry.get("coordinates", [None, None, None])
            time_series.append((
                entry.get("ts"),           # timestamp (ms UTC)
                coords[0],                 # x (m)
                coords[1],                 # y (m)
                coords[2],                 # z (m)
                entry.get("velocity"),     # ταχύτητα (km/h)
                entry.get("ground_type"),  # τύπος εδάφους
            ))

        trajectories[global_id] = {
            "class_name": overview.get("class_name", "unknown"),
            "class_id":   overview.get("class_id", -1),
            "sequence":   seq_id,
            "track_id":   track_id,
            "duration_s": overview.get("duration", 0) / 1000,  # ms → sec
            "length":     overview.get("length", 0),            # αριθμός frames
            "track":      time_series,
        }

    return trajectories


# ─── 4. Εκτύπωση περίληψης ──────────────────────────────────────────────────
def print_summary(trajectories: dict):
    print("\n" + "="*60)
    print(f"  Σύνολο VRU tracks: {len(trajectories)}")
    print("="*60)

    # Στατιστικά ανά κλάση
    from collections import Counter
    class_counts = Counter(v["class_name"] for v in trajectories.values())
    print("\nΚατανομή ανά τύπο VRU:")
    for cls, cnt in class_counts.most_common():
        print(f"  {cls:<15} : {cnt}")

    # Παράδειγμα εκτύπωσης πρώτων 3 tracks
    print("\nΠαράδειγμα (πρώτα 3 tracks):")
    for gid, info in list(trajectories.items())[:3]:
        print(f"\n  Track ID : {gid}")
        print(f"  Κλάση    : {info['class_name']}")
        print(f"  Διάρκεια : {info['duration_s']:.1f} sec  ({info['length']} frames)")
        print(f"  Χρον. σημεία (πρώτα 3):")
        for row in info["track"][:3]:
            ts, x, y, z, vel, gtype = row
            print(f"    ts={ts}  x={x:.2f}m  y={y:.2f}m  vel={vel:.1f}km/h  ground={gtype}")


# ─── 5. Βοηθητική συνάρτηση: tracks ανά χρονική στιγμή ─────────────────────
def build_timestep_index(trajectories: dict) -> dict:
    """
    Επιστρέφει dict:
        timestep_index[timestamp_ms] = [
            {"global_id": ..., "x": ..., "y": ..., "velocity": ...},
            ...
        ]
    Χρήσιμο για: «ποιοι VRUs υπάρχουν σε κάθε χρονική στιγμή;»
    """
    index = defaultdict(list)
    for gid, info in trajectories.items():
        for (ts, x, y, z, vel, gtype) in info["track"]:
            index[ts].append({
                "global_id":  gid,
                "class_name": info["class_name"],
                "x": x, "y": y, "z": z,
                "velocity": vel,
            })
    return dict(index)


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Βήμα 1: Λήψη
    archive = download_file(ZENODO_SAMPLE_URL, DOWNLOAD_DIR)

    # Βήμα 2: Αποσυμπίεση
    extract_archive(archive, EXTRACT_DIR)

    # Βήμα 3: Φόρτωση trajectories
    trajectories = load_vru_trajectories(EXTRACT_DIR)

    # Βήμα 4: Περίληψη
    print_summary(trajectories)

    # Βήμα 5: Index ανά χρονική στιγμή (για μελλοντική χρήση)
    ts_index = build_timestep_index(trajectories)
    print(f"\n[✓] Timestep index: {len(ts_index)} μοναδικές χρονικές στιγμές.")

    # Το trajectories dict είναι έτοιμο για χρήση!
    # Παράδειγμα πρόσβασης:
    #   for gid, info in trajectories.items():
    #       for (ts, x, y, z, vel, gtype) in info["track"]:
    #           ...

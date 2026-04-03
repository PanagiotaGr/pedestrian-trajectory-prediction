"""
download_and_build.py
=====================
1. Κατεβάζει όλα τα IMPTC sets από το Zenodo (~17GB)
2. Τρέχει το build_pedestrian_math_dataset_v2.py για όλα

Χρήση:
    python download_and_build.py

Απαιτήσεις:
    pip install requests tqdm
"""

import os
import subprocess
import requests
from tqdm import tqdm
from pathlib import Path

# ─── Ρυθμίσεις ──────────────────────────────────────────────────────────────
DATA_DIR     = Path(os.path.expanduser("~/imptc_project/data"))
RESULTS_DIR  = Path(os.path.expanduser("~/imptc_project/results"))
SCRIPTS_DIR  = Path(os.path.expanduser("~/imptc_project/scripts"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_BASE = "https://zenodo.org/records/14811016/files"

FILES = [
    ("imptc_set_01.tar.gz", "2.7GB"),
    ("imptc_set_02.tar.gz", "2.5GB"),
    ("imptc_set_03.tar.gz", "2.9GB"),
    ("imptc_set_04.tar.gz", "3.6GB"),
    ("imptc_set_05.tar.gz", "5.1GB"),
]


# ─── Download ────────────────────────────────────────────────────────────────
def download_file(filename: str, size_str: str) -> Path:
    dest = DATA_DIR / filename

    if dest.exists():
        print(f"[✓] Ήδη υπάρχει: {filename}")
        return dest

    url = f"{ZENODO_BASE}/{filename}?download=1"
    print(f"\n[↓] Κατέβασμα {filename} (~{size_str})...")

    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=filename
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[✓] Ολοκληρώθηκε: {dest}")
    return dest


# ─── Run pipeline ────────────────────────────────────────────────────────────
def run_pipeline():
    script = SCRIPTS_DIR / "build_pedestrian_math_dataset_v2.py"
    out_csv = RESULTS_DIR / "pedestrian_math_dataset_v2_full.csv"

    if not script.exists():
        print(f"[!] Script δεν βρέθηκε: {script}")
        return

    print(f"\n[→] Τρέχω: {script.name}")
    print(f"    archives-dir: {DATA_DIR}")
    print(f"    out-csv:      {out_csv}")

    cmd = [
        "python", str(script),
        "--detailed-csv", str(RESULTS_DIR / "interactions_detailed.csv"),
        "--archives-dir", str(DATA_DIR),
        "--out-csv",      str(out_csv),
    ]

    result = subprocess.run(cmd, cwd=str(Path.home() / "imptc_project"))

    if result.returncode == 0:
        print(f"\n[✓] Pipeline ολοκληρώθηκε!")
        print(f"    Output: {out_csv}")
    else:
        print(f"\n[!] Σφάλμα στο pipeline (exit code {result.returncode})")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  IMPTC Full Download & Build Pipeline")
    print("="*55)
    print(f"  Αποθήκευση στο: {DATA_DIR}")
    print(f"  Σύνολο: ~17GB")
    print("="*55)

    # 1. Κατέβασε όλα τα sets
    for filename, size in FILES:
        download_file(filename, size)

    print("\n[✓] Όλα τα archives κατεβήκαν!")

    # 2. Τρέξε το pipeline
    run_pipeline()

    print("\n[✓] Τέλος!")

"""
IMPTC Dataset Mapper
====================
Διαβάζει τα src_info.txt από το VRU trajectory dataset (μικρό)
και βρίσκει τα αντίστοιχα sequences στο μεγάλο dataset (imptc_set_01-05).

Χρήση:
  python imptc_mapper.py --traj_dir ./imptc_trajectory_dataset --output mapping.csv

Επίσης δημιουργεί:
  - mapping.csv      : αντιστοίχιση trajectory → sequence → set file
  - summary.txt      : σύνοψη ποια set files χρειάζεσαι
  - extract_cmds.sh  : έτοιμες εντολές για εξαγωγή μόνο των sequences που θέλεις
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────
# Αντιστοίχιση sequence id → set file
# Sequences 000-049 → imptc_set_01
# Sequences 050-099 → imptc_set_02
# Sequences 100-149 → imptc_set_03
# Sequences 150-199 → imptc_set_04
# Sequences 200-269 → imptc_set_05
# ──────────────────────────────────────────────
def sequence_to_set(seq_id: int) -> str:
    if seq_id < 50:
        return "imptc_set_01.tar.gz"
    elif seq_id < 100:
        return "imptc_set_02.tar.gz"
    elif seq_id < 150:
        return "imptc_set_03.tar.gz"
    elif seq_id < 200:
        return "imptc_set_04.tar.gz"
    else:
        return "imptc_set_05.tar.gz"


def parse_src_info(filepath: Path) -> dict:
    """
    Διαβάζει ένα src_info.txt και επιστρέφει dict με τα πεδία.
    Το αρχείο μπορεί να είναι:
      - JSON format: {"sequence_id": 42, "track_id": "005", ...}
      - Key: Value format:  sequence_id: 42
    """
    info = {}
    text = filepath.read_text(encoding="utf-8").strip()

    # Δοκίμασε JSON πρώτα
    try:
        info = json.loads(text)
        return info
    except json.JSONDecodeError:
        pass

    # Fallback: key: value ανά γραμμή
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            info[key.strip()] = val.strip()

    return info


def extract_sequence_id(info: dict) -> int | None:
    """Βγάζει το sequence id από διάφορα πιθανά ονόματα κλειδιών."""
    for key in ("sequence_id", "seq_id", "sequence", "src_sequence", "source_sequence"):
        if key in info:
            try:
                return int(info[key])
            except (ValueError, TypeError):
                # ίσως είναι string όπως "042"
                try:
                    return int(str(info[key]).lstrip("0") or "0")
                except ValueError:
                    pass
    return None


def scan_trajectory_dataset(traj_dir: Path) -> list[dict]:
    """
    Σαρώνει όλα τα train/eval/test subfolders και μαζεύει src_info για κάθε trajectory.
    """
    results = []
    splits = ["train", "eval", "test"]

    for split in splits:
        split_dir = traj_dir / split
        if not split_dir.exists():
            print(f"  [!] Δεν βρέθηκε: {split_dir}")
            continue

        traj_folders = sorted(split_dir.iterdir())
        print(f"  [{split}] Βρέθηκαν {len(traj_folders)} trajectories")

        for traj_folder in traj_folders:
            if not traj_folder.is_dir():
                continue

            src_file = traj_folder / "src_info.txt"
            if not src_file.exists():
                print(f"    [!] Δεν βρέθηκε src_info.txt σε {traj_folder}")
                continue

            info = parse_src_info(src_file)
            seq_id = extract_sequence_id(info)

            row = {
                "split": split,
                "trajectory_id": traj_folder.name,
                "trajectory_path": str(traj_folder),
                "sequence_id": seq_id if seq_id is not None else "UNKNOWN",
                "set_file": sequence_to_set(seq_id) if seq_id is not None else "UNKNOWN",
                "raw_src_info": str(info),
            }
            results.append(row)

    return results


def write_mapping_csv(results: list[dict], output_path: Path):
    fieldnames = ["split", "trajectory_id", "sequence_id", "set_file", "trajectory_path", "raw_src_info"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Mapping αποθηκεύτηκε: {output_path}")


def write_summary(results: list[dict], output_path: Path):
    # Ποια set files χρειάζονται ανά split
    split_sets = defaultdict(set)
    set_sequences = defaultdict(set)
    set_trajectories = defaultdict(list)

    for r in results:
        split_sets[r["split"]].add(r["set_file"])
        if r["sequence_id"] != "UNKNOWN":
            set_sequences[r["set_file"]].add(int(r["sequence_id"]))
        set_trajectories[r["set_file"]].append(r["trajectory_id"])

    all_sets_needed = set()
    for sets in split_sets.values():
        all_sets_needed.update(sets)

    lines = []
    lines.append("=" * 60)
    lines.append("IMPTC DATASET MAPPER — ΣΥΝΟΨΗ")
    lines.append("=" * 60)
    lines.append(f"\nΣύνολο trajectories: {len(results)}")
    lines.append(f"Set files που χρειάζεσαι: {sorted(all_sets_needed)}\n")

    for split in ["train", "eval", "test"]:
        sets_for_split = split_sets.get(split, set())
        lines.append(f"[{split.upper()}] χρειάζεται: {sorted(sets_for_split)}")

    lines.append("\n" + "-" * 60)
    lines.append("ΑΝΑΛΥΣΗ ΑΝΑ SET FILE:")
    lines.append("-" * 60)

    set_md5 = {
        "imptc_set_01.tar.gz": "a1231057d2edac6daebcb1d39bcd5f25",
        "imptc_set_02.tar.gz": "5601c69c8c965e5d93206ccab04ced6c",
        "imptc_set_03.tar.gz": "5b0de174a1fd3c9d374b8d1613fd563a",
        "imptc_set_04.tar.gz": "39029256faa5d5a57b62b43245766d29",
        "imptc_set_05.tar.gz": "bfdcbae1a2dfb293bd4f6ca3723fe8b5",
    }

    for set_file in sorted(all_sets_needed):
        seqs = sorted(set_sequences.get(set_file, []))
        n_traj = len(set_trajectories.get(set_file, []))
        md5 = set_md5.get(set_file, "N/A")
        lines.append(f"\n{set_file}")
        lines.append(f"  MD5: {md5}")
        lines.append(f"  Trajectories: {n_traj}")
        lines.append(f"  Sequences: {seqs}")

    lines.append("\n" + "=" * 60)

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(f"✓ Summary αποθηκεύτηκε: {output_path}")
    print("\n" + text)


def write_extract_commands(results: list[dict], output_path: Path):
    """
    Δημιουργεί shell script με tar εντολές για εξαγωγή ΜΟΝΟ
    των sequences που χρειάζεσαι από κάθε set file.
    """
    set_sequences = defaultdict(set)
    for r in results:
        if r["sequence_id"] != "UNKNOWN":
            set_sequences[r["set_file"]].add(int(r["sequence_id"]))

    lines = ["#!/bin/bash", "# Εξαγωγή μόνο των απαραίτητων sequences από τα μεγάλα set files", ""]

    for set_file in sorted(set_sequences.keys()):
        seqs = sorted(set_sequences[set_file])
        lines.append(f"# ── {set_file} ──")
        lines.append(f"# Sequences που χρειάζεσαι: {seqs}")
        lines.append("")

        # Εξαγωγή κάθε sequence χωριστά (αποφεύγει εξαγωγή ολόκληρου του αρχείου)
        patterns = " ".join([f'"{seq_id:03d}/*"' for seq_id in seqs])
        lines.append(f"tar -xzf {set_file} {patterns} -C ./extracted/")
        lines.append("")

    lines.append("echo 'Εξαγωγή ολοκληρώθηκε!'")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.chmod(0o755)
    print(f"✓ Extract commands αποθηκεύτηκαν: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Αντιστοίχιση trajectories από το μικρό dataset στα μεγάλα set files."
    )
    parser.add_argument(
        "--traj_dir",
        type=Path,
        default=Path("./imptc_trajectory_dataset"),
        help="Φάκελος με το αποσυμπιεσμένο imptc_trajectory_dataset (default: ./imptc_trajectory_dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./imptc_output"),
        help="Φάκελος για τα αρχεία εξόδου (default: ./imptc_output)",
    )
    args = parser.parse_args()

    if not args.traj_dir.exists():
        print(f"[ERROR] Ο φάκελος {args.traj_dir} δεν βρέθηκε!")
        print("Βήματα:")
        print("  1. Κατέβασε το imptc_trajectory_dataset.tar.gz από το Zenodo")
        print("  2. Αποσυμπίεσε: tar -xzf imptc_trajectory_dataset.tar.gz")
        print("  3. Τρέξε ξανά: python imptc_mapper.py --traj_dir ./imptc_trajectory_dataset")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nΣάρωση: {args.traj_dir}")
    results = scan_trajectory_dataset(args.traj_dir)

    if not results:
        print("[ERROR] Δεν βρέθηκαν trajectories!")
        return

    write_mapping_csv(results, args.output_dir / "mapping.csv")
    write_summary(results, args.output_dir / "summary.txt")
    write_extract_commands(results, args.output_dir / "extract_sequences.sh")


if __name__ == "__main__":
    main()

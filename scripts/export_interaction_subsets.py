import csv
import os
import argparse


VEHICLE_CLASSES = {"car", "truck", "bus"}
VRU_CLASSES = {"person", "bicycle", "scooter", "motorcycle", "stroller"}


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def interaction_family(row):
    other = (row.get("other_class_name") or "").strip().lower()
    target = (row.get("target_class_name") or "").strip().lower()

    if target in VRU_CLASSES and other in VEHICLE_CLASSES:
        return "vru_vehicle"

    if target in VRU_CLASSES and other in VRU_CLASSES:
        return "vru_vru"

    return "other"


def main():
    parser = argparse.ArgumentParser(description="Export VRU interaction subsets")
    parser.add_argument(
        "--input-csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map_and_lights.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.expanduser("~/imptc_project/results/subsets"),
    )
    parser.add_argument("--close-threshold", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_rows(args.input_csv)
    print("Loaded rows:", len(rows))

    enriched = []
    for r in rows:
        rr = dict(r)
        rr["interaction_family"] = interaction_family(r)
        enriched.append(rr)

    fieldnames = list(enriched[0].keys())

    vru_vehicle = [r for r in enriched if r["interaction_family"] == "vru_vehicle"]
    vru_vru = [r for r in enriched if r["interaction_family"] == "vru_vru"]

    close_vru_vehicle = [r for r in vru_vehicle if float(r["dist_xy"]) <= args.close_threshold]
    close_vru_vru = [r for r in vru_vru if float(r["dist_xy"]) <= args.close_threshold]

    write_rows(os.path.join(args.out_dir, "all_interactions_labeled.csv"), enriched, fieldnames)
    write_rows(os.path.join(args.out_dir, "vru_vehicle_cases.csv"), vru_vehicle, fieldnames)
    write_rows(os.path.join(args.out_dir, "vru_vru_cases.csv"), vru_vru, fieldnames)
    write_rows(os.path.join(args.out_dir, "close_vru_vehicle_cases.csv"), close_vru_vehicle, fieldnames)
    write_rows(os.path.join(args.out_dir, "close_vru_vru_cases.csv"), close_vru_vru, fieldnames)

    print("Saved:")
    print(" -", os.path.join(args.out_dir, "all_interactions_labeled.csv"), len(enriched))
    print(" -", os.path.join(args.out_dir, "vru_vehicle_cases.csv"), len(vru_vehicle))
    print(" -", os.path.join(args.out_dir, "vru_vru_cases.csv"), len(vru_vru))
    print(" -", os.path.join(args.out_dir, "close_vru_vehicle_cases.csv"), len(close_vru_vehicle))
    print(" -", os.path.join(args.out_dir, "close_vru_vru_cases.csv"), len(close_vru_vru))


if __name__ == "__main__":
    main()

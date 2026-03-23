import os
import csv
import math
import argparse
from collections import defaultdict

"""
Build nearest-K social interaction features for each pedestrian sample.

Input:
    pedestrian_moments_neighbors.csv

For each sample_id:
    - keep only neighbors within radius
    - sort them by distance
    - keep top-K nearest agents
    - encode relative geometry + speed + type/class

Output:
    pedestrian_social_nearestk.csv
"""

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def encode_type(t):
    # simple numeric encoding
    mapping = {
        "vrus": 1,
        "vehicles": 2,
    }
    return mapping.get(str(t).strip(), 0)

def encode_class(c):
    # extend if needed
    mapping = {
        "person": 1,
        "car": 2,
        "truck": 3,
        "scooter": 4,
        "bicycle": 5,
        "motorcycle": 6,
        "stroller": 7,
    }
    return mapping.get(str(c).strip(), 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
    )
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_social_nearestk.csv")
    )
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    grouped = defaultdict(list)

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["sample_id"]].append(row)

    rows_out = []

    for sample_id, rows in grouped.items():
        # every row of same sample has same target
        first = rows[0]

        target_x = safe_float(first["target_x"])
        target_y = safe_float(first["target_y"])
        target_velocity = safe_float(first["target_velocity"])

        neighbors = []

        for r in rows:
            dist = safe_float(r["distance"], default=9999.0)

            # keep only local neighbors
            if dist > args.radius:
                continue

            other_x = safe_float(r["other_x"])
            other_y = safe_float(r["other_y"])
            other_velocity = safe_float(r["other_velocity"])

            dx = other_x - target_x
            dy = other_y - target_y

            neighbors.append({
                "distance": dist,
                "dx": dx,
                "dy": dy,
                "other_velocity": other_velocity,
                "other_type": r["other_type"],
                "other_class_name": r["other_class_name"],
                "other_type_enc": encode_type(r["other_type"]),
                "other_class_enc": encode_class(r["other_class_name"]),
            })

        # nearest first
        neighbors.sort(key=lambda z: z["distance"])

        # keep only K nearest
        neighbors = neighbors[:args.top_k]

        out_row = {
            "sample_id": sample_id,
            "target_velocity": target_velocity,
            "neighbor_count_within_radius": len(neighbors),
        }

        # fixed-size representation
        for i in range(args.top_k):
            prefix = f"nbr_{i+1}"

            if i < len(neighbors):
                n = neighbors[i]
                out_row[f"{prefix}_dx"] = round(n["dx"], 6)
                out_row[f"{prefix}_dy"] = round(n["dy"], 6)
                out_row[f"{prefix}_distance"] = round(n["distance"], 6)
                out_row[f"{prefix}_velocity"] = round(n["other_velocity"], 6)
                out_row[f"{prefix}_type"] = n["other_type"]
                out_row[f"{prefix}_class"] = n["other_class_name"]
                out_row[f"{prefix}_type_enc"] = n["other_type_enc"]
                out_row[f"{prefix}_class_enc"] = n["other_class_enc"]
            else:
                # zero padding
                out_row[f"{prefix}_dx"] = 0.0
                out_row[f"{prefix}_dy"] = 0.0
                out_row[f"{prefix}_distance"] = 0.0
                out_row[f"{prefix}_velocity"] = 0.0
                out_row[f"{prefix}_type"] = ""
                out_row[f"{prefix}_class"] = ""
                out_row[f"{prefix}_type_enc"] = 0
                out_row[f"{prefix}_class_enc"] = 0

        rows_out.append(out_row)

    # build field order
    fieldnames = ["sample_id", "target_velocity", "neighbor_count_within_radius"]
    for i in range(args.top_k):
        prefix = f"nbr_{i+1}"
        fieldnames += [
            f"{prefix}_dx",
            f"{prefix}_dy",
            f"{prefix}_distance",
            f"{prefix}_velocity",
            f"{prefix}_type",
            f"{prefix}_class",
            f"{prefix}_type_enc",
            f"{prefix}_class_enc",
        ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print("Saved:", args.output)
    print("Rows:", len(rows_out))
    print("Radius:", args.radius)
    print("Top-K:", args.top_k)

if __name__ == "__main__":
    main()

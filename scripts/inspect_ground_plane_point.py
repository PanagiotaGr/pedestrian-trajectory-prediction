import csv
import math
import argparse


def load_ground_points(csv_path):
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "ground_type_id": row["ground_type_id"],
                    "ground_type_name": row["ground_type_name"],
                    "r": row.get("r"),
                    "g": row.get("g"),
                    "b": row.get("b"),
                })
            except Exception:
                continue
    return points


def nearest_point(points, x, y):
    best = None
    best_d2 = None

    for p in points:
        dx = p["x"] - x
        dy = p["y"] - y
        d2 = dx * dx + dy * dy

        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best = p

    return best, math.sqrt(best_d2) if best_d2 is not None else None


def main():
    parser = argparse.ArgumentParser(description="Inspect nearest ground-plane label for one point")
    parser.add_argument("ground_csv", help="CSV exported from ground plane map")
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    args = parser.parse_args()

    points = load_ground_points(args.ground_csv)
    print("Loaded ground points:", len(points))

    best, dist = nearest_point(points, args.x, args.y)
    if best is None:
        print("No ground point found.")
        return

    print("Query point:", (args.x, args.y))
    print("Nearest map point:", (best["x"], best["y"], best["z"]))
    print("XY distance:", dist)
    print("Ground type id:", best["ground_type_id"])
    print("Ground type name:", best["ground_type_name"])
    print("RGB:", (best["r"], best["g"], best["b"]))


if __name__ == "__main__":
    main()

import os
import csv
import math
import argparse
from collections import defaultdict

"""
Improved GRID encoding with:
- count
- mean velocity
- max velocity
- min distance

Handles multiple agents per cell correctly.
"""

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def get_cell(dx, dy, radius, grid_size):
    if dx < -radius or dx > radius or dy < -radius or dy > radius:
        return None

    width = 2 * radius
    cell_size = width / grid_size

    x_shift = dx + radius
    y_shift = dy + radius

    col = min(int(x_shift / cell_size), grid_size - 1)
    row = min(int(y_shift / cell_size), grid_size - 1)

    return row * grid_size + col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--grid-size", type=int, default=3)

    args = parser.parse_args()

    input_path = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
    output_path = os.path.expanduser("~/imptc_project/results/pedestrian_social_grid.csv")

    grouped = defaultdict(list)

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["sample_id"]].append(row)

    n_cells = args.grid_size * args.grid_size
    rows_out = []

    for sample_id, rows in grouped.items():

        first = rows[0]

        tx = safe_float(first["target_x"])
        ty = safe_float(first["target_y"])
        tvel = safe_float(first["target_velocity"])

        # accumulators
        cell_counts = [0]*n_cells
        cell_vel_sum = [0.0]*n_cells
        cell_max_vel = [0.0]*n_cells
        cell_min_dist = [None]*n_cells

        for r in rows:
            dist = safe_float(r["distance"], 9999.0)

            if dist > args.radius:
                continue

            ox = safe_float(r["other_x"])
            oy = safe_float(r["other_y"])
            vel = safe_float(r["other_velocity"])

            dx = ox - tx
            dy = oy - ty

            cell = get_cell(dx, dy, args.radius, args.grid_size)
            if cell is None:
                continue

            # count
            cell_counts[cell] += 1

            # sum velocity (για mean)
            cell_vel_sum[cell] += vel

            # max velocity
            if vel > cell_max_vel[cell]:
                cell_max_vel[cell] = vel

            # min distance
            if cell_min_dist[cell] is None or dist < cell_min_dist[cell]:
                cell_min_dist[cell] = dist

        # build row
        out = {
            "sample_id": sample_id,
            "target_velocity": round(tvel, 6),
        }

        for i in range(n_cells):
            count = cell_counts[i]

            mean_vel = cell_vel_sum[i] / count if count > 0 else 0.0
            max_vel = cell_max_vel[i]
            min_dist = cell_min_dist[i] if cell_min_dist[i] is not None else 0.0

            out[f"cell_{i}_count"] = count
            out[f"cell_{i}_mean_velocity"] = round(mean_vel, 6)
            out[f"cell_{i}_max_velocity"] = round(max_vel, 6)
            out[f"cell_{i}_min_distance"] = round(min_dist, 6)

        rows_out.append(out)

    # fields
    fieldnames = ["sample_id", "target_velocity"]
    for i in range(n_cells):
        fieldnames += [
            f"cell_{i}_count",
            f"cell_{i}_mean_velocity",
            f"cell_{i}_max_velocity",
            f"cell_{i}_min_distance",
        ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print("Saved:", output_path)
    print("Rows:", len(rows_out))


if __name__ == "__main__":
    main()

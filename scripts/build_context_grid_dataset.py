import os
import csv
import math
from collections import defaultdict

"""
Build a context-aware grid representation around each pedestrian sample.

For each sample_id:
- target pedestrian is treated as the origin (0,0)
- only neighbors within radius R are kept
- local square [-R, R] x [-R, R] is split into GRID_SIZE x GRID_SIZE cells
- each cell stores:
    * count of nearby agents
    * mean velocity
    * minimum distance
    * count of pedestrians
    * count of vehicles
- pedestrian traffic light states are merged:
    * f1_state, f2_state, f3_state

Input files:
    pedestrian_moments_neighbors.csv
    pedestrian_light_features.csv

Output:
    pedestrian_context_grid_dataset.csv
"""

NEIGHBORS_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
LIGHTS_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_light_features.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_context_grid_dataset.csv")

RADIUS = 5.0
GRID_SIZE = 3


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def get_cell_index(dx, dy, radius=5.0, grid_size=3):
    """
    Map relative coordinates (dx, dy) into a grid cell index.

    Space considered:
        x in [-radius, radius]
        y in [-radius, radius]

    Returns:
        index in [0, grid_size*grid_size - 1]
        or None if outside the local square
    """
    if dx < -radius or dx > radius or dy < -radius or dy > radius:
        return None

    width = 2.0 * radius
    cell_size = width / grid_size

    # shift coordinates so that [-R, R] becomes [0, 2R]
    x_shift = dx + radius
    y_shift = dy + radius

    col = min(int(x_shift / cell_size), grid_size - 1)
    row = min(int(y_shift / cell_size), grid_size - 1)

    return row * grid_size + col


def load_light_features(path):
    """
    Load per-sample pedestrian traffic light features.
    """
    lights = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lights[row["sample_id"]] = row
    return lights


def main():
    # 1) load traffic light info
    light_map = load_light_features(LIGHTS_PATH)

    # 2) group neighbor rows by sample_id
    grouped = defaultdict(list)
    with open(NEIGHBORS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["sample_id"]].append(row)

    n_cells = GRID_SIZE * GRID_SIZE
    rows_out = []

    for sample_id, rows in grouped.items():
        # same target info in all rows of a sample
        first = rows[0]

        target_x = safe_float(first["target_x"])
        target_y = safe_float(first["target_y"])
        target_velocity = safe_float(first["target_velocity"])

        # per-cell accumulators
        cell_counts = [0] * n_cells
        cell_vel_sum = [0.0] * n_cells
        cell_min_dist = [None] * n_cells
        cell_ped_count = [0] * n_cells
        cell_vehicle_count = [0] * n_cells

        # global local-neighborhood stats
        total_neighbors_within_radius = 0
        total_ped_neighbors = 0
        total_vehicle_neighbors = 0

        # optional nearest-neighbor summary
        nearest_distance = None
        nearest_velocity = None
        nearest_type = ""
        nearest_class = ""

        for r in rows:
            dist = safe_float(r["distance"], default=9999.0)

            # keep only local neighbors
            if dist > RADIUS:
                continue

            other_x = safe_float(r["other_x"])
            other_y = safe_float(r["other_y"])
            other_velocity = safe_float(r["other_velocity"])
            other_type = str(r["other_type"]).strip()
            other_class = str(r["other_class_name"]).strip()

            dx = other_x - target_x
            dy = other_y - target_y

            # local square filter
            cell = get_cell_index(dx, dy, radius=RADIUS, grid_size=GRID_SIZE)
            if cell is None:
                continue

            total_neighbors_within_radius += 1

            if other_type == "vrus":
                total_ped_neighbors += 1
            elif other_type == "vehicles":
                total_vehicle_neighbors += 1

            cell_counts[cell] += 1
            cell_vel_sum[cell] += other_velocity

            if cell_min_dist[cell] is None or dist < cell_min_dist[cell]:
                cell_min_dist[cell] = dist

            if other_type == "vrus":
                cell_ped_count[cell] += 1
            elif other_type == "vehicles":
                cell_vehicle_count[cell] += 1

            # nearest neighbor overall
            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist
                nearest_velocity = other_velocity
                nearest_type = other_type
                nearest_class = other_class

        # 3) build output row
        out_row = {
            "sample_id": sample_id,
            "target_velocity": round(target_velocity, 6),
            "total_neighbors_within_radius": total_neighbors_within_radius,
            "total_ped_neighbors": total_ped_neighbors,
            "total_vehicle_neighbors": total_vehicle_neighbors,
            "nearest_distance_within_radius": round(nearest_distance, 6) if nearest_distance is not None else 0.0,
            "nearest_velocity_within_radius": round(nearest_velocity, 6) if nearest_velocity is not None else 0.0,
            "nearest_type_within_radius": nearest_type,
            "nearest_class_within_radius": nearest_class,
        }

        # 4) add grid cell features
        for i in range(n_cells):
            count = cell_counts[i]
            mean_vel = cell_vel_sum[i] / count if count > 0 else 0.0
            min_dist = cell_min_dist[i] if cell_min_dist[i] is not None else 0.0

            out_row[f"cell_{i}_count"] = count
            out_row[f"cell_{i}_mean_velocity"] = round(mean_vel, 6)
            out_row[f"cell_{i}_min_distance"] = round(min_dist, 6)
            out_row[f"cell_{i}_ped_count"] = cell_ped_count[i]
            out_row[f"cell_{i}_vehicle_count"] = cell_vehicle_count[i]

        # 5) add traffic light vector
        light_row = light_map.get(sample_id, {})

        out_row["f1_state"] = safe_int(light_row.get("f1_state", 0), 0)
        out_row["f2_state"] = safe_int(light_row.get("f2_state", 0), 0)
        out_row["f3_state"] = safe_int(light_row.get("f3_state", 0), 0)

        out_row["n_ped_red"] = safe_int(light_row.get("n_ped_red", 0), 0)
        out_row["n_ped_green"] = safe_int(light_row.get("n_ped_green", 0), 0)
        out_row["n_ped_off"] = safe_int(light_row.get("n_ped_off", 0), 0)

        rows_out.append(out_row)

    # 6) fieldnames
    fieldnames = [
        "sample_id",
        "target_velocity",
        "total_neighbors_within_radius",
        "total_ped_neighbors",
        "total_vehicle_neighbors",
        "nearest_distance_within_radius",
        "nearest_velocity_within_radius",
        "nearest_type_within_radius",
        "nearest_class_within_radius",
    ]

    for i in range(n_cells):
        fieldnames += [
            f"cell_{i}_count",
            f"cell_{i}_mean_velocity",
            f"cell_{i}_min_distance",
            f"cell_{i}_ped_count",
            f"cell_{i}_vehicle_count",
        ]

    fieldnames += [
        "f1_state",
        "f2_state",
        "f3_state",
        "n_ped_red",
        "n_ped_green",
        "n_ped_off",
    ]

    # 7) save
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(rows_out))
    print("Grid:", GRID_SIZE, "x", GRID_SIZE)
    print("Radius:", RADIUS)


if __name__ == "__main__":
    main()

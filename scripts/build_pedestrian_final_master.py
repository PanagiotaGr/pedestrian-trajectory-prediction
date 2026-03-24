import os
import csv
from collections import defaultdict

NEIGHBOR_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
LIGHT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_crossing_behavior.csv")
MAP_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_map_labels.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_final_master.csv")

RADIUS = 5.0
GRID_SIZE = 5
CELL_SIZE = (2 * RADIUS) / GRID_SIZE


def norm_sid(x):
    try:
        return f"{int(float(x)):04d}"
    except:
        return str(x).zfill(4)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


# ---------------- LOAD LIGHTS ----------------
light_info = {}
with open(LIGHT_PATH) as f:
    for r in csv.DictReader(f):
        sid = norm_sid(r["sample_id"])
        light_info[sid] = r


# ---------------- LOAD MAP ----------------
map_info = {}
with open(MAP_PATH) as f:
    for r in csv.DictReader(f):
        sid = norm_sid(r["sample_id"])
        map_info[sid] = r


# ---------------- LOAD NEIGHBORS ----------------
grouped = defaultdict(list)

with open(NEIGHBOR_PATH) as f:
    for r in csv.DictReader(f):
        sid = norm_sid(r["sample_id"])
        r["sample_id"] = sid
        grouped[sid].append(r)


rows_out = []

for sid, rows in grouped.items():

    first = rows[0]

    tx = safe_float(first["target_x"])
    ty = safe_float(first["target_y"])
    tv = safe_float(first["target_velocity"])

    # ---------------- GRID INIT ----------------
    cells = []
    for _ in range(GRID_SIZE * GRID_SIZE):
        cells.append({
            "count": 0,
            "sum_vel": 0.0,
            "min_dist": None
        })

    total_neighbors = 0

    # ---------------- PROCESS NEIGHBORS ----------------
    for r in rows:
        dist = safe_float(r["distance"], 999)

        if dist <= 0 or dist > RADIUS:
            continue

        ox = safe_float(r["other_x"])
        oy = safe_float(r["other_y"])
        vel = safe_float(r["other_velocity"])

        dx = ox - tx
        dy = oy - ty

        gx = int((dx + RADIUS) / CELL_SIZE)
        gy = int((dy + RADIUS) / CELL_SIZE)

        if gx < 0 or gx >= GRID_SIZE or gy < 0 or gy >= GRID_SIZE:
            continue

        idx = gy * GRID_SIZE + gx
        cell = cells[idx]

        cell["count"] += 1
        cell["sum_vel"] += vel

        if cell["min_dist"] is None or dist < cell["min_dist"]:
            cell["min_dist"] = dist

        total_neighbors += 1

    # ---------------- BASE ROW ----------------
    out = {
        "sample_id": sid,
        "scene_path": first["scene_path"],
        "timestamp": first["timestamp"],
        "target_id": first["target_id"],
        "target_class_name": first["target_class_name"],
        "target_x": round(tx, 4),
        "target_y": round(ty, 4),
        "target_velocity": round(tv, 4),
        "total_neighbors": total_neighbors,
    }

    # ---------------- ADD LIGHTS ----------------
    if sid in light_info:
        out.update(light_info[sid])

    # ---------------- ADD MAP ----------------
    if sid in map_info:
        out.update(map_info[sid])

    # ---------------- ADD GRID ----------------
    for i, c in enumerate(cells):
        mean_vel = c["sum_vel"] / c["count"] if c["count"] > 0 else 0.0
        min_dist = c["min_dist"] if c["min_dist"] else 0.0

        out[f"cell_{i}_count"] = c["count"]
        out[f"cell_{i}_mean_vel"] = round(mean_vel, 4)
        out[f"cell_{i}_min_dist"] = round(min_dist, 4)

    rows_out.append(out)


# ---------------- FIELDNAMES ----------------
fieldnames = list(rows_out[0].keys())

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)


print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))
print("Grid:", f"{GRID_SIZE}x{GRID_SIZE}")

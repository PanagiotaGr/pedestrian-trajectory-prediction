import os
import csv
from collections import defaultdict

INPUT_NEIGHBORS = os.path.expanduser("~/imptc_project/results/pedestrian_moments_neighbors.csv")
INPUT_MAP       = os.path.expanduser("~/imptc_project/results/pedestrian_map_labels.csv")
OUTPUT_PATH     = os.path.expanduser("~/imptc_project/results/pedestrian_grid_5x5_clean.csv")

# Grid: 5x5 cells, each 1x1 meter, pedestrian at center (0,0)
# Covers -2.5m to +2.5m in both x and y
GRID_SIZE = 5
CELL_SIZE = 1.0
HALF      = GRID_SIZE * CELL_SIZE / 2  # 2.5


def assign_cell(dx, dy):
    """Return (row, col) for a neighbor offset (dx, dy) from pedestrian.
    Returns None if outside the 5x5 grid."""
    col = int((dx + HALF) / CELL_SIZE)
    row = int((dy + HALF) / CELL_SIZE)
    if 0 <= col < GRID_SIZE and 0 <= row < GRID_SIZE:
        return row, col
    return None


# ------------------------------------------------------------------
# 1) Load map labels (crosswalk, sidewalk, road, bikelane)
# ------------------------------------------------------------------
map_labels = {}
with open(INPUT_MAP, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        map_labels[row["sample_id"]] = {
            "map_label_name": row.get("map_label_name", ""),
            "is_crosswalk":   row.get("is_crosswalk", "0"),
            "is_sidewalk":    row.get("is_sidewalk",  "0"),
            "is_road":        row.get("is_road",      "0"),
            "is_bikelane":    row.get("is_bikelane",  "0"),
        }

# ------------------------------------------------------------------
# 2) Group neighbors by sample
# ------------------------------------------------------------------
grouped = defaultdict(list)
with open(INPUT_NEIGHBORS, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        grouped[row["sample_id"]].append(row)

# ------------------------------------------------------------------
# 3) Build one row per sample
# ------------------------------------------------------------------
def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


rows_out = []

for sample_id, neighbors in grouped.items():
    first = neighbors[0]

    tx  = safe_float(first["target_x"])
    ty  = safe_float(first["target_y"])
    tv  = safe_float(first["target_velocity"])

    # --- pedestrian info ---
    out = {
        "sample_id":        sample_id,
        "scene_path":       first["scene_path"],
        "timestamp":        first["timestamp"],
        "target_id":        first["target_id"],
        "target_class":     first["target_class_name"],
        "target_x":         round(tx, 4),
        "target_y":         round(ty, 4),
        "target_velocity":  round(tv, 4),
    }

    # --- map location ---
    mlabel = map_labels.get(sample_id, {})
    out["map_label"]    = mlabel.get("map_label_name", "")
    out["is_crosswalk"] = mlabel.get("is_crosswalk",  "0")
    out["is_sidewalk"]  = mlabel.get("is_sidewalk",   "0")
    out["is_road"]      = mlabel.get("is_road",       "0")
    out["is_bikelane"]  = mlabel.get("is_bikelane",   "0")

    # --- init 25 cells ---
    cells = [
        {"count": 0, "sum_vel": 0.0, "min_dist": None,
         "ped_count": 0, "vru_count": 0, "vehicle_count": 0}
        for _ in range(GRID_SIZE * GRID_SIZE)
    ]

    for nb in neighbors:
        ox   = safe_float(nb["other_x"])
        oy   = safe_float(nb["other_y"])
        dist = safe_float(nb["distance"], 9999.0)
        vel  = safe_float(nb["other_velocity"])

        dx = ox - tx
        dy = oy - ty

        cell_pos = assign_cell(dx, dy)
        if cell_pos is None:
            continue  # outside 5x5 window → discard

        row_i, col_i = cell_pos
        idx  = row_i * GRID_SIZE + col_i
        cell = cells[idx]

        cell["count"]   += 1
        cell["sum_vel"] += vel

        if cell["min_dist"] is None or dist < cell["min_dist"]:
            cell["min_dist"] = dist

        cls  = nb.get("other_class_name", "")
        typ  = nb.get("other_type", "")

        if cls == "person":
            cell["ped_count"] += 1
        if typ == "vrus":
            cell["vru_count"] += 1
        if typ == "vehicles":
            cell["vehicle_count"] += 1

    # --- flatten cells into columns ---
    # cell_0 = bottom-left, cell_24 = top-right
    # cell_12 = center (where the pedestrian is)
    for i, cell in enumerate(cells):
        mean_vel = cell["sum_vel"] / cell["count"] if cell["count"] > 0 else 0.0
        min_dist = cell["min_dist"] if cell["min_dist"] is not None else 0.0

        out[f"cell_{i:02d}_count"]         = cell["count"]
        out[f"cell_{i:02d}_min_dist"]      = round(min_dist, 4)
        out[f"cell_{i:02d}_mean_vel"]      = round(mean_vel, 4)
        out[f"cell_{i:02d}_ped_count"]     = cell["ped_count"]
        out[f"cell_{i:02d}_vru_count"]     = cell["vru_count"]
        out[f"cell_{i:02d}_vehicle_count"] = cell["vehicle_count"]

    rows_out.append(out)

# ------------------------------------------------------------------
# 4) Write CSV
# ------------------------------------------------------------------
base_fields = [
    "sample_id", "scene_path", "timestamp",
    "target_id", "target_class", "target_x", "target_y", "target_velocity",
    "map_label", "is_crosswalk", "is_sidewalk", "is_road", "is_bikelane",
]
cell_fields = []
for i in range(GRID_SIZE * GRID_SIZE):
    cell_fields += [
        f"cell_{i:02d}_count",
        f"cell_{i:02d}_min_dist",
        f"cell_{i:02d}_mean_vel",
        f"cell_{i:02d}_ped_count",
        f"cell_{i:02d}_vru_count",
        f"cell_{i:02d}_vehicle_count",
    ]

fieldnames = base_fields + cell_fields

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Saved: {OUTPUT_PATH}")
print(f"Samples: {len(rows_out)}")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, cell size: {CELL_SIZE}m")
print(f"Coverage: ±{HALF}m around pedestrian")
print(f"Pedestrian at center cell: cell_12")
print(f"Total feature columns: {len(cell_fields)} ({GRID_SIZE*GRID_SIZE} cells × 6 features)")

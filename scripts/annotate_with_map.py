import csv
import numpy as np
import open3d as o3d
import os

# ─────────────────────────────────────
# Label mapping (από το dataset σου)
# ─────────────────────────────────────
LABEL_MAP = {
    (128,64,128): "road",
    (244,35,232): "sidewalk",
    (81,0,81): "ground",
    (150,100,100): "curb",
    (157,234,50): "road_line",
    (229,165,10): "crosswalk",
    (98,160,234): "bikelane",
    (128,128,128): "unknown",
}

# ─────────────────────────────────────
# Load PLY
# ─────────────────────────────────────
ply_path = os.path.expanduser(
    "~/imptc_project/ground_plane/xung_ground_plane_02.ply"
)

print("Loading PLY...")
pcd = o3d.io.read_point_cloud(ply_path)

points = np.asarray(pcd.points)
colors = (np.asarray(pcd.colors) * 255).astype(int)

print("Points:", len(points))

# KD-tree
kdtree = o3d.geometry.KDTreeFlann(pcd)

# ─────────────────────────────────────
# Query function
# ─────────────────────────────────────
def get_label(x, y):
    query = np.array([x, y, 0.0])
    _, idx, _ = kdtree.search_knn_vector_3d(query, 1)
    i = idx[0]

    rgb = tuple(colors[i])
    return LABEL_MAP.get(rgb, "unknown")

# ─────────────────────────────────────
# Load dataset
# ─────────────────────────────────────
input_csv = os.path.expanduser(
    "~/imptc_project/results/pedestrian_math_dataset_v2.csv"
)

output_csv = os.path.expanduser(
    "~/imptc_project/results/pedestrian_math_with_map.csv"
)

rows = []
with open(input_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print("Rows:", len(rows))

# ─────────────────────────────────────
# Annotate
# ─────────────────────────────────────
for i, r in enumerate(rows):
    ax = float(r["ax"])
    ay = float(r["ay"])
    bx = float(r["bx"])
    by = float(r["by"])

    r["label_a"] = get_label(ax, ay)
    r["label_b"] = get_label(bx, by)

    if i % 1000 == 0:
        print("Processed:", i)

# ─────────────────────────────────────
# Save
# ─────────────────────────────────────
fieldnames = list(rows[0].keys())

with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nDONE")
print("Saved:", output_csv)

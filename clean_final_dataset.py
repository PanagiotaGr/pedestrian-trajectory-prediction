import pandas as pd

# load
df = pd.read_csv("results/pedestrian_final_master.csv")

# =========================
# 1. DROP USELESS COLUMNS
# =========================
drop_cols = [
    "sample_id",
    "scene_path",
    "timestamp",
    "target_id",
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# =========================
# 2. REORDER COLUMNS
# =========================

ego_cols = [
    "target_velocity",
    "displacement",
    "avg_speed",
]

traffic_cols = [
    "f1_state", "f2_state", "f3_state",
    "n_ped_green", "n_ped_red",
    "has_green", "red_only",
]

neighbor_cols = [
    "total_neighbors",
    "nearest_dist",
    "n_neighbors",
]

map_cols = [
    "map_label_name",
    "is_crosswalk",
    "is_sidewalk",
    "is_road",
    "is_bikelane",
    "nearest_map_dist",
]

# grid columns
grid_cols = [c for c in df.columns if "cell_" in c]

# final order
final_cols = (
    ego_cols +
    traffic_cols +
    neighbor_cols +
    map_cols +
    grid_cols
)

# keep only existing
final_cols = [c for c in final_cols if c in df.columns]

df = df[final_cols]

# =========================
# 3. CLEAN VALUES
# =========================

# fill NaNs
df = df.fillna(0)

# =========================
# 4. SAVE
# =========================

df.to_csv("results/dataset_clean_ml_ready.csv", index=False)

print("Saved clean dataset!")
print("Shape:", df.shape)

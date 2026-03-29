import pandas as pd

df = pd.read_csv("results/pedestrian_final_master.csv")

# =========================
# EXACT ORDER (όπως το θες)
# =========================

cols = [
"target_velocity","displacement","avg_speed",
"f1_state","f2_state","f3_state",
"n_ped_green","n_ped_red","has_green","red_only",
"total_neighbors","nearest_dist","n_neighbors",
"map_label_name","is_crosswalk","is_sidewalk","is_road","is_bikelane","nearest_map_dist",
]

# grid
for i in range(25):
    cols += [
        f"cell_{i}_count",
        f"cell_{i}_mean_vel",
        f"cell_{i}_min_dist"
    ]

# =========================
# KEEP ONLY THESE
# =========================

cols = [c for c in cols if c in df.columns]
df = df[cols]

# =========================
# CLEAN VALUES
# =========================

df = df.fillna(0)

# =========================
# SAVE
# =========================

df.to_csv("results/dataset_final_clean.csv", index=False)

print("DONE")
print("Shape:", df.shape)

import pandas as pd
import numpy as np

# =========================
# PARAMETERS
# =========================
INPUT_LEN = 32   # 3.2 sec
OUTPUT_LEN = 48  # 4.8 sec

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset_final.csv")

# =========================
# SORT (VERY IMPORTANT)
# =========================
df = df.sort_values(["track_id", "frame"])

# =========================
# SELECT FEATURES (IMPORTANT)
# =========================
FEATURES = [
    "x", "y",
    "vx", "vy",
    "num_neighbors",
    "traffic_light",
    "dist_crosswalk"
]

# κράτα μόνο columns που υπάρχουν
FEATURES = [f for f in FEATURES if f in df.columns]

print("Using features:", FEATURES)

# =========================
# CREATE SEQUENCES
# =========================
X_list = []
Y_list = []

for track_id, group in df.groupby("track_id"):

    group = group.reset_index(drop=True)

    # skip μικρά tracks
    if len(group) < INPUT_LEN + OUTPUT_LEN:
        continue

    for i in range(len(group) - INPUT_LEN - OUTPUT_LEN):

        past = group.iloc[i:i+INPUT_LEN]
        future = group.iloc[i+INPUT_LEN:i+INPUT_LEN+OUTPUT_LEN]

        # ===== INPUT =====
        X = past[FEATURES].values

        # ===== TARGET (future positions) =====
        Y = future[["x", "y"]].values

        # skip αν έχει NaN
        if np.isnan(X).any() or np.isnan(Y).any():
            continue

        X_list.append(X)
        Y_list.append(Y)

# =========================
# TO NUMPY
# =========================
X = np.array(X_list)
Y = np.array(Y_list)

print("\nDataset created!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# =========================
# SAVE
# =========================
np.save("X.npy", X)
np.save("Y.npy", Y)

print("\nSaved:")
print("X.npy")
print("Y.npy")

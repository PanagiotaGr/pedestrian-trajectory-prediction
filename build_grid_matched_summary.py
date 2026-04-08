"""
build_grid_matched_summary.py
=============================
Aggregates grid_matched_interactions.csv into one row per (A, ts)

Only uses MATCHED neighbors (real B)

Output:
    results/grid_matched_summary.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))

INPUT = RESULTS_DIR / "grid_matched_interactions.csv"
OUTPUT = RESULTS_DIR / "grid_matched_summary.csv"


# ─────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────
def aggregate_group(grp: pd.DataFrame) -> dict:

    # keep only matched
    g = grp[grp["matched_to_pairwise"] == 1]

    n = len(g)

    if n == 0:
        return {
            "total_matched_neighbors": 0,
            "closest_neighbor_distance": np.nan,
            "mean_neighbor_distance": np.nan,
            "max_closing_speed": np.nan,
            "approaching_neighbors": 0,
            "neighbors_on_crosswalk": 0,
            "neighbors_on_sidewalk": 0,
            "neighbors_on_road": 0,
            "front_neighbors": 0,
            "back_neighbors": 0,
            "left_neighbors": 0,
            "right_neighbors": 0,
            "same_direction_neighbors": 0,
            "opposite_direction_neighbors": 0,
            "crossing_neighbors": 0,
            "most_dangerous_distance": np.nan,
            "most_dangerous_closing_speed": np.nan,
        }

    # ── distances
    dists = g["dist_xy"].values

    closest_dist = float(np.min(dists))
    mean_dist = float(np.mean(dists))

    # ── motion
    closing = g["closing_speed"].values

    approaching = int((closing > 0.1).sum())
    max_closing = float(np.max(closing))

    # ── most dangerous = max (closing / distance)
    risk_scores = []
    for i in range(n):
        d = dists[i] if dists[i] > 1e-6 else 1e-6
        risk_scores.append(closing[i] / d)

    idx = int(np.argmax(risk_scores))
    most_dangerous_dist = float(dists[idx])
    most_dangerous_cs = float(closing[idx])

    # ── region counts
    regions = g["region"].value_counts()

    def rc(r): return int(regions.get(r, 0))

    front = rc("front") + rc("front_left") + rc("front_right")
    back  = rc("back")  + rc("back_left")  + rc("back_right")
    left  = rc("left")  + rc("front_left") + rc("back_left")
    right = rc("right") + rc("front_right") + rc("back_right")

    # ── heading
    h = g["heading_relation"].value_counts()
    same = int(h.get("same_direction", 0))
    opp  = int(h.get("opposite_direction", 0))
    cross= int(h.get("crossing", 0))

    # ── ground
    ground = g["ground_type_name_cell"].value_counts()

    on_crosswalk = int(ground.get("crosswalk", 0))
    on_sidewalk  = int(ground.get("sidewalk", 0))
    on_road      = int(ground.get("road", 0))

    return {
        "total_matched_neighbors": n,
        "closest_neighbor_distance": closest_dist,
        "mean_neighbor_distance": mean_dist,
        "max_closing_speed": max_closing,
        "approaching_neighbors": approaching,
        "neighbors_on_crosswalk": on_crosswalk,
        "neighbors_on_sidewalk": on_sidewalk,
        "neighbors_on_road": on_road,
        "front_neighbors": front,
        "back_neighbors": back,
        "left_neighbors": left,
        "right_neighbors": right,
        "same_direction_neighbors": same,
        "opposite_direction_neighbors": opp,
        "crossing_neighbors": cross,
        "most_dangerous_distance": most_dangerous_dist,
        "most_dangerous_closing_speed": most_dangerous_cs,
    }


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
print("=" * 55)
print("  BUILD GRID MATCHED SUMMARY")
print("=" * 55)

print(f"\n[→] Loading {INPUT.name} ...")
df = pd.read_csv(INPUT)

print(f"  Rows: {len(df):,}")

groups = df.groupby(["track_id_A", "ts"], sort=False)

records = []

for i, ((tid, ts), grp) in enumerate(groups):
    if i % 20000 == 0:
        print(f"  Group {i:,}...", end="\r")

    base = grp.iloc[0]

    agg = aggregate_group(grp)

    rec = {
        "track_id_A": tid,
        "ts": ts,
        "real_track_id_A": base["real_track_id_A"],
        "ground_type_name_A": base["ground_type_name_A"],
        "on_crosswalk_A": base["on_crosswalk_A"],
        "any_green_light": base["any_green_light"],
        "any_red_light": base["any_red_light"],
    }

    rec.update(agg)
    records.append(rec)

out = pd.DataFrame(records)

print(f"\n  Output rows: {len(out):,}")

out.to_csv(OUTPUT, index=False)

print(f"[OK] Saved: {OUTPUT}")

# ─────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("  VALIDATION")
print("="*55)

print("Avg matched neighbors:", out["total_matched_neighbors"].mean())
print("Max neighbors:", out["total_matched_neighbors"].max())
print("% with crosswalk neighbors:",
      (out["neighbors_on_crosswalk"] > 0).mean()*100)

print("% with approaching:",
      (out["approaching_neighbors"] > 0).mean()*100)

print("\n[DONE]")

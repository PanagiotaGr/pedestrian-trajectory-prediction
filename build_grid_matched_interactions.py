"""
build_grid_matched_interactions.py
==================================
Builds a detailed A-B interaction dataset restricted to neighbors that
actually appear inside A's 5x5 local grid.

For each pedestrian A and timestamp ts:
- read the 5x5 grid from grid_dataset_final.json
- find all cells containing VRUs
- match each occupied VRU cell to a real B from dataset_relative_geometry.csv
  using nearest (rel_x, rel_y) matching at the same (track_id_A, ts)
- enrich with:
    * A info
    * B info
    * pairwise geometry / velocity / heading / closing speed
    * grid cell position
    * ground type of the occupied cell
    * A-side semantic context (sidewalk/crosswalk/traffic lights)

Inputs
------
results/grid_dataset_final.json
results/dataset_relative_geometry.csv
results/matched_codes.csv
results/pedestrian_context.csv

Output
------
results/grid_matched_interactions.csv

Usage
-----
source venv/bin/activate
python build_grid_matched_interactions.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))

GRID_JSON = RESULTS_DIR / "grid_dataset_final.json"
PAIRWISE_CSV = RESULTS_DIR / "dataset_relative_geometry.csv"
MATCHED_CODES_CSV = RESULTS_DIR / "matched_codes.csv"
PEDESTRIAN_CONTEXT_CSV = RESULTS_DIR / "pedestrian_context.csv"

OUTPUT_CSV = RESULTS_DIR / "grid_matched_interactions.csv"

GRID_SIZE = 5
CENTER = 2
MATCH_POS_TOL = 1.25  # meters, due to grid quantization / approximation


GROUND_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "ground",
    3: "curb",
    4: "road_line",
    5: "crosswalk",
    6: "bikelane",
    7: "unknown",
}

LIGHT_STATES = {
    4: "green",
    10: "red",
    20: "yellow",
    30: "red_yellow",
    2: "yellow_blinking",
    11: "disabled",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def safe_numeric(df: pd.DataFrame, cols: List[str], fill: float = 0.0) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill)


def parse_real_track_id_from_member_path(member_path: str) -> str:
    """
    Example:
        0000_20230322_081506/vrus/000/track.json
    ->
        0000_20230322_081506/000
    """
    if not isinstance(member_path, str) or not member_path:
        return ""
    parts = member_path.strip("/").split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[2]}"
    return ""


def classify_region(rel_x: float, rel_y: float) -> str:
    ax, ay = abs(rel_x), abs(rel_y)
    if ax < 1e-6 and ay < 1e-6:
        return "center"
    if ax > ay * 1.5:
        return "front" if rel_x > 0 else "back"
    elif ay > ax * 1.5:
        return "left" if rel_y > 0 else "right"
    else:
        if rel_x >= 0 and rel_y >= 0:
            return "front_left"
        if rel_x >= 0 and rel_y < 0:
            return "front_right"
        if rel_x < 0 and rel_y >= 0:
            return "back_left"
        return "back_right"


def local_cell_center(row: int, col: int) -> Tuple[float, float]:
    """
    Local grid convention:
    center cell = (2,2)
    col increasing -> +x
    row decreasing -> +y
    """
    rel_x = float(col - CENTER)
    rel_y = float(CENTER - row)
    return rel_x, rel_y


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def heading_relation(heading_diff: float) -> str:
    a = abs(float(heading_diff))
    if a < math.pi / 6:
        return "same_direction"
    if a > 5 * math.pi / 6:
        return "opposite_direction"
    return "crossing"


# ──────────────────────────────────────────────────────────────────────────────
# LOAD SOURCES
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  BUILD GRID-MATCHED INTERACTION DATASET")
print("=" * 60)

print(f"\n[→] Loading {PAIRWISE_CSV.name} ...")
pairwise = pd.read_csv(PAIRWISE_CSV, low_memory=False)
print(f"  Pairwise rows: {len(pairwise):,}")

safe_numeric(
    pairwise,
    [
        "sample_id", "ts", "ax", "ay", "bx", "by",
        "avx", "avy", "bvx", "bvy",
        "speed_a", "speed_b",
        "rel_x", "rel_y", "rel_vx", "rel_vy",
        "dist_xy", "theta_global", "theta_local",
        "heading_a", "heading_b", "heading_diff", "closing_speed",
    ],
)

print(f"[→] Loading {MATCHED_CODES_CSV.name} ...")
matched_codes = pd.read_csv(MATCHED_CODES_CSV, low_memory=False)
matched_codes["sample_id"] = pd.to_numeric(matched_codes["sample_id"], errors="coerce")
matched_codes["real_track_id_A"] = matched_codes["member_path"].apply(parse_real_track_id_from_member_path)

pairwise["sample_id"] = pd.to_numeric(pairwise["sample_id"], errors="coerce")
pairwise = pairwise.merge(
    matched_codes[["sample_id", "real_track_id_A"]].drop_duplicates(),
    on="sample_id",
    how="left"
)

print(f"[→] Loading {PEDESTRIAN_CONTEXT_CSV.name} ...")
ped_ctx = pd.read_csv(PEDESTRIAN_CONTEXT_CSV, low_memory=False)

safe_numeric(
    ped_ctx,
    [
        "ts", "ground_type_id", "dist_to_crosswalk", "on_crosswalk",
        "f1", "f2", "f3", "n_neighbors", "has_close_neighbor", "closest_dist",
    ],
)

ped_ctx_cols = [
    "track_id", "ts",
    "ground_type_id", "ground_type_name",
    "dist_to_crosswalk", "on_crosswalk",
    "f1", "f2", "f3",
    "f1_state", "f2_state", "f3_state",
    "n_neighbors", "has_close_neighbor", "closest_dist",
]
ped_ctx = ped_ctx[ped_ctx_cols].copy()

print(f"[→] Loading {GRID_JSON.name} ...")
with open(GRID_JSON, "r", encoding="utf-8") as f:
    grid_data = json.load(f)
print(f"  Grid tracks: {len(grid_data):,}")


# ──────────────────────────────────────────────────────────────────────────────
# BUILD FAST LOOKUPS
# ──────────────────────────────────────────────────────────────────────────────
# pairwise lookup by (real_track_id_A, ts)
pairwise_groups: Dict[Tuple[str, int], pd.DataFrame] = {}
for key, grp in pairwise.groupby(["real_track_id_A", "ts"], sort=False):
    pairwise_groups[key] = grp.copy()

# pedestrian context lookup by (track_id, ts)
ctx_lookup: Dict[Tuple[str, int], dict] = {}
for _, row in ped_ctx.iterrows():
    ctx_lookup[(str(row["track_id"]), int(row["ts"]))] = row.to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN MATCHING
# ──────────────────────────────────────────────────────────────────────────────
rows_out: List[dict] = []

n_tracks = 0
n_timestamps = 0
n_vru_cells = 0
n_matches = 0
n_unmatched = 0

for tr_i, track in enumerate(grid_data, start=1):
    real_track_id_A = str(track["track_id"])
    n_tracks += 1

    # try to recover short id if wanted later; not required
    if tr_i % 100 == 0:
        print(f"  Track {tr_i:,}/{len(grid_data):,} ...", end="\r")

    for ts_data in track["timesteps"]:
        n_timestamps += 1
        ts = int(ts_data["ts"])

        # A-side semantic context
        ctx = ctx_lookup.get((real_track_id_A, ts), {})
        ground_type_id_A = ctx.get("ground_type_id", np.nan)
        ground_type_name_A = ctx.get("ground_type_name", None)
        dist_to_crosswalk_A = ctx.get("dist_to_crosswalk", np.nan)
        on_crosswalk_A = ctx.get("on_crosswalk", np.nan)

        f1 = ctx.get("f1", ts_data.get("traffic_lights", {}).get("f1", np.nan))
        f2 = ctx.get("f2", ts_data.get("traffic_lights", {}).get("f2", np.nan))
        f3 = ctx.get("f3", ts_data.get("traffic_lights", {}).get("f3", np.nan))
        f1_state = ctx.get("f1_state", LIGHT_STATES.get(int(f1), "unknown") if pd.notna(f1) else None)
        f2_state = ctx.get("f2_state", LIGHT_STATES.get(int(f2), "unknown") if pd.notna(f2) else None)
        f3_state = ctx.get("f3_state", LIGHT_STATES.get(int(f3), "unknown") if pd.notna(f3) else None)

        pair_grp = pairwise_groups.get((real_track_id_A, ts), None)

        # some A info may be present in pairwise, else fallback unavailable
        if pair_grp is not None and len(pair_grp) > 0:
            rowA0 = pair_grp.iloc[0]
            sample_id = rowA0["sample_id"]
            split = rowA0.get("split", None)
            track_id_A_short = rowA0.get("track_id_A", None)
            class_A = rowA0.get("class_A", None)
            ax = rowA0.get("ax", np.nan)
            ay = rowA0.get("ay", np.nan)
            avx = rowA0.get("avx", np.nan)
            avy = rowA0.get("avy", np.nan)
            speed_a = rowA0.get("speed_a", np.nan)
            heading_a = rowA0.get("heading_a", np.nan)
        else:
            sample_id = np.nan
            split = None
            track_id_A_short = None
            class_A = None
            ax = ay = avx = avy = speed_a = heading_a = np.nan

        grid = ts_data["grid"]

        # explicit 1m neighborhood around A from cell ground
        front_1m_ground = GROUND_NAMES.get(int(grid["ground"][1][2]), "unknown")
        back_1m_ground = GROUND_NAMES.get(int(grid["ground"][3][2]), "unknown")
        left_1m_ground = GROUND_NAMES.get(int(grid["ground"][2][1]), "unknown")
        right_1m_ground = GROUND_NAMES.get(int(grid["ground"][2][3]), "unknown")

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if row == CENTER and col == CENTER:
                    continue

                has_vru = int(grid["vrus"][row][col])
                if has_vru != 1:
                    continue

                n_vru_cells += 1

                cell_rel_x, cell_rel_y = local_cell_center(row, col)
                cell_distance = math.sqrt(cell_rel_x**2 + cell_rel_y**2)

                # actual stored local rel motion / position in the grid
                grid_rel_x = float(grid["rel_x"][row][col])
                grid_rel_y = float(grid["rel_y"][row][col])
                grid_rel_vx = float(grid["rel_vx"][row][col])
                grid_rel_vy = float(grid["rel_vy"][row][col])

                ground_type_id_cell = int(grid["ground"][row][col])
                ground_type_name_cell = GROUND_NAMES.get(ground_type_id_cell, "unknown")
                has_vehicle_cell = int(grid["vehicles"][row][col])
                region = classify_region(grid_rel_x if abs(grid_rel_x) > 1e-9 else cell_rel_x,
                                         grid_rel_y if abs(grid_rel_y) > 1e-9 else cell_rel_y)

                matched_row = None
                match_error = np.nan

                if pair_grp is not None and len(pair_grp) > 0:
                    # nearest match in pairwise candidates at same A, same ts
                    cand = pair_grp.copy()
                    cand["match_err"] = np.sqrt(
                        (cand["rel_x"] - grid_rel_x) ** 2 +
                        (cand["rel_y"] - grid_rel_y) ** 2
                    )
                    cand = cand.sort_values("match_err")
                    best = cand.iloc[0]
                    if float(best["match_err"]) <= MATCH_POS_TOL:
                        matched_row = best
                        match_error = float(best["match_err"])

                if matched_row is None:
                    n_unmatched += 1
                    rows_out.append({
                        # A identity
                        "sample_id": sample_id,
                        "split": split,
                        "track_id_A": track_id_A_short,
                        "real_track_id_A": real_track_id_A,
                        "class_A": class_A,
                        "ts": ts,

                        # A motion / position
                        "ax": ax,
                        "ay": ay,
                        "avx": avx,
                        "avy": avy,
                        "speed_a": speed_a,
                        "heading_a": heading_a,

                        # grid-cell info
                        "grid_row": row,
                        "grid_col": col,
                        "cell_rel_x": cell_rel_x,
                        "cell_rel_y": cell_rel_y,
                        "cell_distance": cell_distance,
                        "grid_rel_x": grid_rel_x,
                        "grid_rel_y": grid_rel_y,
                        "grid_rel_vx": grid_rel_vx,
                        "grid_rel_vy": grid_rel_vy,
                        "region": region,
                        "has_vehicle_cell": has_vehicle_cell,
                        "ground_type_id_cell": ground_type_id_cell,
                        "ground_type_name_cell": ground_type_name_cell,
                        "cell_on_crosswalk": int(ground_type_id_cell == 5),

                        # B not matched
                        "track_id_B": None,
                        "class_B": None,
                        "bx": np.nan,
                        "by": np.nan,
                        "bvx": np.nan,
                        "bvy": np.nan,
                        "speed_b": np.nan,
                        "heading_b": np.nan,

                        # pairwise / interaction not matched
                        "rel_x": np.nan,
                        "rel_y": np.nan,
                        "rel_vx": np.nan,
                        "rel_vy": np.nan,
                        "dist_xy": np.nan,
                        "theta_global": np.nan,
                        "theta_local": np.nan,
                        "heading_diff": np.nan,
                        "closing_speed": np.nan,
                        "heading_relation": None,

                        # semantic A context
                        "ground_type_id_A": ground_type_id_A,
                        "ground_type_name_A": ground_type_name_A,
                        "dist_to_crosswalk_A": dist_to_crosswalk_A,
                        "on_crosswalk_A": on_crosswalk_A,

                        # 1m neighborhood around A
                        "front_1m_ground": front_1m_ground,
                        "back_1m_ground": back_1m_ground,
                        "left_1m_ground": left_1m_ground,
                        "right_1m_ground": right_1m_ground,

                        # lights
                        "f1": f1,
                        "f2": f2,
                        "f3": f3,
                        "f1_state": f1_state,
                        "f2_state": f2_state,
                        "f3_state": f3_state,
                        "any_green_light": int(
                            (f1_state == "green") or
                            (f2_state == "green") or
                            (f3_state == "green")
                        ),
                        "any_red_light": int(
                            (f1_state == "red") or
                            (f2_state == "red") or
                            (f3_state == "red")
                        ),

                        # matching metadata
                        "matched_to_pairwise": 0,
                        "match_error_xy": match_error,
                    })
                    continue

                n_matches += 1
                rows_out.append({
                    # A identity
                    "sample_id": matched_row["sample_id"],
                    "split": matched_row["split"],
                    "track_id_A": matched_row["track_id_A"],
                    "real_track_id_A": real_track_id_A,
                    "class_A": matched_row["class_A"],
                    "ts": ts,

                    # A position / motion
                    "ax": matched_row["ax"],
                    "ay": matched_row["ay"],
                    "avx": matched_row["avx"],
                    "avy": matched_row["avy"],
                    "speed_a": matched_row["speed_a"],
                    "heading_a": matched_row["heading_a"],

                    # B identity / motion
                    "track_id_B": matched_row["track_id_B"],
                    "class_B": matched_row["class_B"],
                    "bx": matched_row["bx"],
                    "by": matched_row["by"],
                    "bvx": matched_row["bvx"],
                    "bvy": matched_row["bvy"],
                    "speed_b": matched_row["speed_b"],
                    "heading_b": matched_row["heading_b"],

                    # pairwise interaction
                    "rel_x": matched_row["rel_x"],
                    "rel_y": matched_row["rel_y"],
                    "rel_vx": matched_row["rel_vx"],
                    "rel_vy": matched_row["rel_vy"],
                    "dist_xy": matched_row["dist_xy"],
                    "theta_global": matched_row["theta_global"],
                    "theta_local": matched_row["theta_local"],
                    "heading_diff": matched_row["heading_diff"],
                    "closing_speed": matched_row["closing_speed"],
                    "heading_relation": heading_relation(matched_row["heading_diff"]),

                    # grid-cell info
                    "grid_row": row,
                    "grid_col": col,
                    "cell_rel_x": cell_rel_x,
                    "cell_rel_y": cell_rel_y,
                    "cell_distance": cell_distance,
                    "grid_rel_x": grid_rel_x,
                    "grid_rel_y": grid_rel_y,
                    "grid_rel_vx": grid_rel_vx,
                    "grid_rel_vy": grid_rel_vy,
                    "region": region,
                    "has_vehicle_cell": has_vehicle_cell,
                    "ground_type_id_cell": ground_type_id_cell,
                    "ground_type_name_cell": ground_type_name_cell,
                    "cell_on_crosswalk": int(ground_type_id_cell == 5),

                    # A semantic context
                    "ground_type_id_A": ground_type_id_A,
                    "ground_type_name_A": ground_type_name_A,
                    "dist_to_crosswalk_A": dist_to_crosswalk_A,
                    "on_crosswalk_A": on_crosswalk_A,

                    # 1m neighborhood around A
                    "front_1m_ground": front_1m_ground,
                    "back_1m_ground": back_1m_ground,
                    "left_1m_ground": left_1m_ground,
                    "right_1m_ground": right_1m_ground,

                    # lights
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "f1_state": f1_state,
                    "f2_state": f2_state,
                    "f3_state": f3_state,
                    "any_green_light": int(
                        (f1_state == "green") or
                        (f2_state == "green") or
                        (f3_state == "green")
                    ),
                    "any_red_light": int(
                        (f1_state == "red") or
                        (f2_state == "red") or
                        (f3_state == "red")
                    ),

                    # matching metadata
                    "matched_to_pairwise": 1,
                    "match_error_xy": match_error,
                })


# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(rows_out)
out_df.to_csv(OUTPUT_CSV, index=False)

# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  VALIDATION")
print("=" * 60)
print(f"  Tracks processed:             {n_tracks:,}")
print(f"  Timestamps processed:         {n_timestamps:,}")
print(f"  VRU cells found in 5x5:       {n_vru_cells:,}")
print(f"  Matched A-B rows:             {n_matches:,}")
print(f"  Unmatched VRU cells:          {n_unmatched:,}")
if n_vru_cells > 0:
    print(f"  Match rate:                   {100.0 * n_matches / n_vru_cells:.1f}%")

if len(out_df) > 0:
    print(f"  Output rows written:          {len(out_df):,}")
    if "on_crosswalk_A" in out_df.columns:
        print(f"  % A on crosswalk:             {100.0 * (out_df['on_crosswalk_A'] == 1).mean():.1f}%")
    print(f"  % matched rows:               {100.0 * out_df['matched_to_pairwise'].mean():.1f}%")
    matched_only = out_df[out_df["matched_to_pairwise"] == 1]
    if len(matched_only) > 0:
        print(f"  Mean match error (m):         {matched_only['match_error_xy'].mean():.3f}")
        print(f"  Mean B distance (m):          {matched_only['dist_xy'].mean():.3f}")
        print(f"  Mean closing speed:           {matched_only['closing_speed'].mean():.3f}")

print(f"\n[OK] Saved: {OUTPUT_CSV}")

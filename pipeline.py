"""
pipeline.py — Clean Dataset Pipeline
======================================

ASSUMPTIONS & DESIGN DECISIONS:
---------------------------------
1. Single source of truth: grid_dataset_final.json
   - Contains: scene, track_id, class, ts, x, y, θ, grid (5x5), traffic lights
   - No external CSVs needed

2. Coordinate systems:
   GLOBAL frame: x=East, y=North (dataset world coordinates)
   LOCAL frame A: origin at pedestrian A, x=heading direction, y=left

   Rotation global→local:
     R_θ = [[cos θ,  sin θ],
             [-sin θ, cos θ]]

   Inverse (local→global):
     R_θ^T = [[cos θ, -sin θ],
               [sin θ,  cos θ]]

   IMPORTANT: rel_x, rel_y in grid are in LOCAL frame of A.
   To reconstruct absolute position of neighbor B:
     p_B = p_A + R_θA^T · [rel_x, rel_y]

3. Neighbor definition:
   - Neighbors come from the 5x5 grid (vrus layer)
   - Grid is APPROXIMATE: each cell covers 1m x 1m
   - rel_x/y = center of the occupied cell in local frame
   - NOT exact positions — there is up to 0.5m cell quantization error
   - After inverse rotation → approximate global position

4. Matching strategy:
   - Build exact agent positions from grid_dataset_final.json
   - For each neighbor cell, reconstruct approximate global pos
   - Match to nearest exact agent within threshold
   - Threshold must account for: cell quantization (0.5m) + float error
   - Use threshold = 1.5m

5. What was WRONG before:
   a) nb_abs = p_A + nb_rel  ← WRONG (missing inverse rotation)
   b) frame_idx instead of ts ← temporal mismatch
   c) Multiple CSV sources with different scene/track_id formats ← inconsistent
   d) Threshold too large (1.5m) without accounting for rotation error

OUTPUT SCHEMA (final_dataset.csv):
  track_id       : full track identifier (scene/local_id)
  scene          : scene identifier
  class          : VRU class (person, bicycle, etc.)
  ts             : UTC timestamp (microseconds)
  x, y           : global position of A (meters)
  speed          : scalar speed (m/s)
  heading_rad    : heading angle θ_A (radians, 0=East)
  ground_type_id : ground type ID (0-7)
  ground_type    : ground type name
  on_crosswalk   : 1 if pedestrian is on crosswalk
  f1/f2/f3       : traffic light codes
  f1/f2/f3_state : traffic light names (green/red/...)
  n_neighbors    : number of VRU neighbors in 5x5 grid
  has_close_nb   : 1 if any neighbor within 5m
  closest_dist   : distance to closest neighbor (m)
  nbN_rel_x      : neighbor N rel_x in LOCAL frame of A
  nbN_rel_y      : neighbor N rel_y in LOCAL frame of A
  nbN_abs_x      : neighbor N reconstructed GLOBAL x
  nbN_abs_y      : neighbor N reconstructed GLOBAL y
  nbN_dist       : distance A→neighbor (m)
  nbN_class      : matched class label ("unknown" if no match)
  nbN_match_err  : distance between reconstructed and matched position (m)
"""

import os
import json
import csv
import math
import numpy as np
from collections import defaultdict
from pathlib import Path

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[!] scipy not found. pip install scipy")
    exit(1)

# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════
RESULTS_DIR     = Path(os.path.expanduser("~/imptc_project/results"))
INPUT_JSON      = RESULTS_DIR / "grid_dataset_final.json"
OUTPUT_CSV      = RESULTS_DIR / "final_dataset.csv"
VALIDATION_CSV  = RESULTS_DIR / "final_dataset_validation.csv"

GRID_SIZE        = 5
CELL_SIZE        = 1.0        # meters per cell
MAX_NEIGHBORS    = 3
# Matching threshold accounts for:
#   - cell quantization: up to sqrt(2)*0.5 ≈ 0.71m
#   - floating point precision
#   - slight position drift between frames
MATCH_THRESHOLD  = 1.5        # meters
CLOSE_DIST       = 5.0        # meters — "close neighbor" threshold

GROUND_NAMES = {
    0: "road", 1: "sidewalk", 2: "ground", 3: "curb",
    4: "road_line", 5: "crosswalk", 6: "bikelane", 7: "unknown"
}
LIGHT_STATES = {
    4: "green", 10: "red", 20: "yellow",
    30: "red_yellow", 2: "yellow_blinking", 11: "disabled"
}


# ════════════════════════════════════════════════════════════
# COORDINATE TRANSFORMS
# ════════════════════════════════════════════════════════════
def local_to_global_offset(rel_x, rel_y, theta):
    """
    Convert local frame vector → global frame vector.

    Local frame: x = heading direction, y = left of A
    Global frame: x = East, y = North

    R_θ^T · [rel_x, rel_y] = [cos θ · rel_x - sin θ · rel_y,
                                sin θ · rel_x + cos θ · rel_y]

    Args:
        rel_x, rel_y : coordinates in LOCAL frame of A
        theta        : heading angle of A in global frame (radians)

    Returns:
        (dx, dy) : offset in GLOBAL frame
    """
    c, s = math.cos(theta), math.sin(theta)
    dx = c * rel_x - s * rel_y
    dy = s * rel_x + c * rel_y
    return dx, dy


def reconstruct_global(px, py, theta, rel_x, rel_y):
    """
    Reconstruct global position of neighbor B from:
        - A's global position (px, py)
        - A's heading (theta)
        - B's local frame coordinates (rel_x, rel_y)

    Formula: p_B = p_A + R_θA^T · [rel_x, rel_y]
    """
    dx, dy = local_to_global_offset(rel_x, rel_y, theta)
    return px + dx, py + dy


# ════════════════════════════════════════════════════════════
# STAGE 1: Load JSON + Build Agent Index
# ════════════════════════════════════════════════════════════
def load_and_build_agent_index(json_path):
    """
    Load grid_dataset_final.json and build exact agent position index.

    Agent index: (scene, ts) → KDTree over exact agent positions

    NOTE: This uses EXACT positions (ax_global, ay_global) from the JSON,
    which is the ground truth for matching.
    """
    print(f"[STAGE 1] Loading {json_path.name}...")
    with open(json_path) as f:
        data = json.load(f)

    n_tracks = len(data)
    n_frames = sum(len(t["timesteps"]) for t in data)
    print(f"  tracks : {n_tracks:,}")
    print(f"  frames : {n_frames:,}")

    # Build raw index
    raw = defaultdict(list)   # (scene, ts) → list of agents
    for track in data:
        scene    = track["scene"]      # e.g. "0000_20230322_081506"
        track_id = track["track_id"]   # e.g. "0000_20230322_081506/000"
        cls      = track["class"]      # e.g. "person"

        for ts_data in track["timesteps"]:
            ts = int(ts_data["ts"])    # exact UTC timestamp (μs)
            x  = float(ts_data["ax_global"])
            y  = float(ts_data["ay_global"])

            raw[(scene, ts)].append({
                "track_id": track_id,
                "class":    cls,
                "x":        x,
                "y":        y,
            })

    # Build KDTree index
    print(f"  Building KDTrees for {len(raw):,} (scene,ts) pairs...")
    agent_index = {}
    for (scene, ts), agents in raw.items():
        pts = np.array([[a["x"], a["y"]] for a in agents], dtype=np.float64)
        agent_index[(scene, ts)] = {
            "tree":   KDTree(pts),
            "agents": agents,
        }

    print(f"  [OK] Agent index ready")
    return data, agent_index


# ════════════════════════════════════════════════════════════
# STAGE 2: Match neighbor cell → exact agent class
# ════════════════════════════════════════════════════════════
def match_neighbor_class(scene, ts, px, py, theta,
                          rel_x, rel_y,
                          self_track_id, agent_index):
    """
    Match a grid neighbor cell to an exact agent.

    Steps:
    1. Reconstruct approximate global position of neighbor:
       nb_approx = p_A + R_θA^T · [rel_x, rel_y]
    2. Query KDTree at (scene, ts) for nearest agent
    3. Skip self (self_track_id)
    4. Accept match if distance < MATCH_THRESHOLD

    Returns:
        class_name  : matched class or "unknown"
        match_error : distance between reconstructed and matched position (m)
    """
    nb_x, nb_y = reconstruct_global(px, py, theta, rel_x, rel_y)

    key = (scene, ts)
    if key not in agent_index:
        return "unknown", None

    entry  = agent_index[key]
    tree   = entry["tree"]
    agents = entry["agents"]
    n      = len(agents)

    # Query k nearest (skip self)
    k = min(n, 5)
    raw_dists, raw_idxs = tree.query([[nb_x, nb_y]], k=k)
    dists = np.atleast_1d(raw_dists[0])
    idxs  = np.atleast_1d(raw_idxs[0])

    for d, i in zip(dists, idxs):
        if float(d) > MATCH_THRESHOLD:
            break
        agent = agents[int(i)]
        if agent["track_id"] != self_track_id:
            return agent["class"], round(float(d), 4)

    return "unknown", None


# ════════════════════════════════════════════════════════════
# STAGE 3: Extract neighbors from grid
# ════════════════════════════════════════════════════════════
def extract_neighbors_from_grid(grid, px, py, theta):
    """
    Extract VRU neighbors from the 5x5 grid.

    Grid cell [row, col] → local frame position:
        local_x = (col - 2) * CELL_SIZE   (East in local frame)
        local_y = (2 - row) * CELL_SIZE   (North in local frame)

    NOTE: rel_x/y stored in grid are the center values of the cell
    in local frame. There is up to 0.5m quantization error.

    Returns sorted list of neighbors (by distance, closest first).
    """
    neighbors = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Skip center cell (self)
            if row == 2 and col == 2:
                continue

            if grid["vrus"][row][col] != 1:
                continue

            # Local frame coordinates (from grid)
            rel_x = float(grid["rel_x"][row][col])
            rel_y = float(grid["rel_y"][row][col])
            dist  = math.sqrt(rel_x**2 + rel_y**2)

            # Reconstructed global position
            abs_x, abs_y = reconstruct_global(px, py, theta, rel_x, rel_y)

            neighbors.append({
                "row":   row,
                "col":   col,
                "rel_x": round(rel_x, 4),
                "rel_y": round(rel_y, 4),
                "abs_x": round(abs_x, 4),
                "abs_y": round(abs_y, 4),
                "dist":  round(dist, 4),
            })

    # Sort by distance (closest first)
    neighbors.sort(key=lambda n: n["dist"])
    return neighbors


# ════════════════════════════════════════════════════════════
# STAGE 4: Build final CSV
# ════════════════════════════════════════════════════════════
def build_final_csv(data, agent_index):
    print(f"\n[STAGE 4] Building final dataset...")

    fieldnames = [
        # Identity
        "track_id", "scene", "class", "ts",
        # Global position & motion
        "x", "y", "speed", "heading_rad",
        # Environment
        "ground_type_id", "ground_type", "on_crosswalk",
        # Traffic lights
        "f1", "f2", "f3", "f1_state", "f2_state", "f3_state",
        # Neighbor summary
        "n_neighbors", "has_close_nb", "closest_dist",
    ]
    for i in range(1, MAX_NEIGHBORS + 1):
        fieldnames += [
            f"nb{i}_rel_x",      # local frame x
            f"nb{i}_rel_y",      # local frame y
            f"nb{i}_abs_x",      # reconstructed global x
            f"nb{i}_abs_y",      # reconstructed global y
            f"nb{i}_dist",       # distance A→B (m)
            f"nb{i}_class",      # matched class label
            f"nb{i}_match_err",  # matching error (m)
        ]

    total     = sum(len(t["timesteps"]) for t in data)
    done      = 0
    nb_total  = 0
    nb_matched = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for track in data:
            scene    = track["scene"]
            track_id = track["track_id"]
            cls      = track["class"]

            for ts_data in track["timesteps"]:
                done += 1
                if done % 50000 == 0:
                    pct = nb_matched/nb_total*100 if nb_total > 0 else 0
                    print(f"  {done:,}/{total:,}  nb_match={pct:.1f}%...", end="\r")

                ts    = int(ts_data["ts"])
                px    = float(ts_data["ax_global"])
                py    = float(ts_data["ay_global"])
                speed = float(ts_data.get("speed_a", 0))
                theta = float(ts_data.get("theta_a", 0))
                grid  = ts_data["grid"]
                tl    = ts_data.get("traffic_lights", {})

                # Ground type (cell [2,2] = under pedestrian's feet)
                gid   = int(grid["ground"][2][2])
                gname = GROUND_NAMES.get(gid, "unknown")

                # Traffic lights
                f1 = tl.get("f1", 11)
                f2 = tl.get("f2", 11)
                f3 = tl.get("f3", 11)

                # Neighbors
                neighbors = extract_neighbors_from_grid(grid, px, py, theta)
                n_nb      = len(neighbors)
                has_cl    = 1 if any(n["dist"] < CLOSE_DIST for n in neighbors) else 0
                cl_dist   = neighbors[0]["dist"] if neighbors else ""

                row_data = {
                    "track_id":     track_id,
                    "scene":        scene,
                    "class":        cls,
                    "ts":           ts,
                    "x":            round(px, 4),
                    "y":            round(py, 4),
                    "speed":        round(speed, 4),
                    "heading_rad":  round(theta, 4),
                    "ground_type_id": gid,
                    "ground_type":  gname,
                    "on_crosswalk": 1 if gid == 5 else 0,
                    "f1": f1, "f2": f2, "f3": f3,
                    "f1_state": LIGHT_STATES.get(f1, "unknown"),
                    "f2_state": LIGHT_STATES.get(f2, "unknown"),
                    "f3_state": LIGHT_STATES.get(f3, "unknown"),
                    "n_neighbors":  n_nb,
                    "has_close_nb": has_cl,
                    "closest_dist": cl_dist,
                }

                # Match each neighbor
                for ni in range(MAX_NEIGHBORS):
                    i1 = ni + 1
                    if ni < len(neighbors):
                        nb = neighbors[ni]
                        nb_total += 1

                        nb_cls, match_err = match_neighbor_class(
                            scene, ts, px, py, theta,
                            nb["rel_x"], nb["rel_y"],
                            track_id, agent_index
                        )

                        if nb_cls != "unknown":
                            nb_matched += 1

                        row_data[f"nb{i1}_rel_x"]     = nb["rel_x"]
                        row_data[f"nb{i1}_rel_y"]     = nb["rel_y"]
                        row_data[f"nb{i1}_abs_x"]     = nb["abs_x"]
                        row_data[f"nb{i1}_abs_y"]     = nb["abs_y"]
                        row_data[f"nb{i1}_dist"]      = nb["dist"]
                        row_data[f"nb{i1}_class"]     = nb_cls
                        row_data[f"nb{i1}_match_err"] = match_err or ""
                    else:
                        for key in [f"nb{i1}_rel_x", f"nb{i1}_rel_y",
                                    f"nb{i1}_abs_x", f"nb{i1}_abs_y",
                                    f"nb{i1}_dist", f"nb{i1}_class",
                                    f"nb{i1}_match_err"]:
                            row_data[key] = ""

                writer.writerow(row_data)

    pct = nb_matched/nb_total*100 if nb_total > 0 else 0
    print(f"\n  [OK] {total:,} frames written")
    print(f"  Neighbor match rate: {nb_matched:,}/{nb_total:,} ({pct:.1f}%)")


# ════════════════════════════════════════════════════════════
# STAGE 5: Validation
# ════════════════════════════════════════════════════════════
def validate():
    print(f"\n[STAGE 5] Validation...")
    import pandas as pd
    from collections import Counter

    df = pd.read_csv(OUTPUT_CSV, low_memory=False)
    print(f"  Rows: {len(df):,}")
    print(f"  Tracks: {df['track_id'].nunique():,}")
    print(f"  Scenes: {df['scene'].nunique():,}")

    has_nb = df[df["n_neighbors"] > 0].copy()
    print(f"\n  Frames with neighbors: {len(has_nb):,} ({len(has_nb)/len(df)*100:.1f}%)")

    # Match rate
    known = has_nb["nb1_class"].notna() & \
            (has_nb["nb1_class"] != "") & \
            (has_nb["nb1_class"] != "unknown")
    print(f"  nb1 match rate: {known.sum():,}/{len(has_nb):,} ({known.sum()/len(has_nb)*100:.1f}%)")

    # Class distribution
    print(f"\n  nb1_class distribution:")
    for cls, cnt in Counter(has_nb["nb1_class"]).most_common():
        print(f"    {cls:<15}: {cnt:>6,}")

    # Match error distribution
    errs = pd.to_numeric(has_nb["nb1_match_err"], errors="coerce").dropna()
    if len(errs) > 0:
        print(f"\n  Match error (nb1):")
        print(f"    mean : {errs.mean():.4f} m")
        print(f"    median: {errs.median():.4f} m")
        print(f"    max  : {errs.max():.4f} m")

    # Sanity: distance distribution
    dists = pd.to_numeric(has_nb["closest_dist"], errors="coerce").dropna()
    print(f"\n  Closest neighbor distance:")
    print(f"    mean: {dists.mean():.3f} m")
    print(f"    min : {dists.min():.3f} m")
    print(f"    max : {dists.max():.3f} m")

    # Ground type
    print(f"\n  Ground type distribution:")
    for g, cnt in df["ground_type"].value_counts().items():
        print(f"    {g:<12}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

    # Traffic lights
    print(f"\n  f1 state distribution:")
    for s, cnt in df["f1_state"].value_counts().items():
        print(f"    {s:<16}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

    # Save validation summary
    summary = {
        "total_rows": len(df),
        "total_tracks": df["track_id"].nunique(),
        "total_scenes": df["scene"].nunique(),
        "frames_with_neighbors": len(has_nb),
        "nb1_match_rate_pct": round(known.sum()/len(has_nb)*100, 2) if len(has_nb) > 0 else 0,
        "mean_match_error_m": round(errs.mean(), 4) if len(errs) > 0 else None,
        "mean_closest_dist_m": round(dists.mean(), 3),
    }
    pd.DataFrame([summary]).to_csv(VALIDATION_CSV, index=False)
    print(f"\n  [OK] Validation summary → {VALIDATION_CSV}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  CLEAN DATASET PIPELINE")
    print("=" * 60)

    # Stage 1: Load + index
    data, agent_index = load_and_build_agent_index(INPUT_JSON)

    # Stage 4: Build CSV
    build_final_csv(data, agent_index)

    # Stage 5: Validate
    validate()

    print(f"\n[DONE]")
    print(f"  Output:     {OUTPUT_CSV}")
    print(f"  Validation: {VALIDATION_CSV}")

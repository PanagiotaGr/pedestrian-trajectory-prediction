"""
build_full_context_from_csv_fixed.py
====================================
Build a pedestrian-centered full-context dataset from pairwise relative geometry,
with CORRECT merge to pedestrian_context.csv using matched_codes.csv.

Inputs
------
1. results/dataset_relative_geometry.csv
   Pairwise rows: one row per A-B interaction at timestamp ts

2. results/matched_codes.csv
   Maps sample_id to the real scene-based pedestrian identity through member_path
   Example:
       0000_20230322_081506/vrus/000/track.json
   becomes:
       0000_20230322_081506/000

3. results/pedestrian_context.csv
   Per-pedestrian contextual information (ground type, crosswalk, traffic lights, etc.)

Output
------
results/dataset_A_full_context_fixed.csv

One row per (track_id_A, ts), containing:
- aggregated social context around A
- class composition of neighbors
- merged semantic context for A from pedestrian_context.csv

Usage
-----
source venv/bin/activate
python build_full_context_from_csv_fixed.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))

PAIRWISE_CSV = RESULTS_DIR / "dataset_relative_geometry.csv"
MATCHED_CODES_CSV = RESULTS_DIR / "matched_codes.csv"
PEDESTRIAN_CONTEXT_CSV = RESULTS_DIR / "pedestrian_context.csv"

OUTPUT_CSV = RESULTS_DIR / "dataset_A_full_context_fixed.csv"

EPS = 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def classify_region(rel_x: float, rel_y: float) -> str:
    """
    Classify B position in A's local frame.
    Convention:
        +x = front
        -x = back
        +y = left
        -y = right
    """
    ax, ay = abs(rel_x), abs(rel_y)

    if ax < EPS and ay < EPS:
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


def classify_heading_relation(heading_diff: float) -> str:
    """
    Classify heading relation from heading difference in radians.

    same_direction: |diff| < 30 deg
    opposite_direction: |diff| > 150 deg
    crossing: otherwise
    """
    a = abs(float(heading_diff))
    if a < math.pi / 6:
        return "same_direction"
    if a > 5 * math.pi / 6:
        return "opposite_direction"
    return "crossing"


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
    # expected: [scene_id, 'vrus', ped_id, 'track.json']
    if len(parts) >= 3:
        scene_id = parts[0]
        ped_id = parts[2]
        return f"{scene_id}/{ped_id}"
    return ""


def safe_numeric(df: pd.DataFrame, cols: list[str], fill: float = 0.0) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill)


def count_by_mask(values: np.ndarray, mask: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    return int(mask.sum())


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_group(grp: pd.DataFrame) -> Dict:
    """
    Aggregate all B neighbors around pedestrian A at a fixed timestamp ts.
    """
    row0 = grp.iloc[0]
    n = len(grp)

    dists = grp["dist_xy"].to_numpy(dtype=float)
    closing = grp["closing_speed"].to_numpy(dtype=float)
    rel_vx = grp["rel_vx"].to_numpy(dtype=float)
    rel_vy = grp["rel_vy"].to_numpy(dtype=float)
    heading_diff = grp["heading_diff"].to_numpy(dtype=float)

    # Regions
    regions = grp.apply(lambda r: classify_region(r["rel_x"], r["rel_y"]), axis=1)
    region_counts = regions.value_counts().to_dict()

    def rc(name: str) -> int:
        return int(region_counts.get(name, 0))

    front_total = rc("front") + rc("front_left") + rc("front_right")
    back_total = rc("back") + rc("back_left") + rc("back_right")
    left_total = rc("left") + rc("front_left") + rc("back_left")
    right_total = rc("right") + rc("front_right") + rc("back_right")

    # Distance bins
    close_neighbors = int((dists < 2.0).sum())
    medium_neighbors = int(((dists >= 2.0) & (dists < 4.0)).sum())
    far_neighbors = int((dists >= 4.0).sum())

    # Motion
    approaching_count = int((closing > 0.1).sum())
    receding_count = int((closing < -0.1).sum())

    rel_speed_mag = np.sqrt(rel_vx**2 + rel_vy**2)

    # Heading
    heading_rel = pd.Series(heading_diff).apply(classify_heading_relation)
    heading_counts = heading_rel.value_counts().to_dict()

    # Risk
    collision_risk = int(np.any((dists < 1.0) & (closing > 0.0)))

    interaction_intensity = float(
        np.sum((1.0 / (dists + EPS)) * np.maximum(closing, 0.0))
    )

    # Class composition
    class_counts = {}
    if "class_B" in grp.columns:
        vc = grp["class_B"].fillna("unknown").astype(str).str.lower().value_counts()
        class_counts = vc.to_dict()

    def class_count(name: str) -> int:
        return int(class_counts.get(name, 0))

    return {
        # identity
        "sample_id": row0["sample_id"],
        "split": row0.get("split", ""),
        "track_id_A": row0["track_id_A"],
        "real_track_id_A": row0.get("real_track_id_A", ""),
        "ts": row0["ts"],

        # A state
        "ax": round(float(row0["ax"]), 4),
        "ay": round(float(row0["ay"]), 4),
        "speed_a": round(float(row0["speed_a"]), 4),
        "heading_a": round(float(row0["heading_a"]), 4),

        # counts
        "total_neighbors": n,
        "close_neighbors": close_neighbors,
        "medium_neighbors": medium_neighbors,
        "far_neighbors": far_neighbors,

        # directional counts
        "front_count": rc("front"),
        "back_count": rc("back"),
        "left_count": rc("left"),
        "right_count": rc("right"),
        "front_left_count": rc("front_left"),
        "front_right_count": rc("front_right"),
        "back_left_count": rc("back_left"),
        "back_right_count": rc("back_right"),

        "front_total": front_total,
        "back_total": back_total,
        "left_total": left_total,
        "right_total": right_total,

        # density
        "is_isolated": int(n == 0),
        "is_crowded": int(n >= 3),
        "high_density_front": int(front_total >= 2),
        "lateral_density": left_total + right_total,

        # motion
        "approaching_count": approaching_count,
        "receding_count": receding_count,
        "avg_closing_speed": round(float(closing.mean()), 4) if n else 0.0,
        "max_closing_speed": round(float(closing.max()), 4) if n else 0.0,

        # relative speed
        "avg_relative_speed": round(float(rel_speed_mag.mean()), 4) if n else 0.0,
        "max_relative_speed": round(float(rel_speed_mag.max()), 4) if n else 0.0,

        # heading relations
        "same_dir_count": int(heading_counts.get("same_direction", 0)),
        "opp_dir_count": int(heading_counts.get("opposite_direction", 0)),
        "crossing_count": int(heading_counts.get("crossing", 0)),

        # distance stats
        "min_distance": round(float(dists.min()), 4) if n else 0.0,
        "mean_distance": round(float(dists.mean()), 4) if n else 0.0,
        "max_distance": round(float(dists.max()), 4) if n else 0.0,

        # risk
        "collision_risk": collision_risk,
        "interaction_intensity": round(interaction_intensity, 4),

        # class composition
        "pedestrian_neighbor_count": class_count("pedestrian"),
        "cyclist_neighbor_count": class_count("cyclist"),
        "scooter_neighbor_count": class_count("scooter"),
        "motorcycle_neighbor_count": class_count("motorcycle"),
        "stroller_neighbor_count": class_count("stroller"),
        "unknown_neighbor_count": class_count("unknown"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  BUILD PEDESTRIAN FULL CONTEXT DATASET (FIXED)")
    print("=" * 55)

    # 1) Load pairwise data
    print(f"\n[→] Loading {PAIRWISE_CSV.name}...")
    df = pd.read_csv(PAIRWISE_CSV, low_memory=False)
    rows_loaded = len(df)
    print(f"  Rows: {rows_loaded:,}")

    safe_numeric(
        df,
        [
            "sample_id", "ts", "ax", "ay", "speed_a", "heading_a",
            "rel_x", "rel_y", "rel_vx", "rel_vy",
            "dist_xy", "heading_diff", "closing_speed"
        ],
        fill=0.0,
    )

    # 2) Load matched_codes and build real_track_id mapping
    print(f"[→] Loading {MATCHED_CODES_CSV.name}...")
    matched = pd.read_csv(MATCHED_CODES_CSV, low_memory=False)

    if "sample_id" not in matched.columns or "member_path" not in matched.columns:
        raise ValueError(
            "matched_codes.csv must contain columns: sample_id, member_path"
        )

    matched["real_track_id_A"] = matched["member_path"].apply(parse_real_track_id_from_member_path)
    matched = matched[["sample_id", "real_track_id_A"]].drop_duplicates()

    # sample_id in pairwise csv may be string-like '0000'; normalize to int if possible
    matched["sample_id"] = pd.to_numeric(matched["sample_id"], errors="coerce")
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce")

    # join pairwise -> matched_codes
    df = df.merge(matched, on="sample_id", how="left")

    missing_real_ids = int(df["real_track_id_A"].isna().sum())
    print(f"[→] sample_id → real_track_id_A mapping missing in {missing_real_ids:,} pairwise rows")

    # 3) Aggregate into one row per (track_id_A, ts)
    print(f"[→] Aggregating by (track_id_A, ts)...")
    groups = df.groupby(["track_id_A", "ts"], sort=False)

    records = []
    for i, (_, grp) in enumerate(groups, start=1):
        if i % 10000 == 0:
            print(f"  {i:,} groups...", end="\r")
        records.append(aggregate_group(grp))

    agg = pd.DataFrame(records)
    rows_saved_premerge = len(agg)
    print(f"\n  Aggregated rows: {rows_saved_premerge:,}")

    # 4) Load pedestrian context
    print(f"[→] Loading {PEDESTRIAN_CONTEXT_CSV.name}...")
    ctx = pd.read_csv(PEDESTRIAN_CONTEXT_CSV, low_memory=False)

    # Normalize types for merge
    if "track_id" not in ctx.columns or "ts" not in ctx.columns:
        raise ValueError(
            "pedestrian_context.csv must contain columns: track_id, ts"
        )

    safe_numeric(
        ctx,
        [
            "ts", "ground_type_id", "dist_to_crosswalk", "on_crosswalk",
            "f1", "f2", "f3", "n_neighbors", "has_close_neighbor", "closest_dist"
        ],
        fill=0.0,
    )

    # Keep only needed context columns
    ctx_cols = [
        "track_id", "ts",
        "ground_type_id", "ground_type_name",
        "dist_to_crosswalk", "on_crosswalk",
        "f1", "f2", "f3",
        "f1_state", "f2_state", "f3_state",
        "n_neighbors", "has_close_neighbor", "closest_dist"
    ]
    ctx_keep = ctx[ctx_cols].copy()

    # 5) Merge aggregated social context with semantic context
    print(f"[→] Merging aggregated rows with pedestrian_context.csv...")
    out = agg.merge(
        ctx_keep,
        left_on=["real_track_id_A", "ts"],
        right_on=["track_id", "ts"],
        how="left"
    )

    # optional derived flags
    out["any_green_light"] = (
        (out["f1_state"] == "green") |
        (out["f2_state"] == "green") |
        (out["f3_state"] == "green")
    ).fillna(False).astype(int)

    out["any_red_light"] = (
        (out["f1_state"] == "red") |
        (out["f2_state"] == "red") |
        (out["f3_state"] == "red")
    ).fillna(False).astype(int)

    # neighbors on crosswalk cannot be computed from pedestrian_context.csv directly for all Bs
    # unless B-side semantic context is also merged separately. Keep placeholder from A-side merge only.
    # We expose A-side crosswalk state correctly here.
    out["neighbors_on_crosswalk"] = np.nan

    # 6) Save
    out.to_csv(OUTPUT_CSV, index=False)
    size_mb = OUTPUT_CSV.stat().st_size / 1e6
    print(f"\n[OK] Saved: {OUTPUT_CSV.name} ({size_mb:.1f} MB)")

    # 7) Validation
    matched_context_rows = int(out["ground_type_id"].notna().sum())
    unmatched_rows = int(out["ground_type_id"].isna().sum())

    pct_on_crosswalk = (
        100.0 * float((out["on_crosswalk"] == 1).mean())
        if "on_crosswalk" in out.columns else 0.0
    )

    pct_any_green = 100.0 * float(out["any_green_light"].mean()) if len(out) else 0.0
    pct_any_red = 100.0 * float(out["any_red_light"].mean()) if len(out) else 0.0

    print("\n" + "=" * 55)
    print("  VALIDATION")
    print("=" * 55)
    print(f"  Rows loaded:         {rows_loaded:,}")
    print(f"  Rows saved:          {len(out):,}")
    print(f"  Matched context:     {matched_context_rows:,}")
    print(f"  Unmatched rows:      {unmatched_rows:,}")
    print(f"  Avg neighbors:       {out['total_neighbors'].mean():.3f}")
    print(f"  % on_crosswalk (A):  {pct_on_crosswalk:.1f}%")
    print(f"  % any green light:   {pct_any_green:.1f}%")
    print(f"  % any red light:     {pct_any_red:.1f}%")
    print(f"  % collision risk:    {100.0 * float(out['collision_risk'].mean()):.1f}%")

    print("\n  Neighbor class totals:")
    for col in [
        "pedestrian_neighbor_count",
        "cyclist_neighbor_count",
        "scooter_neighbor_count",
        "motorcycle_neighbor_count",
        "stroller_neighbor_count",
        "unknown_neighbor_count",
    ]:
        print(f"    {col:<28}: {int(out[col].sum()):,}")

    print("\n  Directional averages:")
    for col in ["front_total", "back_total", "left_total", "right_total"]:
        print(f"    {col:<16}: {out[col].mean():.3f}")

    print(f"\n[DONE] → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

import os
import json
import math
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RESULTS_DIR = Path(os.path.expanduser("~/imptc_project/results"))
INPUT_JSON = RESULTS_DIR / "grid_dataset_final.json"

OUT_CELL = RESULTS_DIR / "grid_cell_context.csv"
OUT_SUMMARY = RESULTS_DIR / "grid_context_summary.csv"

GRID_SIZE = 5
CENTER = 2

GROUND_NAMES = {
    0: "road", 1: "sidewalk", 2: "ground", 3: "curb",
    4: "road_line", 5: "crosswalk", 6: "bikelane", 7: "unknown"
}

LIGHT_STATES = {
    4: "green", 10: "red", 20: "yellow",
    30: "red_yellow", 2: "yellow_blinking", 11: "disabled"
}

# ─────────────────────────────────────────────
# REGION CLASSIFICATION
# ─────────────────────────────────────────────
def classify_region(x, y):
    ax, ay = abs(x), abs(y)
    if ax < 1e-6 and ay < 1e-6:
        return "center"
    if ax > ay:
        return "front" if x > 0 else "back"
    else:
        return "left" if y > 0 else "right"

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading JSON...")
    with open(INPUT_JSON) as f:
        data = json.load(f)

    cell_rows = []
    summary_rows = []

    sample_id = 0

    for track in data:
        track_id = track["track_id"]

        for ts_data in track["timesteps"]:

            ts = ts_data["ts"]
            grid = ts_data["grid"]
            tl = ts_data.get("traffic_lights", {})

            f1 = tl.get("f1", 11)
            f2 = tl.get("f2", 11)
            f3 = tl.get("f3", 11)

            f1s = LIGHT_STATES.get(f1, "unknown")
            f2s = LIGHT_STATES.get(f2, "unknown")
            f3s = LIGHT_STATES.get(f3, "unknown")

            # ───────── SUMMARY accumulators ─────────
            total_vru = 0
            total_veh = 0

            front_vru = back_vru = left_vru = right_vru = 0
            front_cw = back_cw = left_cw = right_cw = 0

            nearest_vru_dist = 999
            nearest_veh_dist = 999

            # 1m neighborhood
            neigh = {
                "left": {}, "right": {}, "front": {}, "back": {}
            }

            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):

                    rel_x = (col - CENTER) * 1.0
                    rel_y = (CENTER - row) * 1.0
                    dist = math.sqrt(rel_x**2 + rel_y**2)

                    vru = grid["vrus"][row][col]
                    veh = grid["vehicles"][row][col]
                    gid = int(grid["ground"][row][col])
                    gname = GROUND_NAMES.get(gid, "unknown")

                    rvx = float(grid["rel_vx"][row][col])
                    rvy = float(grid["rel_vy"][row][col])

                    region = classify_region(rel_x, rel_y)

                    # ───── CELL OUTPUT ─────
                    cell_rows.append({
                        "sample_id": sample_id,
                        "track_id": track_id,
                        "ts": ts,
                        "row": row,
                        "col": col,
                        "rel_x": rel_x,
                        "rel_y": rel_y,
                        "distance": dist,
                        "has_vru": vru,
                        "has_vehicle": veh,
                        "ground_id": gid,
                        "ground_name": gname,
                        "rel_vx": rvx,
                        "rel_vy": rvy,
                        "region": region,
                        "f1": f1, "f2": f2, "f3": f3,
                        "f1_state": f1s,
                        "f2_state": f2s,
                        "f3_state": f3s
                    })

                    # ───── SUMMARY ─────
                    if vru == 1:
                        total_vru += 1
                        nearest_vru_dist = min(nearest_vru_dist, dist)

                        if region == "front": front_vru += 1
                        if region == "back": back_vru += 1
                        if region == "left": left_vru += 1
                        if region == "right": right_vru += 1

                    if veh == 1:
                        total_veh += 1
                        nearest_veh_dist = min(nearest_veh_dist, dist)

                    if gname == "crosswalk":
                        if region == "front": front_cw += 1
                        if region == "back": back_cw += 1
                        if region == "left": left_cw += 1
                        if region == "right": right_cw += 1

                    # ───── 1m cells ─────
                    if row == 2 and col == 1:
                        neigh["left"] = {"g": gname, "vru": vru, "veh": veh, "vx": rvx, "vy": rvy}
                    if row == 2 and col == 3:
                        neigh["right"] = {"g": gname, "vru": vru, "veh": veh, "vx": rvx, "vy": rvy}
                    if row == 1 and col == 2:
                        neigh["front"] = {"g": gname, "vru": vru, "veh": veh, "vx": rvx, "vy": rvy}
                    if row == 3 and col == 2:
                        neigh["back"] = {"g": gname, "vru": vru, "veh": veh, "vx": rvx, "vy": rvy}

            center_gid = int(grid["ground"][2][2])
            center_g = GROUND_NAMES.get(center_gid, "unknown")

            summary_rows.append({
                "sample_id": sample_id,
                "track_id": track_id,
                "ts": ts,

                "ground_A": center_g,
                "on_crosswalk_A": 1 if center_gid == 5 else 0,

                "f1_state": f1s,
                "f2_state": f2s,
                "f3_state": f3s,

                "total_vru_cells": total_vru,
                "total_vehicle_cells": total_veh,

                "front_vru": front_vru,
                "back_vru": back_vru,
                "left_vru": left_vru,
                "right_vru": right_vru,

                "front_crosswalk": front_cw,
                "back_crosswalk": back_cw,
                "left_crosswalk": left_cw,
                "right_crosswalk": right_cw,

                "nearest_vru_dist": nearest_vru_dist,
                "nearest_vehicle_dist": nearest_veh_dist,

                "left_ground": neigh.get("left", {}).get("g"),
                "right_ground": neigh.get("right", {}).get("g"),
                "front_ground": neigh.get("front", {}).get("g"),
                "back_ground": neigh.get("back", {}).get("g"),

                "left_vru": neigh.get("left", {}).get("vru"),
                "right_vru": neigh.get("right", {}).get("vru"),
                "front_vru_1m": neigh.get("front", {}).get("vru"),
                "back_vru_1m": neigh.get("back", {}).get("vru"),
            })

            sample_id += 1

    # SAVE
    pd.DataFrame(cell_rows).to_csv(OUT_CELL, index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    print("\nDONE")
    print("Cells:", len(cell_rows))
    print("Summary:", len(summary_rows))


if __name__ == "__main__":
    main()

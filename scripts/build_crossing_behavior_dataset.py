import os
import csv

INPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_final_model_dataset.csv")
OUTPUT_PATH = os.path.expanduser("~/imptc_project/results/pedestrian_crossing_behavior.csv")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default


rows_out = []

with open(INPUT_PATH, "r") as f:
    reader = csv.DictReader(f)

    for r in reader:
        if r["status"] != "ok":
            continue

        # βασικά
        x = safe_float(r.get("target_x", 0))
        y = safe_float(r.get("target_y", 0))
        timestamp = r["timestamp"]

        displacement = safe_float(r.get("displacement"))
        speed = safe_float(r.get("avg_speed_est"))

        # lights
        f1 = safe_int(r.get("f1_state"))
        f2 = safe_int(r.get("f2_state"))
        f3 = safe_int(r.get("f3_state"))

        n_green = safe_int(r.get("n_ped_green"))
        n_red = safe_int(r.get("n_ped_red"))

        # context
        nearest_dist = safe_float(r.get("nearest_dist"))
        neighbors = safe_int(r.get("n_neighbors_found"))

        has_green = 1 if n_green > 0 else 0
        red_only = 1 if (n_green == 0 and n_red > 0) else 0
        is_moving = 1 if displacement > 0.5 else 0  # threshold μπορείς να αλλάξεις

        rows_out.append({
            "sample_id": r["sample_id"],
            "scene_path": r["scene_path"],
            "timestamp": timestamp,

            "target_x": x,
            "target_y": y,

            "displacement": displacement,
            "avg_speed": speed,

            "f1_state": f1,
            "f2_state": f2,
            "f3_state": f3,

            "n_ped_green": n_green,
            "n_ped_red": n_red,

            "has_green": has_green,
            "red_only": red_only,
            "is_moving": is_moving,

            "nearest_dist": nearest_dist,
            "n_neighbors": neighbors,
        })


fieldnames = list(rows_out[0].keys())

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(rows_out))

#ΛΑΘΟΣ ΑΡΧΕΙΟΟΟΟ
import os
import csv
import json
import argparse
from bisect import bisect_left


STATE_MAP = {
    2: "yellow_blinking",
    4: "green",
    10: "red",
    11: "disabled",
    20: "yellow",
    30: "red_yellow",
}


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_signal_data(scene_dir):
    path = os.path.join(scene_dir, "context", "traffic_light_signals.json")
    if not os.path.exists(path):
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    status_data = obj.get("status_data", {})
    ts_list = sorted(int(ts) for ts in status_data.keys())
    return ts_list, status_data


def nearest_timestamp(ts_list, target_ts):
    if not ts_list:
        return None

    i = bisect_left(ts_list, target_ts)

    if i == 0:
        return ts_list[0]
    if i == len(ts_list):
        return ts_list[-1]

    before = ts_list[i - 1]
    after = ts_list[i]

    if abs(target_ts - before) <= abs(after - target_ts):
        return before
    return after


def decode_state(code):
    try:
        code = int(code)
    except Exception:
        return ""
    return STATE_MAP.get(code, f"unknown_{code}")


def main():
    parser = argparse.ArgumentParser(description="Annotate pedestrian dataset with traffic light states f1/f2/f3")
    parser.add_argument(
        "--input-csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map.csv"),
    )
    parser.add_argument(
        "--scenes-root",
        default=os.path.expanduser("~/imptc_project/raw_scene"),
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map_and_lights.csv"),
    )
    args = parser.parse_args()

    rows = load_csv_rows(args.input_csv)
    print("Loaded rows:", len(rows))

    # cache ανά scene
    signal_cache = {}

    out_rows = []

    for idx, r in enumerate(rows):
        scene_path = r.get("scene_path", "")
        ts_str = r.get("ts", "")

        try:
            target_ts = int(ts_str)
        except Exception:
            target_ts = None

        scene_dir = os.path.join(args.scenes_root, scene_path)

        if scene_path not in signal_cache:
            ts_list, status_data = load_signal_data(scene_dir)
            signal_cache[scene_path] = (ts_list, status_data)

        ts_list, status_data = signal_cache[scene_path]

        nearest_ts = None
        signal_row = {}

        if target_ts is not None and ts_list:
            nearest_ts = nearest_timestamp(ts_list, target_ts)
            signal_row = status_data.get(str(nearest_ts), {})

        f1 = signal_row.get("f1", "")
        f2 = signal_row.get("f2", "")
        f3 = signal_row.get("f3", "")

        r["nearest_signal_ts"] = "" if nearest_ts is None else str(nearest_ts)

        r["f1"] = f1
        r["f2"] = f2
        r["f3"] = f3

        r["f1_state"] = decode_state(f1) if f1 != "" else ""
        r["f2_state"] = decode_state(f2) if f2 != "" else ""
        r["f3_state"] = decode_state(f3) if f3 != "" else ""

        out_rows.append(r)

        if idx % 1000 == 0:
            print("Processed:", idx)

    fieldnames = list(out_rows[0].keys())

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nDONE")
    print("Saved:", args.output_csv)
    print("Rows:", len(out_rows))


if __name__ == "__main__":
    main()

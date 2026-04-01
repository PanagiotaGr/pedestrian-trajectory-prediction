import os
import csv
import json
import tarfile
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


def decode_state(code):
    try:
        code = int(code)
    except Exception:
        return ""
    return STATE_MAP.get(code, f"unknown_{code}")


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def load_signal_data_from_archive(archive_path, scene_path):
    member_name = f"{scene_path}/context/traffic_light_signals.json"

    if not os.path.exists(archive_path):
        return None, None

    try:
        with tarfile.open(archive_path, "r:*") as tar:
            try:
                member = tar.getmember(member_name)
            except KeyError:
                return None, None

            f = tar.extractfile(member)
            if f is None:
                return None, None

            obj = json.load(f)
    except Exception:
        return None, None

    status_data = obj.get("status_data", {})
    if not isinstance(status_data, dict):
        return None, None

    ts_list = sorted(int(ts) for ts in status_data.keys())
    return ts_list, status_data


def main():
    parser = argparse.ArgumentParser(
        description="Annotate pedestrian dataset with traffic light states from tar.gz archives"
    )
    parser.add_argument(
        "--input-csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map.csv"),
    )
    parser.add_argument(
        "--archives-dir",
        default=os.path.expanduser("~/imptc_project/data"),
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map_and_lights.csv"),
    )
    args = parser.parse_args()

    rows = load_csv_rows(args.input_csv)
    print("Loaded rows:", len(rows))

    # cache ανά (archive, scene_path)
    signal_cache = {}

    out_rows = []

    for idx, r in enumerate(rows):
        archive_name = r.get("archive", "")
        scene_path = r.get("scene_path", "")
        ts_str = r.get("ts", "")

        try:
            target_ts = int(ts_str)
        except Exception:
            target_ts = None

        archive_path = os.path.join(args.archives_dir, archive_name)

        cache_key = (archive_name, scene_path)
        if cache_key not in signal_cache:
            ts_list, status_data = load_signal_data_from_archive(archive_path, scene_path)
            signal_cache[cache_key] = (ts_list, status_data)

        ts_list, status_data = signal_cache[cache_key]

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

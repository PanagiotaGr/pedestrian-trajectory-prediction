import os
import json
import argparse


STATE_NAMES = {
    2: "yellow-blinking",
    4: "green",
    10: "red",
    11: "disabled",
    20: "yellow",
    30: "red-yellow",
}


def load_status_data(scene_dir):
    path = os.path.join(scene_dir, "context", "traffic_light_signals.json")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["status_data"]


def nearest_timestamp(status_data, target_ts):
    target_ts = int(target_ts)
    ts_list = [int(ts) for ts in status_data.keys()]
    best_ts = min(ts_list, key=lambda x: abs(x - target_ts))
    return str(best_ts)


def main():
    parser = argparse.ArgumentParser(description="Inspect pedestrian traffic lights at one timestamp")
    parser.add_argument("scene_dir")
    parser.add_argument("timestamp")
    args = parser.parse_args()

    status_data = load_status_data(args.scene_dir)
    best_ts = nearest_timestamp(status_data, args.timestamp)
    rec = status_data[best_ts]

    print("Scene:", args.scene_dir)
    print("Requested timestamp:", args.timestamp)
    print("Nearest traffic-light timestamp:", best_ts)
    print()

    for key in ["f1", "f2", "f3"]:
        code = rec.get(key)
        print(f"{key}: code={code}, state={STATE_NAMES.get(code, 'unknown')}")


if __name__ == "__main__":
    main()

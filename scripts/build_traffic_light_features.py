import os
import csv
import json

traffic_csv = os.path.expanduser("~/imptc_project/results/pedestrian_traffic_lights.csv")
out_csv = os.path.expanduser("~/imptc_project/results/pedestrian_light_features.csv")

# 0 = off/disabled
# 1 = red
# 2 = green
# 3 = yellow
# 4 = red-yellow
# 5 = yellow-blinking
STATE_MAP = {
    11: 0,
    10: 1,
    4: 2,
    20: 3,
    30: 4,
    2: 5,
}

PEDESTRIAN_SIGNALS = ["f1", "f2", "f3"]

rows_out = []

with open(traffic_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sample_id = row["sample_id"]
        status = row.get("status", "")

        out_row = {
            "sample_id": sample_id,
            "f1_state": "",
            "f2_state": "",
            "f3_state": "",
            "n_ped_red": 0,
            "n_ped_green": 0,
            "n_ped_off": 0,
            "n_ped_yellow": 0,
            "n_ped_redyellow": 0,
            "n_ped_yellow_blinking": 0,
            "status": status,
        }

        if status != "ok":
            rows_out.append(out_row)
            continue

        try:
            signal_states = json.loads(row["signal_states_json"])
        except Exception:
            out_row["status"] = "bad_json"
            rows_out.append(out_row)
            continue

        mapped_states = {}

        for sig in PEDESTRIAN_SIGNALS:
            raw_val = signal_states.get(sig, None)

            if raw_val is None:
                mapped_states[sig] = ""
                continue

            try:
                raw_val = int(raw_val)
            except Exception:
                mapped_states[sig] = ""
                continue

            mapped_states[sig] = STATE_MAP.get(raw_val, "")

        out_row["f1_state"] = mapped_states["f1"]
        out_row["f2_state"] = mapped_states["f2"]
        out_row["f3_state"] = mapped_states["f3"]

        states = [mapped_states[s] for s in PEDESTRIAN_SIGNALS if mapped_states[s] != ""]

        out_row["n_ped_off"] = sum(1 for x in states if x == 0)
        out_row["n_ped_red"] = sum(1 for x in states if x == 1)
        out_row["n_ped_green"] = sum(1 for x in states if x == 2)
        out_row["n_ped_yellow"] = sum(1 for x in states if x == 3)
        out_row["n_ped_redyellow"] = sum(1 for x in states if x == 4)
        out_row["n_ped_yellow_blinking"] = sum(1 for x in states if x == 5)

        rows_out.append(out_row)

fieldnames = [
    "sample_id",
    "f1_state",
    "f2_state",
    "f3_state",
    "n_ped_red",
    "n_ped_green",
    "n_ped_off",
    "n_ped_yellow",
    "n_ped_redyellow",
    "n_ped_yellow_blinking",
    "status",
]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print("Saved:", out_csv)
print("Rows:", len(rows_out))

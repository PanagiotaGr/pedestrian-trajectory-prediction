import pandas as pd
import numpy as np

INPUT_CSV = "/home/pg2a1/imptc_project/results/pedestrian_final_clean.csv"
OUTPUT_CSV = "/home/pg2a1/imptc_project/results/pedestrian_neighbors_within_5m_2d.csv"

RADIUS_METERS = 5.0


def build_neighbors_within_radius_2d(df: pd.DataFrame, radius: float = 5.0) -> pd.DataFrame:
    """
    Για κάθε timestamp:
    - κάθε pedestrian θεωρείται target
    - βρίσκουμε όλους τους άλλους agents στο ίδιο frame
    - υπολογίζουμε dx, dy και ευκλείδεια απόσταση σε 2D
    - κρατάμε μόνο όσους είναι εντός radius
    """

    required_cols = [
        "sample_id",
        "scene_path",
        "timestamp",
        "track_id",
        "class_name",
        "x",
        "y",
        "velocity",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Λείπουν στήλες από το input CSV: {missing}")

    results = []

    grouped = df.groupby(["scene_path", "timestamp"], sort=False)

    for (scene_path, timestamp), frame_df in grouped:
        frame_df = frame_df.reset_index(drop=True)

        # Targets = μόνο pedestrians
        targets_df = frame_df[frame_df["class_name"] == "person"].copy()

        if targets_df.empty:
            continue

        for _, target in targets_df.iterrows():
            tx = float(target["x"])
            ty = float(target["y"])

            target_id = str(target["track_id"])
            target_class_name = target["class_name"]
            target_velocity = float(target["velocity"])

            for _, other in frame_df.iterrows():
                other_id = str(other["track_id"])

                # Δεν θέλουμε ο target να είναι γείτονας του εαυτού του
                if other_id == target_id:
                    continue

                ox = float(other["x"])
                oy = float(other["y"])

                dx = ox - tx
                dy = oy - ty

                distance = np.sqrt(dx**2 + dy**2)

                if distance <= radius:
                    results.append({
                        "sample_id": target["sample_id"],
                        "scene_path": scene_path,
                        "timestamp": timestamp,
                        "target_id": target_id,
                        "target_class_name": target_class_name,
                        "target_x_ego": 0.0,
                        "target_y_ego": 0.0,
                        "target_velocity": target_velocity,
                        "other_id": other_id,
                        "other_class_name": other["class_name"],
                        "other_velocity": float(other["velocity"]),
                        "dx": dx,
                        "dy": dy,
                        "distance": distance,
                    })

    out_df = pd.DataFrame(results)

    if not out_df.empty:
        out_df = out_df.sort_values(
            by=["scene_path", "timestamp", "target_id", "distance"],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)

    return out_df


def main():
    df = pd.read_csv(INPUT_CSV)

    out_df = build_neighbors_within_radius_2d(df, radius=RADIUS_METERS)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(out_df)}")
    print(f"Radius: {RADIUS_METERS}")


if __name__ == "__main__":
    main()

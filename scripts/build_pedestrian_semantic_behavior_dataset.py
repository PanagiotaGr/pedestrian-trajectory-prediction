import os
import pandas as pd

map_path = os.path.expanduser("~/imptc_project/results/pedestrian_map_labels.csv")
behavior_path = os.path.expanduser("~/imptc_project/results/pedestrian_crossing_behavior.csv")
out_path = os.path.expanduser("~/imptc_project/results/pedestrian_semantic_behavior.csv")

map_df = pd.read_csv(map_path)
beh_df = pd.read_csv(behavior_path)

# σβήνουμε λάθος coordinates από το behavior dataset αν υπάρχουν
for col in ["target_x", "target_y"]:
    if col in beh_df.columns:
        beh_df = beh_df.drop(columns=[col])

df = beh_df.merge(
    map_df[
        [
            "sample_id",
            "target_x",
            "target_y",
            "map_label_id",
            "map_label_name",
            "is_crosswalk",
            "is_sidewalk",
            "is_road",
            "is_bikelane",
            "nearest_map_dist",
        ]
    ],
    on="sample_id",
    how="left"
)

df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

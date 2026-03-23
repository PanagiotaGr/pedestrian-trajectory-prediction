import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data_path = os.path.expanduser("~/imptc_project/results/pedestrian_ego_final_dataset.csv")

df = pd.read_csv(data_path)

# κρατάμε valid rows
if "status" in df.columns:
    df = df[df["status"] == "ok"]

# target
y = df["displacement"]

# στήλες που ΔΕΝ πρέπει να μπουν στο μοντέλο
drop_cols = [
    "sample_id",
    "scene_path",
    "archive",
    "timestamp",
    "target_id",
    "target_class_name",
    "nearest_type",
    "nearest_class",
    "light_status",
    "signal_states_json",
    "displacement",
    "matched_signal_ts",
    "avg_speed_est",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# κρατάμε μόνο numeric
X = X.select_dtypes(include=["number"])

print("Rows:", len(df))
print("Num features:", len(X.columns))
print("Features:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("MAE:", mae)

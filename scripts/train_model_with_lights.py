import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data_path = os.path.expanduser("~/imptc_project/results/pedestrian_model_dataset.csv")

df = pd.read_csv(data_path)

df = df[(df["status"] == "ok") & (df["light_status"] == "ok")].copy()

feature_cols = [
    "nearest_type",
    "nearest_dist",
    "n_neighbors_found",
    "ped_signal_count",
    "ped_unique_signal_states",
    "ped_has_green",
    "ped_has_red",
    "ped_has_yellow",
    "ped_has_redyellow",
    "ped_has_yellow_blinking",
    "ped_has_disabled",
    "ped_all_green",
    "ped_all_red",
    "ped_majority_green",
    "ped_majority_red",
]

target_col = "displacement"

X = df[feature_cols].copy()
y = df[target_col].copy()

categorical_features = ["nearest_type"]
numeric_features = [c for c in feature_cols if c not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=10
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("Features with pedestrian lights only:", feature_cols)
print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("MAE:", mae)

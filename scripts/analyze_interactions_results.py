import os
import pandas as pd
import matplotlib.pyplot as plt

summary_path = os.path.expanduser("~/imptc_project/results/interactions_summary.csv")
detailed_path = os.path.expanduser("~/imptc_project/results/interactions_detailed.csv")
out_dir = os.path.expanduser("~/imptc_project/results/plots_interactions")

os.makedirs(out_dir, exist_ok=True)

# load data
df = pd.read_csv(summary_path)

# καθάρισμα unmatched
df_ok = df[df["target_found"] == 1]

print("Total samples:", len(df))
print("Matched samples:", len(df_ok))

# -------------------------------
# 1. Histogram interactions ανά sample
# -------------------------------
plt.figure()
df_ok["n_interacting_tracks"].hist(bins=40)
plt.xlabel("Number of interactions per sample")
plt.ylabel("Frequency")
plt.title("Distribution of interactions per sample")
plt.savefig(os.path.join(out_dir, "hist_interactions_per_sample.png"))

# -------------------------------
# 2. VRU vs Vehicle interactions
# -------------------------------
plt.figure()
vru_total = df_ok["n_vru_interactions"].sum()
veh_total = df_ok["n_vehicle_interactions"].sum()

plt.bar(["VRUs", "Vehicles"], [vru_total, veh_total])
plt.title("Total interactions: VRUs vs Vehicles")
plt.savefig(os.path.join(out_dir, "bar_vru_vs_vehicle.png"))

# -------------------------------
# 3. Strongest interaction distance
# -------------------------------
plt.figure()
df_ok["strongest_min_dist"].dropna().astype(float).hist(bins=40)
plt.xlabel("Minimum distance (strongest interaction)")
plt.ylabel("Frequency")
plt.title("Distribution of minimum distances")
plt.savefig(os.path.join(out_dir, "hist_min_distance.png"))

# -------------------------------
# 4. Top samples
# -------------------------------
top = df_ok.sort_values("n_interacting_tracks", ascending=False).head(20)
top.to_csv(os.path.join(out_dir, "top_20_samples.csv"), index=False)

print("Saved plots in:", out_dir)

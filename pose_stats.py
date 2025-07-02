#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob

sns.set(style='whitegrid')
DATA_DIR = "./pose_csvs"  # path to folder with CSVs
OUTPUT_DIR = "./slam_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all CSV files
all_data = []
for file_path in glob(os.path.join(DATA_DIR, "*.csv")):
    filename = os.path.basename(file_path).replace(".csv", "")
    algorithm, map_name = filename.split("_", 1)
    df = pd.read_csv(file_path, delimiter=";")
    df["Algorithm"] = algorithm
    df["Map"] = map_name
    df["Time [s]"] = df.index * 5  # assuming 5-second intervals
    df.rename(columns={
        "slam_x": "pose x",
        "slam_y": "pose y",
        "gt_x": "gt x",
        "gt_y": "gt y",
        "error": "error"
    }, inplace=True)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# === TABLE: Summary Statistics ===
summary = data.groupby(["Algorithm", "Map"]).agg(
    Total_Time_s=('Time [s]', 'max'),
    Avg_Error=('error', 'mean'),
    Max_Error=('error', 'max'),
    Min_Error=('error', 'min'),
    Std_Dev=('error', 'std')
).reset_index()
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"), index=False)
print("Saved summary_statistics.csv")

# === PLOT 1: Mean position error over time (separate for each map) ===
for map_name, map_group in data.groupby("Map"):
    plt.figure(figsize=(12, 6))
    for alg, alg_group in map_group.groupby("Algorithm"):
        mean_error = alg_group.groupby("Time [s]")["error"].mean()
        plt.plot(mean_error.index, mean_error.values, label=alg)
    plt.title(f"Position error over time: {map_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"error_over_time_{map_name}.png"))
    plt.close()

# === PLOT 2: Trajectories (separate for each file) ===
for (alg, map_name), grp in data.groupby(["Algorithm", "Map"]):
    plt.figure()
    plt.plot(grp["gt x"], grp["gt y"], label="Ground truth", linewidth=2)
    plt.plot(grp["pose x"], grp["pose y"], label="Estimated position", linestyle='--')
    plt.title(f"Trajectory {alg} – {map_name}")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"trajectory_{alg}_{map_name}.png"))
    plt.close()

# === PLOT 3: Boxplot of error distribution ===
# plt.figure(figsize=(10, 6))
# sns.boxplot(x="Algorithm", y="error", data=data)
# plt.title("Distribution of position error for algorithms")
# plt.ylabel("Position error [m]")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_error_distribution.png"))
# plt.close()

# === PLOT 4: ECDF (Empirical Cumulative Distribution Function) ===
plt.figure(figsize=(10, 6))
for alg, grp in data.groupby("Algorithm"):
    errors = pd.to_numeric(grp["error"], errors='coerce').dropna()
    sorted_errors = np.sort(errors)
    ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    if len(sorted_errors) > 0:
        plt.plot(sorted_errors, ecdf, label=alg)

plt.title("Empirical cumulative distribution of position error")
plt.xlabel("Error [m]")
plt.ylabel("Cumulative probability")
plt.legend(title="Algorithm")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ecdf_error.png"))
plt.close()

# === PLOT 5: Mean error of algorithms for each map ===
avg_errors = data.groupby(["Algorithm", "Map"])["error"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_errors, x="Algorithm", y="error", hue="Map")
plt.title("Mean position error of algorithms across maps")
plt.ylabel("Mean error [m]")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "barplot_avg_error.png"))
plt.close()

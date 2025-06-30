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
    df["Algorytm"] = algorithm
    df["Świat"] = map_name
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
summary = data.groupby(["Algorytm", "Świat"]).agg(
    Total_Time_s=('Time [s]', 'max'),
    Avg_Error=('error', 'mean'),
    Max_Error=('error', 'max'),
    Min_Error=('error', 'min'),
    Std_Dev=('error', 'std')
).reset_index()
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"), index=False)
print("Saved summary_statistics.csv")

# === WYKRES 1: Średni błąd położenia w czasie (osobno dla każdej mapy) ===
for map_name, map_group in data.groupby("Świat"):
    plt.figure(figsize=(12, 6))
    for alg, alg_group in map_group.groupby("Algorytm"):
        mean_error = alg_group.groupby("Time [s]")["error"].mean()
        plt.plot(mean_error.index, mean_error.values, label=alg)
    plt.title(f"Błąd położenia w czasie: {map_name}")
    plt.xlabel("Czas [s]")
    plt.ylabel("Błąd położenia [m]")
    plt.legend(title="Algorytm")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"error_over_time_{map_name}.png"))
    plt.close()


# === WYKRES 2: Trajektorie (osobno dla każdego pliku) ===
for (alg, map_name), grp in data.groupby(["Algorytm", "Świat"]):
    plt.figure()
    plt.plot(grp["gt x"], grp["gt y"], label="Pozycja rzeczywista", linewidth=2)
    plt.plot(grp["pose x"], grp["pose y"], label="Pozycja szacowana", linestyle='--')
    plt.title(f"Trajektoria {alg} – {map_name}")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"trajectory_{alg}_{map_name}.png"))
    plt.close()


# === WYKRES 3: Wykres pudełkowy rozkładu błędu ===
# plt.figure(figsize=(10, 6))
# sns.boxplot(x="Algorithm", y="error", data=data)
# plt.title("Rozkład błędu położenia dla algorytmów")
# plt.ylabel("Błąd położenia [m]")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_error_distribution.png"))
# plt.close()


# === WYKRES 4: ECDF (Empiryczna dystrybuanta skumulowana) ===
plt.figure(figsize=(10, 6))
for alg, grp in data.groupby("Algorytm"):
    errors = pd.to_numeric(grp["error"], errors='coerce').dropna()
    sorted_errors = np.sort(errors)
    ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    if len(sorted_errors) > 0:
        plt.plot(sorted_errors, ecdf, label=alg)

plt.title("Empiryczna dystrybuanta skumulowana błędu położenia")
plt.xlabel("Błąd [m]")
plt.ylabel("Prawdopodobieństwo skumulowane")
plt.legend(title="Algorytm")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ecdf_error.png"))
plt.close()


# === WYKRES 5: Średni błąd algorytmu na każdej mapie ===
avg_errors = data.groupby(["Algorytm", "Świat"])["error"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_errors, x="Algorytm", y="error", hue="Świat")
plt.title("Średni błąd położenia algorytmu w poszczególnych światach")
plt.ylabel("Średni błąd [m]")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "barplot_avg_error.png"))
plt.close()


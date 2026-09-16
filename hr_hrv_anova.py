import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

FILES = {
    "BPM": "bpm_hr.txt",
    "RMSSD": "rmssd.txt",
    "SDNN": "sdnn.txt",
    "TH": "new_TH_statistics.txt",
    "SF": "new_SF_statistics.txt",
}
SCR_FILE = "SCR_Peaks_N_and_SCR_Peaks_Amplitude_Mean.txt"
condition_map = {
    1: {"Frequency": "10", "Modality": "sound"},
    2: {"Frequency": "10", "Modality": "vibration"},
    3: {"Frequency": "10", "Modality": "light"},
    4: {"Frequency": "40", "Modality": "vibration"},
    5: {"Frequency": "10", "Modality": "sound+light"},
    6: {"Frequency": "nostim", "Modality": "nostim"},
    7: {"Frequency": "40", "Modality": "sound+light"},
    8: {"Frequency": "40", "Modality": "light"},
    9: {"Frequency": "40", "Modality": "sound"},
}

def parse_matrix(path):
    text = Path(path).read_text()
    matrix_text = text.split("average for each condition")[0]
    raw_rows = re.findall(r"\[([^\[\]]+)\]", matrix_text, flags=re.S)

    rows = []

    for raw in raw_rows:
        values = []

        for token in raw.replace("\n", " ").split():
            if token.lower() == "nan":
                values.append(np.nan)
            else:
                try:
                    values.append(float(token))
                except ValueError:
                    values = []
                    break

        if len(values) == 9:
            rows.append(values)

    matrix = np.asarray(rows, dtype=float)

    if matrix.shape != (19, 9):
        raise ValueError(f"Expected 19x9, got {matrix.shape}")

    return matrix
def parse_scr_matrix(path, start_text, end_text):

    text = Path(path).read_text()

    section = text.split(start_text)[1]
    section = section.split(end_text)[0]

    raw_rows = re.findall(
        r"\[([^\[\]]+)\]",
        section,
        flags=re.S
    )

    rows = []

    for raw in raw_rows:

        values = []

        for token in raw.replace("\n", " ").split():

            if token.lower() == "nan":
                values.append(np.nan)

            else:
                try:
                    values.append(float(token))

                except ValueError:
                    values = []
                    break

        if len(values) == 9:
            rows.append(values)

    matrix = np.asarray(rows, dtype=float)

    if matrix.shape != (19, 9):
        raise ValueError(f"Expected 19x9, got {matrix.shape}")

    return matrix

matrices = {
    name: parse_matrix(filename)
    for name, filename in FILES.items()
}
matrices["SCR_Peaks_N"] = parse_scr_matrix(
    SCR_FILE,
    "#######SCR_Peaks_N average for each condition and participant#######",
    "#######SCR_Peaks_N average for each condition#######"
)

matrices["SCR_Amplitude"] = parse_scr_matrix(
    SCR_FILE,
    "#######SCR_Peaks_Amplitude_Mean average for each condition and participant#######",
    "######SCR_Peaks_Amplitude_Mean average for each condition#######"
)
records = []

for i in range(19):
    for condition in range(1, 10):
        row = {
            "Participant": f"P{i+1}",
            "Condition": condition,
            "Frequency": condition_map[condition]["Frequency"],
            "Modality": condition_map[condition]["Modality"],
        }

        for measure, matrix in matrices.items():
            row[measure] = matrix[i, condition - 1]

        records.append(row)

df = pd.DataFrame(records)
df.to_csv("hr_hrv_long_data.csv", index=False)

stim_df = df[df["Condition"] != 6].copy()

anova_results = []

for measure in ["BPM", "RMSSD", "SDNN", "TH", "SF", "SCR_Peaks_N", "SCR_Amplitude"]:
    data = stim_df[
        ["Participant", "Frequency", "Modality", measure]
    ].dropna()

    counts = data.groupby("Participant").size()
    complete_participants = counts[counts == 8].index
    data = data[data["Participant"].isin(complete_participants)]

    result = AnovaRM(
        data=data,
        depvar=measure,
        subject="Participant",
        within=["Modality", "Frequency"]
    ).fit()

    print(f"\n{measure}")
    print(result)

    table = result.anova_table.reset_index().rename(
        columns={"index": "Effect"}
    )
    table.insert(0, "Measure", measure)
    anova_results.append(table)

anova_results_df = pd.concat(anova_results, ignore_index=True)
anova_results_df.to_csv("hr_hrv_anova_results.csv", index=False)

posthoc_results = []

for measure in ["BPM", "RMSSD", "SDNN", "SF"]:

    nostim = df[df["Condition"] == 6][
        ["Participant", measure]
    ].rename(columns={measure: "NoStim"})

    for condition in [1, 2, 3, 4, 5, 7, 8, 9]:

        current = df[df["Condition"] == condition][
            ["Participant", "Frequency", "Modality", measure]
        ].rename(columns={measure: "Stim"})

        paired = current.merge(
            nostim,
            on="Participant"
        ).dropna()

        t_stat, p_value = stats.ttest_rel(
            paired["Stim"],
            paired["NoStim"]
        )

        posthoc_results.append({
            "Measure": measure,
            "Condition": condition,
            "Frequency": paired["Frequency"].iloc[0],
            "Modality": paired["Modality"].iloc[0],
            "t": t_stat,
            "p": p_value
        })

posthoc_df = pd.DataFrame(posthoc_results)
posthoc_df.to_csv("hr_hrv_posthoc_vs_nostim.csv", index=False)

additional_posthoc_results = []

modalities = ["sound", "vibration", "light", "sound+light"]

bpm_modality = (
    df[df["Condition"] != 6]
    .groupby(["Participant", "Modality"])["BPM"]
    .mean()
    .reset_index()
)

for i in range(len(modalities)):

    for j in range(i + 1, len(modalities)):

        mod1 = modalities[i]
        mod2 = modalities[j]

        a = bpm_modality[bpm_modality["Modality"] == mod1][
            ["Participant", "BPM"]
        ].rename(columns={"BPM": "a"})

        b = bpm_modality[bpm_modality["Modality"] == mod2][
            ["Participant", "BPM"]
        ].rename(columns={"BPM": "b"})

        paired = a.merge(b, on="Participant").dropna()

        t_stat, p_value = stats.ttest_rel(
            paired["a"],
            paired["b"]
        )

        additional_posthoc_results.append({
            "Measure": "BPM",
            "Effect": "Modality",
            "Comparison": f"{mod1} vs {mod2}",
            "Mean_1": paired["a"].mean(),
            "Mean_2": paired["b"].mean(),
            "Difference": paired["a"].mean() - paired["b"].mean(),
            "t": t_stat,
            "p": p_value
        })

for modality in modalities:
    hz10 = df[
        (df["Modality"] == modality) &
        (df["Frequency"] == "10")
    ][["Participant", "BPM"]].rename(columns={"BPM": "hz10"})

    hz40 = df[
        (df["Modality"] == modality) &
        (df["Frequency"] == "40")
    ][["Participant", "BPM"]].rename(columns={"BPM": "hz40"})

    paired = hz10.merge(hz40, on="Participant").dropna()

    t_stat, p_value = stats.ttest_rel(
        paired["hz10"],
        paired["hz40"]
    )

    additional_posthoc_results.append({
    "Measure": "BPM",
    "Effect": "Modality:Frequency",
    "Comparison": f"{modality}: 10 Hz vs 40 Hz",
    "Mean_1": paired["hz10"].mean(),
    "Mean_2": paired["hz40"].mean(),
    "Difference": paired["hz10"].mean() - paired["hz40"].mean(),
    "t": t_stat,
    "p": p_value
})

sf_modality = (
    df[df["Condition"] != 6]
    .groupby(["Participant", "Modality"])["SF"]
    .mean()
    .reset_index()
)

for i in range(len(modalities)):
    for j in range(i + 1, len(modalities)):
        mod1 = modalities[i]
        mod2 = modalities[j]

        a = sf_modality[sf_modality["Modality"] == mod1][
            ["Participant", "SF"]
        ].rename(columns={"SF": "a"})

        b = sf_modality[sf_modality["Modality"] == mod2][
            ["Participant", "SF"]
        ].rename(columns={"SF": "b"})

        paired = a.merge(b, on="Participant").dropna()

        t_stat, p_value = stats.ttest_rel(
            paired["a"],
            paired["b"]
        )

        additional_posthoc_results.append({
    "Measure": "SF",
    "Effect": "Modality",
    "Comparison": f"{mod1} vs {mod2}",
    "Mean_1": paired["a"].mean(),
    "Mean_2": paired["b"].mean(),
    "Difference": paired["a"].mean() - paired["b"].mean(),
    "t": t_stat,
    "p": p_value
})

sf_frequency = (
    df[df["Condition"] != 6]
    .groupby(["Participant", "Frequency"])["SF"]
    .mean()
    .reset_index()
)

hz10 = sf_frequency[sf_frequency["Frequency"] == "10"][
    ["Participant", "SF"]
].rename(columns={"SF": "hz10"})

hz40 = sf_frequency[sf_frequency["Frequency"] == "40"][
    ["Participant", "SF"]
].rename(columns={"SF": "hz40"})

paired = hz10.merge(hz40, on="Participant").dropna()

t_stat, p_value = stats.ttest_rel(
    paired["hz10"],
    paired["hz40"]
)

additional_posthoc_results.append({
    "Measure": "SF",
    "Effect": "Frequency",
    "Comparison": "10 Hz vs 40 Hz",
    "Mean_1": paired["hz10"].mean(),
    "Mean_2": paired["hz40"].mean(),
    "Difference": paired["hz10"].mean() - paired["hz40"].mean(),
    "t": t_stat,
    "p": p_value
})

additional_posthoc_df = pd.DataFrame(additional_posthoc_results)

additional_posthoc_df.to_csv(
    "additional_posthoc_results.csv",
    index=False
)
import matplotlib.pyplot as plt

order = ["sound", "vibration", "light", "sound+light"]


bpm_plot = (
    stim_df.groupby(["Frequency", "Modality"])["BPM"]
    .agg(["mean", "sem"])
    .reset_index()
)

plt.figure()

for freq in [10, 40]:
    data = bpm_plot[bpm_plot["Frequency"] == freq].copy()
    data["Modality"] = pd.Categorical(
        data["Modality"],
        categories=order,
        ordered=True
    )
    data = data.sort_values("Modality")

    plt.errorbar(
        data["Modality"],
        data["mean"],
        yerr=data["sem"],
        marker="o",
        capsize=4,
        label=f"{freq} Hz"
    )

plt.xlabel("Modality")
plt.ylabel("Mean BPM")
plt.title("BPM by Modality and Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("bpm_interaction_plot.png", dpi=300)
plt.close()


b# BPM interaction
bpm_plot = (
    stim_df.groupby(["Frequency", "Modality"])["BPM"]
    .agg(["mean", "sem"])
    .reset_index()
)

order = ["sound", "vibration", "light", "sound+light"]

plt.figure()

for freq in bpm_plot["Frequency"].dropna().unique():
    data = bpm_plot[bpm_plot["Frequency"] == freq].copy()

    data["Modality"] = pd.Categorical(
        data["Modality"],
        categories=order,
        ordered=True
    )
    data = data.sort_values("Modality")

    label = str(freq)
    if "Hz" not in label:
        label += " Hz"

    plt.errorbar(
        data["Modality"],
        data["mean"],
        yerr=data["sem"],
        marker="o",
        capsize=4,
        label=label
    )

plt.xlabel("Modality")
plt.ylabel("Mean BPM")
plt.title("BPM by Modality and Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("bpm_interaction_plot.png", dpi=300)
plt.close()

sf_modality = (
    stim_df.groupby(["Participant", "Modality"])["SF"]
    .mean()
    .reset_index()
    .groupby("Modality")["SF"]
    .agg(["mean", "sem"])
    .reindex(order)
)

plt.figure()

plt.bar(
    sf_modality.index,
    sf_modality["mean"],
    yerr=sf_modality["sem"],
    capsize=4
)

plt.xlabel("Modality")
plt.ylabel("Mean SF")
plt.title("SF by Modality")
plt.tight_layout()
plt.savefig("sf_modality_plot.png", dpi=300)
plt.close()


sf_frequency = (
    stim_df.groupby(["Participant", "Frequency"])["SF"]
    .mean()
    .reset_index()
    .groupby("Frequency")["SF"]
    .agg(["mean", "sem"])
)

plt.figure()

plt.bar(
    sf_frequency.index.astype(str),
    sf_frequency["mean"],
    yerr=sf_frequency["sem"],
    capsize=4
)

plt.xlabel("Frequency")
plt.ylabel("Mean SF")
plt.title("SF by Frequency")
plt.tight_layout()
plt.savefig("sf_frequency_plot.png", dpi=300)
plt.close()
condition_labels = {
    1: "10 Hz sound",
    2: "10 Hz vibration",
    3: "10 Hz light",
    4: "40 Hz vibration",
    5: "10 Hz sound+light",
    7: "40 Hz sound+light",
    8: "40 Hz light",
    9: "40 Hz sound"
}

conditions = [1, 2, 3, 4, 5, 7, 8, 9]

for measure in ["BPM", "RMSSD", "SDNN", "SF"]:
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes = axes.flatten()
if measure == "BPM":
    y_min, y_max = 50, 95
elif measure == "RMSSD":
    y_min, y_max = None, None
elif measure == "SDNN":
    y_min, y_max = 20, 180
elif measure == "SF":
    y_min, y_max = 0, 35
    nostim = df[df["Condition"] == 6][["Participant", measure]].rename(
        columns={measure: "NoStim"}
    )

    for ax, condition in zip(axes, conditions):
        stim = df[df["Condition"] == condition][["Participant", measure]].rename(
            columns={measure: "Stim"}
        )

        paired = stim.merge(nostim, on="Participant").dropna()

        for _, row in paired.iterrows():
            ax.plot(
                [0, 1],
                [row["NoStim"], row["Stim"]],
                marker="o",
                alpha=0.4
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No stim", "Stim"])
        ax.set_title(condition_labels[condition])

        if ax in axes[::4]:
            ax.set_ylabel(measure)

    fig.suptitle(f"{measure}: Stimulation vs No Stimulation")
    plt.tight_layout()
    plt.savefig(f"{measure.lower()}_stim_vs_nostim.png", dpi=300)
    plt.close()
print("\nDone.")
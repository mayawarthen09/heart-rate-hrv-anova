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

for measure in ["BPM", "RMSSD", "SDNN", "TH", "SF", "SCR_Amplitude"]:

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
import numpy as np
import matplotlib.pyplot as plt

order = ["sound", "vibration", "light", "sound+light"]

condition_order = [6, 1, 9, 2, 4, 3, 8, 5, 7]

condition_labels = {
    6: "No stim",
    1: "10 Hz\nSound",
    9: "40 Hz\nSound",
    2: "10 Hz\nVibration",
    4: "40 Hz\nVibration",
    3: "10 Hz\nLight",
    8: "40 Hz\nLight",
    5: "10 Hz\nSound + Light",
    7: "40 Hz\nSound + Light"
}


def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=1.2
    )

    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=12
    )


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def finish_plot(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()


bpm_participant_modality = (
    stim_df.groupby(["Participant", "Modality"])["BPM"]
    .mean()
    .reset_index()
)

bpm_modality = (
    bpm_participant_modality
    .groupby("Modality")["BPM"]
    .agg(["mean", "sem"])
    .reindex(order)
)

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(order))

ax.bar(
    x,
    bpm_modality["mean"],
    yerr=bpm_modality["sem"],
    capsize=5,
    alpha=0.75,
    edgecolor="black"
)

for i, modality in enumerate(order):
    values = bpm_participant_modality[
        bpm_participant_modality["Modality"] == modality
    ]["BPM"].dropna().values

    jitter = np.linspace(-0.12, 0.12, len(values))

    ax.scatter(
        np.full(len(values), i) + jitter,
        values,
        color="black",
        alpha=0.6,
        s=28,
        zorder=3
    )

ax.set_xticks(x)
ax.set_xticklabels(["Sound", "Vibration", "Light", "Sound + Light"])
ax.set_ylabel("Mean BPM")
ax.set_title("Heart Rate by Modality")

y = max(bpm_modality["mean"] + bpm_modality["sem"]) + 2

add_sig_bracket(
    ax,
    0,
    1,
    y,
    0.5,
    sig_stars(0.03933)
)

ax.set_ylim(top=y + 3)

finish_plot(ax)

plt.savefig("bpm_modality_plot.png", dpi=300, bbox_inches="tight")
plt.close()


interaction_df = stim_df.copy()

interaction_df["Frequency_clean"] = (
    interaction_df["Frequency"]
    .astype(str)
    .str.replace(" Hz", "", regex=False)
)

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(order))
width = 0.32

all_bpm_values = []

for j, freq in enumerate(["10", "40"]):
    means = []
    sems = []

    for modality in order:
        values = interaction_df[
            (interaction_df["Frequency_clean"] == freq) &
            (interaction_df["Modality"] == modality)
        ]["BPM"].dropna().values

        means.append(values.mean())
        sems.append(pd.Series(values).sem())
        all_bpm_values.extend(values)

    offset = -width / 2 if freq == "10" else width / 2
    xpos = x + offset

    ax.bar(
        xpos,
        means,
        width=width,
        yerr=sems,
        capsize=4,
        alpha=0.75,
        edgecolor="black",
        label=f"{freq} Hz"
    )

    for i, modality in enumerate(order):
        values = interaction_df[
            (interaction_df["Frequency_clean"] == freq) &
            (interaction_df["Modality"] == modality)
        ]["BPM"].dropna().values

        jitter = np.linspace(-0.04, 0.04, len(values))

        ax.scatter(
            np.full(len(values), xpos[i]) + jitter,
            values,
            color="black",
            alpha=0.45,
            s=20,
            zorder=3
        )

ax.set_xticks(x)
ax.set_xticklabels(["Sound", "Vibration", "Light", "Sound + Light"])
ax.set_ylabel("BPM")
ax.set_title("Heart Rate by Modality and Frequency")
ax.legend(frameon=False)

y = max(all_bpm_values) + 3

add_sig_bracket(
    ax,
    x[3] - width / 2,
    x[3] + width / 2,
    y,
    0.5,
    sig_stars(0.000127)
)

ax.set_ylim(top=y + 3)

finish_plot(ax)

plt.savefig("bpm_interaction_plot.png", dpi=300, bbox_inches="tight")
plt.close()


sf_participant_modality = (
    stim_df.groupby(["Participant", "Modality"])["SF"]
    .mean()
    .reset_index()
)

sf_modality = (
    sf_participant_modality
    .groupby("Modality")["SF"]
    .agg(["mean", "sem"])
    .reindex(order)
)

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(order))

ax.bar(
    x,
    sf_modality["mean"],
    yerr=sf_modality["sem"],
    capsize=5,
    alpha=0.75,
    edgecolor="black"
)

for i, modality in enumerate(order):
    values = sf_participant_modality[
        sf_participant_modality["Modality"] == modality
    ]["SF"].dropna().values

    jitter = np.linspace(-0.12, 0.12, len(values))

    ax.scatter(
        np.full(len(values), i) + jitter,
        values,
        color="black",
        alpha=0.6,
        s=28,
        zorder=3
    )

ax.set_xticks(x)
ax.set_xticklabels(["Sound", "Vibration", "Light", "Sound + Light"])
ax.set_ylabel("SCR Frequency")
ax.set_title("Skin Conductance Response Frequency by Modality")

y1 = max(sf_modality["mean"] + sf_modality["sem"]) + 1

add_sig_bracket(
    ax,
    0,
    3,
    y1,
    0.3,
    sig_stars(0.00860)
)

add_sig_bracket(
    ax,
    1,
    3,
    y1 + 1.2,
    0.3,
    sig_stars(0.00760)
)

ax.set_ylim(top=y1 + 3)

finish_plot(ax)

plt.savefig("sf_modality_plot.png", dpi=300, bbox_inches="tight")
plt.close()


sf_freq_df = stim_df.copy()

sf_freq_df["Frequency_clean"] = (
    sf_freq_df["Frequency"]
    .astype(str)
    .str.replace(" Hz", "", regex=False)
)

sf_participant_frequency = (
    sf_freq_df
    .groupby(["Participant", "Frequency_clean"])["SF"]
    .mean()
    .reset_index()
)

frequency_order = ["10", "40"]

sf_frequency = (
    sf_participant_frequency
    .groupby("Frequency_clean")["SF"]
    .agg(["mean", "sem"])
    .reindex(frequency_order)
)

fig, ax = plt.subplots(figsize=(6, 6))

x = np.arange(2)

ax.bar(
    x,
    sf_frequency["mean"],
    yerr=sf_frequency["sem"],
    capsize=5,
    alpha=0.75,
    edgecolor="black"
)

for i, freq in enumerate(frequency_order):
    values = sf_participant_frequency[
        sf_participant_frequency["Frequency_clean"] == freq
    ]["SF"].dropna().values

    jitter = np.linspace(-0.08, 0.08, len(values))

    ax.scatter(
        np.full(len(values), i) + jitter,
        values,
        color="black",
        alpha=0.6,
        s=28,
        zorder=3
    )

ax.set_xticks(x)
ax.set_xticklabels(["10 Hz", "40 Hz"])
ax.set_ylabel("SCR Frequency")
ax.set_title("Skin Conductance Response Frequency by Frequency")

y = max(sf_frequency["mean"] + sf_frequency["sem"]) + 1

add_sig_bracket(
    ax,
    0,
    1,
    y,
    0.3,
    sig_stars(0.04318)
)

ax.set_ylim(top=y + 2)

finish_plot(ax)

plt.savefig("sf_frequency_plot.png", dpi=300, bbox_inches="tight")
plt.close()


def plot_all_conditions(
    measure,
    ylabel,
    title,
    filename,
    significant_pairs=None
):
    summary = (
        df.groupby("Condition")[measure]
        .agg(["mean", "sem"])
        .reindex(condition_order)
    )

    x = np.arange(len(condition_order))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x,
        summary["mean"],
        yerr=summary["sem"],
        capsize=5,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8
    )

    all_values = []

    for i, condition in enumerate(condition_order):
        values = (
            df[df["Condition"] == condition][measure]
            .dropna()
            .values
        )

        all_values.extend(values)

        jitter = np.linspace(-0.18, 0.18, len(values))

        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=30,
            alpha=0.65,
            color="black",
            zorder=3
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [condition_labels[c] for c in condition_order],
        rotation=25,
        ha="right"
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if significant_pairs:
        data_max = max(all_values)
        data_min = min(all_values)
        data_range = data_max - data_min

        if data_range == 0:
            data_range = 1

        y = data_max + 0.08 * data_range
        h = 0.03 * data_range
        step = 0.10 * data_range

        for x1, x2, p in significant_pairs:
            add_sig_bracket(
                ax,
                x1,
                x2,
                y,
                h,
                sig_stars(p)
            )

            y += step

        current_bottom, _ = ax.get_ylim()

        ax.set_ylim(
            current_bottom,
            y + step
        )

    finish_plot(ax)

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


plot_all_conditions(
    measure="BPM",
    ylabel="BPM",
    title="Heart Rate by Condition",
    filename="bpm_stim_vs_nostim.png"
)

plot_all_conditions(
    measure="RMSSD",
    ylabel="RMSSD",
    title="RMSSD by Condition",
    filename="rmssd_stim_vs_nostim.png"
)

plot_all_conditions(
    measure="SDNN",
    ylabel="SDNN",
    title="SDNN by Condition",
    filename="sdnn_stim_vs_nostim.png"
)

plot_all_conditions(
    measure="TH",
    ylabel="Temperature",
    title="Temperature by Condition",
    filename="th_stim_vs_nostim.png",
    significant_pairs=[
        (0, 2, 0.02836)
    ]
)

plot_all_conditions(
    measure="SF",
    ylabel="SCR Frequency",
    title="Skin Conductance Response Frequency by Condition",
    filename="sf_stim_vs_nostim.png",
    significant_pairs=[
        (0, 3, 0.00471),
        (0, 5, 0.04769)
    ]
)

plot_all_conditions(
    measure="SCR_Amplitude",
    ylabel="SCR Amplitude",
    title="Skin Conductance Response Amplitude by Condition",
    filename="scr_amplitude_stim_vs_nostim.png"
)
bpm_line = (
    stim_df.groupby(["Frequency", "Modality"])["BPM"]
    .agg(["mean", "sem"])
    .reset_index()
)

bpm_line["Frequency_clean"] = (
    bpm_line["Frequency"]
    .astype(str)
    .str.replace(" Hz", "", regex=False)
)

fig, ax = plt.subplots(figsize=(8, 6))

for freq in ["10", "40"]:
    data = bpm_line[bpm_line["Frequency_clean"] == freq].copy()

    data["Modality"] = pd.Categorical(
        data["Modality"],
        categories=order,
        ordered=True
    )

    data = data.sort_values("Modality")

    ax.errorbar(
        np.arange(len(order)),
        data["mean"],
        yerr=data["sem"],
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=4,
        label=f"{freq} Hz"
    )

ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(["Sound", "Vibration", "Light", "Sound + Light"])
ax.set_ylabel("BPM")
ax.set_title("Heart Rate by Modality and Frequency")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bpm_y = bpm_line["mean"].max() + bpm_line["sem"].max() + 1.5

add_sig_bracket(
    ax,
    2.92,
    3.08,
    bpm_y,
    0.3,
    sig_stars(0.000127)
)

ax.set_ylim(top=bpm_y + 2)

plt.tight_layout()
plt.savefig("bpm_interaction_line_plot.png", dpi=300, bbox_inches="tight")
plt.close()


sf_line = (
    stim_df.groupby(["Frequency", "Modality"])["SF"]
    .agg(["mean", "sem"])
    .reset_index()
)

sf_line["Frequency_clean"] = (
    sf_line["Frequency"]
    .astype(str)
    .str.replace(" Hz", "", regex=False)
)

fig, ax = plt.subplots(figsize=(8, 6))

for freq in ["10", "40"]:
    data = sf_line[sf_line["Frequency_clean"] == freq].copy()

    data["Modality"] = pd.Categorical(
        data["Modality"],
        categories=order,
        ordered=True
    )

    data = data.sort_values("Modality")

    ax.errorbar(
        np.arange(len(order)),
        data["mean"],
        yerr=data["sem"],
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=4,
        label=f"{freq} Hz"
    )

ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(["Sound", "Vibration", "Light", "Sound + Light"])
ax.set_ylabel("SCR Frequency")
ax.set_title("Skin Conductance Response Frequency by Modality and Frequency")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("sf_interaction_line_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nDone.")
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

    bars = ax.bar(
        x,
        summary["mean"],
        yerr=summary["sem"],
        capsize=5,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8
    )

    bars[0].set_alpha(0.4)
    bars[0].set_hatch("//")

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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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

        bottom, _ = ax.get_ylim()

        ax.set_ylim(
            bottom,
            y + step
        )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


plot_all_conditions(
    measure="BPM",
    ylabel="Heart Rate (BPM)",
    title="Heart Rate by Stimulation Condition",
    filename="bpm_all_conditions.png"
)

plot_all_conditions(
    measure="RMSSD",
    ylabel="RMSSD",
    title="RMSSD by Stimulation Condition",
    filename="rmssd_all_conditions.png"
)

plot_all_conditions(
    measure="SDNN",
    ylabel="SDNN",
    title="SDNN by Stimulation Condition",
    filename="sdnn_all_conditions.png"
)

plot_all_conditions(
    measure="TH",
    ylabel="Temperature",
    title="Temperature by Stimulation Condition",
    filename="temperature_all_conditions.png",
    significant_pairs=[
        (0, 2, 0.02836)
    ]
)

plot_all_conditions(
    measure="SF",
    ylabel="SCR Frequency",
    title="Skin Conductance Response Frequency by Stimulation Condition",
    filename="scr_frequency_all_conditions.png",
    significant_pairs=[
        (0, 3, 0.00471),
        (0, 5, 0.04769)
    ]
)

plot_all_conditions(
    measure="SCR_Peaks_N",
    ylabel="SCR Peak Count",
    title="SCR Peak Count by Stimulation Condition",
    filename="scr_peaks_all_conditions.png"
)

plot_all_conditions(
    measure="SCR_Amplitude",
    ylabel="SCR Amplitude",
    title="Skin Conductance Response Amplitude by Stimulation Condition",
    filename="scr_amplitude_all_conditions.png"
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
pca_measures = [
    "BPM",
    "RMSSD",
    "SDNN",
    "TH",
    "SF",
    "SCR_Amplitude"
]

pca_df = df[
    ["Participant", "Condition", "Frequency", "Modality"] + pca_measures
].dropna().copy()

X = pca_df[pca_measures].astype(float)

X_z = (X - X.mean()) / X.std(ddof=0)

U, S, Vt = np.linalg.svd(X_z, full_matrices=False)

scores = U * S

explained_variance = (S ** 2) / (len(X_z) - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()

loadings = Vt.T

pca_df["PC1"] = scores[:, 0]
pca_df["PC2"] = scores[:, 1]

pca_scores_output = pca_df[
    [
        "Participant",
        "Condition",
        "Frequency",
        "Modality",
        "PC1",
        "PC2"
    ]
]

pca_scores_output.to_csv(
    "physiology_pca_scores.csv",
    index=False
)

loading_df = pd.DataFrame(
    {
        "Measure": pca_measures,
        "PC1": loadings[:, 0],
        "PC2": loadings[:, 1]
    }
)

loading_df.to_csv(
    "physiology_pca_loadings.csv",
    index=False
)

variance_df = pd.DataFrame(
    {
        "Component": [
            f"PC{i + 1}"
            for i in range(len(explained_variance_ratio))
        ],
        "Explained_Variance": explained_variance_ratio
    }
)

variance_df.to_csv(
    "physiology_pca_variance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(
    np.arange(len(pca_measures)) - 0.18,
    loading_df["PC1"],
    width=0.36,
    label="PC1"
)

ax.bar(
    np.arange(len(pca_measures)) + 0.18,
    loading_df["PC2"],
    width=0.36,
    label="PC2"
)

ax.axhline(0, linewidth=0.8)

ax.set_xticks(np.arange(len(pca_measures)))
ax.set_xticklabels(
    [
        "BPM",
        "RMSSD",
        "SDNN",
        "Temperature",
        "SCR Frequency",
        "SCR Amplitude"
    ],
    rotation=25,
    ha="right"
)

ax.set_ylabel("Loading")
ax.set_title("Physiological PCA Loadings")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "physiology_pca_loadings.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))

for condition in condition_order:
    data = pca_df[pca_df["Condition"] == condition]

    ax.scatter(
        data["PC1"],
        data["PC2"],
        alpha=0.65,
        label=condition_labels[condition].replace("\n", " ")
    )

ax.set_xlabel(
    f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)"
)
ax.set_ylabel(
    f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)"
)

ax.set_title("Combined Physiological Response")
ax.legend(
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "physiology_pca_scores.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))

components_to_show = min(
    6,
    len(explained_variance_ratio)
)

ax.bar(
    np.arange(1, components_to_show + 1),
    explained_variance_ratio[:components_to_show] * 100
)

ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax.set_title("PCA Explained Variance")
ax.set_xticks(
    np.arange(1, components_to_show + 1)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "physiology_pca_variance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


bpm_by_modality = (
    stim_df
    .groupby(["Participant", "Modality"])["BPM"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(order))

for participant in sorted(
    bpm_by_modality["Participant"].unique()
):
    pdata = (
        bpm_by_modality[
            bpm_by_modality["Participant"] == participant
        ]
        .set_index("Modality")
        .reindex(order)
    )

    ax.plot(
        x,
        pdata["BPM"],
        marker="o",
        alpha=0.55,
        linewidth=1.2
    )

ax.set_xticks(x)
ax.set_xticklabels(
    ["Sound", "Vibration", "Light", "Sound + Light"]
)

ax.set_ylabel("BPM")
ax.set_title("Individual Heart Rate Across Modalities")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "bpm_participant_trajectories_modality.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


bpm_condition_df = df[
    ["Participant", "Condition", "BPM"]
].dropna().copy()

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(condition_order))

for participant in sorted(
    bpm_condition_df["Participant"].unique()
):
    pdata = (
        bpm_condition_df[
            bpm_condition_df["Participant"] == participant
        ]
        .set_index("Condition")
        .reindex(condition_order)
    )

    ax.plot(
        x,
        pdata["BPM"],
        marker="o",
        alpha=0.5,
        linewidth=1.1
    )

ax.set_xticks(x)

ax.set_xticklabels(
    [
        condition_labels[c]
        for c in condition_order
    ],
    rotation=25,
    ha="right"
)

ax.set_ylabel("BPM")
ax.set_title("Individual Heart Rate Across Conditions")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "bpm_participant_trajectories_conditions.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


highlight_participant = 1

fig, ax = plt.subplots(figsize=(9, 6))

for participant in sorted(
    bpm_by_modality["Participant"].unique()
):
    pdata = (
        bpm_by_modality[
            bpm_by_modality["Participant"] == participant
        ]
        .set_index("Modality")
        .reindex(order)
    )

    if participant == highlight_participant:
        ax.plot(
            x[:4],
            pdata["BPM"],
            marker="o",
            linewidth=3,
            label=f"Participant {participant}"
        )
    else:
        ax.plot(
            x[:4],
            pdata["BPM"],
            alpha=0.2,
            linewidth=1
        )

ax.set_xticks(x[:4])

ax.set_xticklabels(
    ["Sound", "Vibration", "Light", "Sound + Light"]
)

ax.set_ylabel("BPM")
ax.set_title(
    f"Heart Rate Trajectory: Participant {highlight_participant}"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "bpm_highlighted_participant.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
bpm_control = (
    df[df["Condition"] == 6][["Participant", "BPM"]]
    .rename(columns={"BPM": "NoStim_BPM"})
)

bpm_delta_df = (
    df[["Participant", "Condition", "BPM"]]
    .merge(bpm_control, on="Participant")
)

bpm_delta_df["Delta_BPM"] = (
    bpm_delta_df["BPM"] - bpm_delta_df["NoStim_BPM"]
)

bpm_delta_df.to_csv(
    "bpm_change_from_nostim.csv",
    index=False
)


fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(condition_order))

for participant in sorted(
    bpm_delta_df["Participant"].unique()
):
    pdata = (
        bpm_delta_df[
            bpm_delta_df["Participant"] == participant
        ]
        .set_index("Condition")
        .reindex(condition_order)
    )

    ax.plot(
        x,
        pdata["Delta_BPM"],
        marker="o",
        alpha=0.55,
        linewidth=1.1
    )

ax.axhline(0, linewidth=1)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        condition_labels[c]
        for c in condition_order
    ],
    rotation=25,
    ha="right"
)

ax.set_ylabel("Change in BPM from No Stimulation")
ax.set_title("Individual Heart Rate Change from No-Stimulation Control")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "bpm_change_from_nostim_trajectories.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


participant_bpm_summary = (
    df.groupby("Participant")["BPM"]
    .mean()
    .reset_index(name="Mean_BPM")
)

participant_bpm_summary = participant_bpm_summary.merge(
    bpm_control,
    on="Participant"
)

participant_bpm_summary.to_csv(
    "participant_bpm_summary.csv",
    index=False
)


fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    participant_bpm_summary["NoStim_BPM"],
    participant_bpm_summary["Mean_BPM"],
    s=60,
    alpha=0.75
)

for _, row in participant_bpm_summary.iterrows():
    ax.text(
        row["NoStim_BPM"] + 0.15,
        row["Mean_BPM"] + 0.15,
        row["Participant"],
        fontsize=8
    )

ax.set_xlabel("No-Stimulation BPM")
ax.set_ylabel("Mean BPM Across All Conditions")
ax.set_title("Participant Baseline Heart Rate vs Overall Heart Rate")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "bpm_baseline_vs_overall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
from scipy.stats import pearsonr

baseline_corr_df = (
    df.groupby("Participant")["BPM"]
    .mean()
    .reset_index(name="Mean_BPM")
    .merge(
        df[df["Condition"] == 6][["Participant", "BPM"]]
        .rename(columns={"BPM": "NoStim_BPM"}),
        on="Participant"
    )
)

r_value, p_value = pearsonr(
    baseline_corr_df["NoStim_BPM"],
    baseline_corr_df["Mean_BPM"]
)

print("\nBaseline BPM vs Mean BPM")
print(f"r = {r_value:.3f}")
print(f"p = {p_value:.5f}")


responder_summary = (
    bpm_delta_df[bpm_delta_df["Condition"] != 6]
    .assign(Abs_Delta_BPM=lambda d: d["Delta_BPM"].abs())
    .groupby("Participant")["Abs_Delta_BPM"]
    .mean()
    .reset_index(name="Mean_Absolute_Delta_BPM")
)

responder_summary = responder_summary.sort_values(
    "Mean_Absolute_Delta_BPM",
    ascending=False
)

responder_summary.to_csv(
    "bpm_responder_summary.csv",
    index=False
)

print("\nMean absolute BPM change by participant")
print(responder_summary)


fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    responder_summary["Participant"],
    responder_summary["Mean_Absolute_Delta_BPM"]
)

ax.set_xlabel("Participant")
ax.set_ylabel("Mean Absolute Change in BPM")
ax.set_title("Magnitude of Heart Rate Response by Participant")

ax.tick_params(
    axis="x",
    rotation=45
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    "bpm_response_magnitude_by_participant.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
print("\nDone.")
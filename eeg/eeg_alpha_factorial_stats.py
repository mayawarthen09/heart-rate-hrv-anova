from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


data_file = Path(
    "eeg_outputs/all_participants_alpha_power.csv"
)

output_folder = Path(
    "eeg_outputs/alpha_factorial_statistics"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


condition_info = {
    1: ("Sound", 10),
    2: ("Vibration", 10),
    3: ("Light", 10),
    4: ("Vibration", 40),
    5: ("Sound + Light", 10),
    7: ("Sound + Light", 40),
    8: ("Light", 40),
    9: ("Sound", 40)
}


condition_labels = {
    1: "10 Hz Sound",
    2: "10 Hz Vibration",
    3: "10 Hz Light",
    4: "40 Hz Vibration",
    5: "10 Hz Sound + Light",
    6: "No Stimulation",
    7: "40 Hz Sound + Light",
    8: "40 Hz Light",
    9: "40 Hz Sound"
}


df = pd.read_csv(data_file)


print("\nData loaded:")
print(df.head())

print("\nParticipants:")
print(df["Participant"].nunique())


# --------------------------------------------------
# FACTORIAL ANOVA
# Exclude Condition 6 because it has no modality/frequency cell
# --------------------------------------------------

stim_df = df[
    df["Condition"] != 6
].copy()


stim_df["Modality"] = stim_df[
    "Condition"
].map(
    lambda x: condition_info[x][0]
)


stim_df["Frequency"] = stim_df[
    "Condition"
].map(
    lambda x: str(condition_info[x][1])
)


print("\nFactorial data:")
print(
    stim_df[
        [
            "Participant",
            "Condition",
            "Modality",
            "Frequency",
            "Difference_dB"
        ]
    ].head(10)
)


anova = AnovaRM(
    data=stim_df,
    depvar="Difference_dB",
    subject="Participant",
    within=[
        "Modality",
        "Frequency"
    ]
).fit()


print("\n" + "=" * 70)
print("ALPHA MODALITY x FREQUENCY REPEATED-MEASURES ANOVA")
print("=" * 70)

print(anova)


with open(
    output_folder / "alpha_factorial_anova.txt",
    "w"
) as file:
    file.write(str(anova))


# --------------------------------------------------
# SUMMARY TABLE
# --------------------------------------------------

summary = (
    stim_df
    .groupby(
        [
            "Modality",
            "Frequency"
        ]
    )["Difference_dB"]
    .agg(
        Mean="mean",
        SD="std",
        N="count"
    )
    .reset_index()
)


summary["SEM"] = (
    summary["SD"]
    / np.sqrt(summary["N"])
)


print("\nCondition means:")
print(summary)


summary.to_csv(
    output_folder
    / "alpha_modality_frequency_summary.csv",
    index=False
)


# --------------------------------------------------
# SIMPLE FREQUENCY EFFECTS
# 10 Hz vs 40 Hz inside each modality
# --------------------------------------------------

frequency_tests = []


for modality in [
    "Sound",
    "Vibration",
    "Light",
    "Sound + Light"
]:

    modality_df = stim_df[
        stim_df["Modality"] == modality
    ]

    wide = modality_df.pivot(
        index="Participant",
        columns="Frequency",
        values="Difference_dB"
    ).dropna()


    t_stat, p_value = ttest_rel(
        wide["10"],
        wide["40"]
    )


    frequency_tests.append({
        "Modality": modality,
        "Mean_10Hz": wide["10"].mean(),
        "Mean_40Hz": wide["40"].mean(),
        "Mean_Difference_10_minus_40":
            wide["10"].mean()
            - wide["40"].mean(),
        "t": t_stat,
        "p_uncorrected": p_value,
        "N": len(wide)
    })


frequency_df = pd.DataFrame(
    frequency_tests
)


reject, corrected_p, _, _ = multipletests(
    frequency_df["p_uncorrected"],
    method="holm"
)


frequency_df["p_holm"] = corrected_p
frequency_df["Significant_Holm"] = reject


print("\n" + "=" * 70)
print("10 Hz vs 40 Hz WITHIN EACH MODALITY")
print("=" * 70)

print(frequency_df)


frequency_df.to_csv(
    output_folder
    / "alpha_frequency_within_modality.csv",
    index=False
)


# --------------------------------------------------
# MODALITY PAIRWISE COMPARISONS
# Average 10 + 40 Hz within each modality first
# --------------------------------------------------

modality_means = (
    stim_df
    .groupby(
        [
            "Participant",
            "Modality"
        ]
    )["Difference_dB"]
    .mean()
    .reset_index()
)


modality_wide = modality_means.pivot(
    index="Participant",
    columns="Modality",
    values="Difference_dB"
)


modalities = [
    "Sound",
    "Vibration",
    "Light",
    "Sound + Light"
]


modality_tests = []


for i in range(len(modalities)):

    for j in range(
        i + 1,
        len(modalities)
    ):

        modality_a = modalities[i]
        modality_b = modalities[j]

        paired = modality_wide[
            [
                modality_a,
                modality_b
            ]
        ].dropna()


        t_stat, p_value = ttest_rel(
            paired[modality_a],
            paired[modality_b]
        )


        modality_tests.append({
            "Modality_A": modality_a,
            "Modality_B": modality_b,
            "Mean_A":
                paired[modality_a].mean(),
            "Mean_B":
                paired[modality_b].mean(),
            "Mean_Difference":
                paired[modality_a].mean()
                - paired[modality_b].mean(),
            "t": t_stat,
            "p_uncorrected": p_value,
            "N": len(paired)
        })


modality_df = pd.DataFrame(
    modality_tests
)


reject, corrected_p, _, _ = multipletests(
    modality_df["p_uncorrected"],
    method="holm"
)


modality_df["p_holm"] = corrected_p
modality_df["Significant_Holm"] = reject


print("\n" + "=" * 70)
print("PAIRWISE MODALITY COMPARISONS")
print("=" * 70)

print(modality_df)


modality_df.to_csv(
    output_folder
    / "alpha_modality_pairwise.csv",
    index=False
)


# --------------------------------------------------
# STIMULATION CONDITIONS VS NO STIMULATION
# --------------------------------------------------

wide_conditions = df.pivot(
    index="Participant",
    columns="Condition",
    values="Difference_dB"
)


no_stim_tests = []


for condition in [
    1, 2, 3, 4, 5, 7, 8, 9
]:

    paired = wide_conditions[
        [
            condition,
            6
        ]
    ].dropna()


    t_stat, p_value = ttest_rel(
        paired[condition],
        paired[6]
    )


    no_stim_tests.append({
        "Condition": condition,
        "Condition_Name":
            condition_labels[condition],
        "Mean_Stim":
            paired[condition].mean(),
        "Mean_NoStim":
            paired[6].mean(),
        "Mean_Difference_Stim_minus_NoStim":
            paired[condition].mean()
            - paired[6].mean(),
        "t": t_stat,
        "p_uncorrected": p_value,
        "N": len(paired)
    })


no_stim_df = pd.DataFrame(
    no_stim_tests
)


reject, corrected_p, _, _ = multipletests(
    no_stim_df["p_uncorrected"],
    method="holm"
)


no_stim_df["p_holm"] = corrected_p
no_stim_df["Significant_Holm"] = reject


print("\n" + "=" * 70)
print("STIMULATION VS NO STIMULATION")
print("=" * 70)

print(no_stim_df)


no_stim_df.to_csv(
    output_folder
    / "alpha_stim_vs_no_stim.csv",
    index=False
)


# --------------------------------------------------
# INTERACTION GRAPH
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 6)
)


for modality in modalities:

    modality_summary = (
        summary[
            summary["Modality"]
            == modality
        ]
        .copy()
    )


    modality_summary[
        "Frequency_num"
    ] = modality_summary[
        "Frequency"
    ].astype(int)


    modality_summary = (
        modality_summary
        .sort_values(
            "Frequency_num"
        )
    )


    ax.errorbar(
        modality_summary[
            "Frequency_num"
        ],
        modality_summary[
            "Mean"
        ],
        yerr=modality_summary[
            "SEM"
        ],
        marker="o",
        capsize=4,
        label=modality
    )


ax.axhline(
    0,
    linewidth=1
)


ax.set_xticks(
    [10, 40]
)


ax.set_xlabel(
    "Stimulation Frequency (Hz)"
)


ax.set_ylabel(
    "Alpha Power Change (Post − Pre, dB)"
)


ax.set_title(
    "Alpha Power Change by Modality and Frequency"
)


ax.legend(
    title="Modality"
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


plt.tight_layout()


plt.savefig(
    output_folder
    / "alpha_modality_frequency_interaction.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nFinished.")

print(
    "Results saved in:",
    output_folder.resolve()
)
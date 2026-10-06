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
    "eeg_outputs/alpha_statistics"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


condition_order = [
    1, 2, 3, 4, 5, 6, 7, 8, 9
]

condition_labels = {
    1: "10 Hz\nSound",
    2: "10 Hz\nVibration",
    3: "10 Hz\nLight",
    4: "40 Hz\nVibration",
    5: "10 Hz\nSound + Light",
    6: "No\nStimulation",
    7: "40 Hz\nSound + Light",
    8: "40 Hz\nLight",
    9: "40 Hz\nSound"
}


df = pd.read_csv(data_file)

print("\nData loaded:")
print(df.head())

print("\nNumber of participants:")
print(df["Participant"].nunique())

print("\nRows per condition:")
print(df.groupby("Condition")["Participant"].nunique())


alpha_df = df[
    [
        "Participant",
        "Condition",
        "Condition_Name",
        "Difference_dB"
    ]
].dropna().copy()


print("\nMean alpha change by condition:")

condition_summary = (
    alpha_df
    .groupby(
        ["Condition", "Condition_Name"]
    )["Difference_dB"]
    .agg(
        Mean="mean",
        SD="std",
        N="count"
    )
    .reset_index()
)

condition_summary["SEM"] = (
    condition_summary["SD"]
    / np.sqrt(condition_summary["N"])
)

print(condition_summary)

condition_summary.to_csv(
    output_folder
    / "alpha_condition_summary.csv",
    index=False
)


anova = AnovaRM(
    data=alpha_df,
    depvar="Difference_dB",
    subject="Participant",
    within=["Condition"]
).fit()


print("\n" + "=" * 60)
print("REPEATED-MEASURES ANOVA")
print("=" * 60)

print(anova)


with open(
    output_folder / "alpha_anova.txt",
    "w"
) as file:

    file.write(str(anova))


wide = alpha_df.pivot(
    index="Participant",
    columns="Condition",
    values="Difference_dB"
)


comparisons = []


for i in range(len(condition_order)):

    for j in range(
        i + 1,
        len(condition_order)
    ):

        condition_a = condition_order[i]
        condition_b = condition_order[j]

        paired = wide[
            [condition_a, condition_b]
        ].dropna()

        t_stat, p_value = ttest_rel(
            paired[condition_a],
            paired[condition_b]
        )

        mean_a = paired[
            condition_a
        ].mean()

        mean_b = paired[
            condition_b
        ].mean()

        comparisons.append({
            "Condition_A": condition_a,
            "Condition_B": condition_b,
            "Mean_A": mean_a,
            "Mean_B": mean_b,
            "Mean_Difference": (
                mean_a - mean_b
            ),
            "t": t_stat,
            "p_uncorrected": p_value,
            "N": len(paired)
        })


posthoc_df = pd.DataFrame(
    comparisons
)


reject, corrected_p, _, _ = (
    multipletests(
        posthoc_df["p_uncorrected"],
        method="holm"
    )
)


posthoc_df[
    "p_holm"
] = corrected_p

posthoc_df[
    "Significant_Holm"
] = reject


posthoc_df.to_csv(
    output_folder
    / "alpha_posthoc_pairwise.csv",
    index=False
)


print("\n" + "=" * 60)
print("PAIRWISE FOLLOW-UP TESTS")
print("=" * 60)

significant_uncorrected = (
    posthoc_df[
        posthoc_df[
            "p_uncorrected"
        ] < 0.05
    ]
)

print(
    "\nUncorrected p < .05:"
)

if len(
    significant_uncorrected
) == 0:

    print("None")

else:

    print(
        significant_uncorrected[
            [
                "Condition_A",
                "Condition_B",
                "Mean_Difference",
                "t",
                "p_uncorrected",
                "p_holm"
            ]
        ]
    )


significant_corrected = (
    posthoc_df[
        posthoc_df[
            "Significant_Holm"
        ]
    ]
)

print(
    "\nSignificant after Holm correction:"
)

if len(
    significant_corrected
) == 0:

    print("None")

else:

    print(
        significant_corrected[
            [
                "Condition_A",
                "Condition_B",
                "Mean_Difference",
                "t",
                "p_uncorrected",
                "p_holm"
            ]
        ]
    )


means = []

sems = []


for condition in condition_order:

    values = alpha_df[
        alpha_df[
            "Condition"
        ] == condition
    ]["Difference_dB"]

    means.append(
        values.mean()
    )

    sems.append(
        values.sem()
    )


fig, ax = plt.subplots(
    figsize=(12, 7)
)


x_positions = np.arange(
    len(condition_order)
)


ax.bar(
    x_positions,
    means,
    yerr=sems,
    capsize=4,
    alpha=0.55
)


for participant in sorted(
    alpha_df[
        "Participant"
    ].unique()
):

    participant_data = (
        alpha_df[
            alpha_df[
                "Participant"
            ] == participant
        ]
        .set_index(
            "Condition"
        )
    )

    participant_values = []

    for condition in condition_order:

        if condition in participant_data.index:

            participant_values.append(
                participant_data.loc[
                    condition,
                    "Difference_dB"
                ]
            )

        else:

            participant_values.append(
                np.nan
            )

    ax.scatter(
        x_positions,
        participant_values,
        s=18,
        alpha=0.45
    )


ax.axhline(
    0,
    linewidth=1
)


ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [
        condition_labels[
            condition
        ]
        for condition
        in condition_order
    ]
)


ax.set_ylabel(
    "Alpha Power Change (Post − Pre, dB)"
)

ax.set_xlabel(
    "Condition"
)

ax.set_title(
    "Change in Alpha Power (8–12 Hz) Across Conditions"
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
    / "alpha_change_by_condition.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nFinished.")

print(
    "Results saved in:",
    output_folder.resolve()
)
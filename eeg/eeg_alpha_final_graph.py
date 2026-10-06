from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_file = Path(
    "eeg_outputs/all_participants_alpha_power.csv"
)

output_folder = Path(
    "eeg_outputs/final_figures"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(data_file)


condition_order = [
    6,
    1,
    9,
    2,
    4,
    3,
    8,
    5,
    7
]


condition_labels = {
    6: "No\nStimulation",
    1: "10 Hz\nSound",
    9: "40 Hz\nSound",
    2: "10 Hz\nVibration",
    4: "40 Hz\nVibration",
    3: "10 Hz\nLight",
    8: "40 Hz\nLight",
    5: "10 Hz\nSound + Light",
    7: "40 Hz\nSound + Light"
}


means = []
sems = []


for condition in condition_order:

    values = df[
        df["Condition"] == condition
    ]["Difference_dB"]

    means.append(
        values.mean()
    )

    sems.append(
        values.sem()
    )


x = np.arange(
    len(condition_order)
)


fig, ax = plt.subplots(
    figsize=(13, 7)
)


bars = ax.bar(
    x,
    means,
    yerr=sems,
    capsize=4,
    alpha=0.65,
    edgecolor="black",
    linewidth=0.8
)


# Make the No Stimulation bar look visually distinct
bars[0].set_alpha(0.35)
bars[0].set_hatch("//")


# Add all participant points
rng = np.random.default_rng(42)


for i, condition in enumerate(
    condition_order
):

    values = df[
        df["Condition"] == condition
    ]["Difference_dB"].values

    jitter = rng.normal(
        0,
        0.06,
        size=len(values)
    )

    ax.scatter(
        np.full(
            len(values),
            x[i]
        ) + jitter,
        values,
        s=28,
        alpha=0.55,
        zorder=3
    )


ax.axhline(
    0,
    linewidth=1
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        condition_labels[c]
        for c in condition_order
    ]
)


ax.set_ylabel(
    "Alpha Power Change (Post − Pre, dB)"
)

ax.set_xlabel(
    "Condition"
)

ax.set_title(
    "Change in Alpha Power (8–12 Hz) Across Stimulation Conditions"
)


ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


# --------------------------------------------------
# Significance bracket function
# --------------------------------------------------

def add_sig_bracket(
    ax,
    x1,
    x2,
    y,
    h,
    text
):

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
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


# --------------------------------------------------
# Significant corrected comparisons only
# --------------------------------------------------

all_values = df[
    "Difference_dB"
].dropna()

y_top = all_values.max()


# No Stim vs 10 Hz Sound + Light
x_no_stim = condition_order.index(6)
x_10_sl = condition_order.index(5)

add_sig_bracket(
    ax,
    x_no_stim,
    x_10_sl,
    y_top + 0.35,
    0.12,
    "*"
)


# 10 Hz Sound + Light vs 40 Hz Sound + Light
x_40_sl = condition_order.index(7)

add_sig_bracket(
    ax,
    x_10_sl,
    x_40_sl,
    y_top + 0.85,
    0.12,
    "**"
)


ax.text(
    0.01,
    0.98,
    "* p < .05, ** p < .01 after Holm correction",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10
)


plt.tight_layout()


plt.savefig(
    output_folder
    / "alpha_change_all_conditions.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved:",
    output_folder
    / "alpha_change_all_conditions.png"
)
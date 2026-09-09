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
}

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


matrices = {
    name: parse_matrix(filename)
    for name, filename in FILES.items()
}

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

for measure in ["BPM", "RMSSD", "SDNN"]:
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

for measure in ["BPM", "RMSSD", "SDNN"]:

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

print("\nDone.")
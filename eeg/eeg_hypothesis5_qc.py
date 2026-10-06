from pathlib import Path
import pandas as pd

agr = pd.read_csv(
    "eeg_outputs/hypothesis5/agr_baseline_corrected.csv"
)

ersp = pd.read_csv(
    "eeg_outputs/hypothesis5/occipital_alpha_ersp.csv"
)

expected_channels = {
    "Sound": {"Cz", "Fz"},
    "Light": {"O1", "O2", "P3", "P4"},
    "Sound + Light": {"O1", "O2", "P3", "P4", "Cz", "Fz"},
    "Vibration": {"Fz", "C3", "Cz", "C4", "P3", "Pz", "P4"},
}

print("\nMISSING EXPECTED CHANNELS")
print("=" * 60)

for _, row in agr.iterrows():
    used = set(str(row["Channels"]).split(","))
    expected = expected_channels[row["Modality"]]
    missing = expected - used

    if missing:
        print(
            f'Participant {row["Participant"]}, '
            f'Condition {row["Condition"]}, '
            f'{row["Modality"]}: missing {sorted(missing)}'
        )


print("\nLARGEST ABSOLUTE AGR CHANGES")
print("=" * 60)

agr["Abs_AGR_Change"] = agr[
    "Baseline_Corrected_AGR"
].abs()

print(
    agr.sort_values(
        "Abs_AGR_Change",
        ascending=False
    )[
        [
            "Participant",
            "Condition",
            "Modality",
            "Frequency",
            "Baseline_AGR",
            "Stim_AGR",
            "Baseline_Corrected_AGR"
        ]
    ].head(15)
)


print("\nLARGEST ERSP VALUES")
print("=" * 60)

print(
    ersp.sort_values(
        "Alpha_ERSP",
        ascending=False
    )[
        [
            "Participant",
            "Condition",
            "Modality",
            "Frequency",
            "Baseline_Alpha_Power",
            "Stim_Alpha_Power",
            "Alpha_ERSP"
        ]
    ].head(15)
)


print("\nSMALLEST ERSP VALUES")
print("=" * 60)

print(
    ersp.sort_values(
        "Alpha_ERSP",
        ascending=True
    )[
        [
            "Participant",
            "Condition",
            "Modality",
            "Frequency",
            "Baseline_Alpha_Power",
            "Stim_Alpha_Power",
            "Alpha_ERSP"
        ]
    ].head(15)
)
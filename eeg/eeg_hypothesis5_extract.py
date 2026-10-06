from pathlib import Path

import mne
import numpy as np
import pandas as pd


participants = range(10, 34)
conditions = range(1, 10)

condition_info = {
    1: ("Sound", 10),
    2: ("Vibration", 10),
    3: ("Light", 10),
    4: ("Vibration", 40),
    5: ("Sound + Light", 10),
    6: ("No Stimulation", 0),
    7: ("Sound + Light", 40),
    8: ("Light", 40),
    9: ("Sound", 40)
}

electrodes_by_modality = {
    "Light": ["O1", "O2", "P3", "P4"],
    "Sound": ["Cz", "Fz"],
    "Sound + Light": ["O1", "O2", "P3", "P4", "Cz", "Fz"],
    "Vibration": ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4"]
}

occipital_alpha_electrodes = [
    "O1",
    "O2",
    "Oz",
    "Pz"
]


output_folder = Path(
    "eeg_outputs/hypothesis5"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


agr_results = []
ersp_results = []


def get_available_channels(raw, requested_channels):

    available = []

    for channel in requested_channels:

        if channel in raw.ch_names:
            available.append(channel)

    return available


def hilbert_band_power(
    raw,
    channels,
    fmin,
    fmax
):

    available_channels = get_available_channels(
        raw,
        channels
    )

    if len(available_channels) == 0:

        raise ValueError(
            f"No requested channels found: {channels}"
        )

    data = raw.copy().pick(
        available_channels
    )

    data.filter(
        l_freq=fmin,
        h_freq=fmax,
        verbose=False
    )

    data.apply_hilbert(
        envelope=True,
        verbose=False
    )

    amplitude = data.get_data()

    power = amplitude ** 2

    mean_power = np.mean(power)

    return mean_power, available_channels


def analyze_condition(
    participant,
    condition
):

    modality, frequency = (
        condition_info[condition]
    )

    folder = Path(
        f"en{participant}"
    )

    pre_file = (
        folder
        / f"{participant}-{condition}-a.set"
    )

    stim_file = (
        folder
        / f"{participant}-{condition}-b.set"
    )


    if not pre_file.exists():

        print(
            "Missing:",
            pre_file
        )

        return


    if not stim_file.exists():

        print(
            "Missing:",
            stim_file
        )

        return


    print(
        f"Participant {participant}, "
        f"Condition {condition}, "
        f"{modality}, {frequency} Hz"
    )


    baseline = mne.io.read_raw_eeglab(
        pre_file,
        preload=True,
        verbose=False
    )

    stimulation = mne.io.read_raw_eeglab(
        stim_file,
        preload=True,
        verbose=False
    )


    # --------------------------------------------------
    # AGR
    # --------------------------------------------------

    if condition != 6:

        agr_channels = (
            electrodes_by_modality[
                modality
            ]
        )


        baseline_alpha, used_channels = (
            hilbert_band_power(
                baseline,
                agr_channels,
                9.5,
                10.5
            )
        )

        baseline_gamma, _ = (
            hilbert_band_power(
                baseline,
                agr_channels,
                39.5,
                40.5
            )
        )


        stimulation_alpha, _ = (
            hilbert_band_power(
                stimulation,
                agr_channels,
                9.5,
                10.5
            )
        )

        stimulation_gamma, _ = (
            hilbert_band_power(
                stimulation,
                agr_channels,
                39.5,
                40.5
            )
        )


        baseline_agr = (
            baseline_alpha
            / baseline_gamma
        )

        stimulation_agr = (
            stimulation_alpha
            / stimulation_gamma
        )

        baseline_corrected_agr = (
            stimulation_agr
            - baseline_agr
        )


        agr_results.append({
            "Participant": participant,
            "Condition": condition,
            "Modality": modality,
            "Frequency": frequency,
            "Channels":
                ",".join(used_channels),
            "Baseline_Alpha_Power":
                baseline_alpha,
            "Baseline_Gamma_Power":
                baseline_gamma,
            "Stim_Alpha_Power":
                stimulation_alpha,
            "Stim_Gamma_Power":
                stimulation_gamma,
            "Baseline_AGR":
                baseline_agr,
            "Stim_AGR":
                stimulation_agr,
            "Baseline_Corrected_AGR":
                baseline_corrected_agr
        })


    # --------------------------------------------------
    # OCCIPITAL ALPHA ERSP
    # 9–12 Hz, O1 O2 Oz Pz
    # --------------------------------------------------

    baseline_occipital_alpha, used_occipital = (
        hilbert_band_power(
            baseline,
            occipital_alpha_electrodes,
            9,
            12
        )
    )

    stimulation_occipital_alpha, _ = (
        hilbert_band_power(
            stimulation,
            occipital_alpha_electrodes,
            9,
            12
        )
    )


    ersp = (
        stimulation_occipital_alpha
        / baseline_occipital_alpha
    )


    ersp_results.append({
        "Participant": participant,
        "Condition": condition,
        "Modality": modality,
        "Frequency": frequency,
        "Channels":
            ",".join(used_occipital),
        "Baseline_Alpha_Power":
            baseline_occipital_alpha,
        "Stim_Alpha_Power":
            stimulation_occipital_alpha,
        "Alpha_ERSP":
            ersp
    })


for participant in participants:

    for condition in conditions:

        try:

            analyze_condition(
                participant,
                condition
            )

        except Exception as e:

            print(
                f"ERROR Participant {participant}, "
                f"Condition {condition}: {e}"
            )


agr_df = pd.DataFrame(
    agr_results
)

ersp_df = pd.DataFrame(
    ersp_results
)


agr_df.to_csv(
    output_folder
    / "agr_baseline_corrected.csv",
    index=False
)


ersp_df.to_csv(
    output_folder
    / "occipital_alpha_ersp.csv",
    index=False
)


print("\nFinished.")

print(
    "AGR rows:",
    len(agr_df)
)

print(
    "ERSP rows:",
    len(ersp_df)
)

print(
    "Saved to:",
    output_folder.resolve()
)
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


# ============================================================
# SETTINGS
# ============================================================

PARTICIPANTS = range(10, 34)

BASELINE_SECONDS = 30
STIM_SECONDS = 120

OUTPUT = Path("eeg_outputs/hypothesis5a_FINAL")
OUTPUT.mkdir(parents=True, exist_ok=True)

SUMMARY_DIR = OUTPUT / "participant_summary_figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIRMED EXCLUSIONS
# ============================================================

# Only exclude blocks for a recording/QC reason.
# Do NOT exclude simply because the result is extreme.

EXCLUDE_BLOCKS = {
    (10, 5): "Severely corrupted/disconnected recording"
}


# ============================================================
# CONDITION INFORMATION
# ============================================================

condition_info = {
    1: ("Sound", 10),
    2: ("Vibration", 10),
    3: ("Light", 10),
    4: ("Vibration", 40),
    5: ("Sound + Light", 10),
    6: ("No Stimulation", 0),
    7: ("Sound + Light", 40),
    8: ("Light", 40),
    9: ("Sound", 40),
}


condition_names = {
    1: "10 Hz Sound",
    2: "10 Hz Vibration",
    3: "10 Hz Light",
    4: "40 Hz Vibration",
    5: "10 Hz Sound + Light",
    6: "No Stimulation",
    7: "40 Hz Sound + Light",
    8: "40 Hz Light",
    9: "40 Hz Sound",
}


condition_pairs = {

    "Sound": {
        10: 1,
        40: 9,
    },

    "Vibration": {
        10: 2,
        40: 4,
    },

    "Light": {
        10: 3,
        40: 8,
    },

    "Sound + Light": {
        10: 5,
        40: 7,
    },
}


modalities = [
    "Sound",
    "Vibration",
    "Light",
    "Sound + Light",
]


# ============================================================
# PREREGISTERED H5A ELECTRODES
# ============================================================

electrodes_by_modality = {

    "Sound": [
        "Cz",
        "Fz",
    ],

    "Vibration": [
        "Fz",
        "C3",
        "Cz",
        "C4",
        "P3",
        "Pz",
        "P4",
    ],

    "Light": [
        "O1",
        "O2",
        "P3",
        "P4",
    ],

    "Sound + Light": [
        "O1",
        "O2",
        "P3",
        "P4",
        "Cz",
        "Fz",
    ],
}


agr_rows = []
nostim_rows = []
topomap_rows = []
qc_rows = []


# ============================================================
# HELPERS
# ============================================================

def crop_first_seconds(raw, seconds):

    sfreq = raw.info["sfreq"]

    n_samples = int(
        round(seconds * sfreq)
    )

    if raw.n_times < n_samples:

        raise ValueError(
            f"Recording contains only "
            f"{raw.n_times / sfreq:.2f} s; "
            f"{seconds} s required."
        )

    tmax = (
        n_samples - 1
    ) / sfreq

    return raw.copy().crop(
        tmin=0,
        tmax=tmax,
        include_tmax=True
    )


def load_pair(participant, condition):

    folder = Path(
        f"en{participant}"
    )

    baseline_file = (
        folder
        / f"{participant}-{condition}-a.set"
    )

    stim_file = (
        folder
        / f"{participant}-{condition}-b.set"
    )


    if not baseline_file.exists():

        raise FileNotFoundError(
            f"Missing {baseline_file}"
        )


    if not stim_file.exists():

        raise FileNotFoundError(
            f"Missing {stim_file}"
        )


    baseline_raw = (
        mne.io.read_raw_eeglab(
            baseline_file,
            preload=True,
            verbose=False
        )
    )


    stim_raw = (
        mne.io.read_raw_eeglab(
            stim_file,
            preload=True,
            verbose=False
        )
    )


    baseline = crop_first_seconds(
        baseline_raw,
        BASELINE_SECONDS
    )


    stimulation = crop_first_seconds(
        stim_raw,
        STIM_SECONDS
    )


    return baseline, stimulation


def missing_channels(raw, required):

    return [
        ch
        for ch in required
        if ch not in raw.ch_names
    ]


def common_eeg_channels(raws):

    channel_lists = []

    for raw in raws:

        eeg_raw = (
            raw.copy()
            .pick(
                picks="eeg"
            )
        )

        channel_lists.append(
            eeg_raw.ch_names
        )


    common = (
        channel_lists[0].copy()
    )


    for channels in channel_lists[1:]:

        common = [
            ch
            for ch in common
            if ch in channels
        ]


    return common


def hilbert_channel_power(
    raw,
    channels,
    fmin,
    fmax
):

    temp = (
        raw.copy()
        .pick(channels)
    )


    temp.filter(
        l_freq=fmin,
        h_freq=fmax,
        verbose=False
    )


    temp.apply_hilbert(
        envelope=False,
        verbose=False
    )


    analytic = temp.get_data()

    power = (
        np.abs(analytic) ** 2
    )


    # Mean across time.
    # Leaves one value per channel.

    return power.mean(
        axis=1
    )


def hilbert_mean_power(
    raw,
    channels,
    fmin,
    fmax
):

    return (
        hilbert_channel_power(
            raw,
            channels,
            fmin,
            fmax
        )
        .mean()
    )


def psd_values(raw):

    spectrum = (
        raw.compute_psd(
            fmin=1,
            fmax=50,
            verbose=False
        )
    )


    data, frequencies = (
        spectrum.get_data(
            return_freqs=True
        )
    )


    mean_db = (
        10
        * np.log10(
            data.mean(
                axis=0
            )
        )
    )


    return frequencies, mean_db


def make_mask(
    info,
    highlighted_channels
):

    mask = np.array([
        channel in highlighted_channels
        for channel in info["ch_names"]
    ])


    return mask


def plot_topomap(
    ax,
    values,
    info,
    title,
    vmin,
    vmax,
    required,
    cmap
):

    mask = make_mask(
        info,
        required
    )


    image, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        sensors=True,
        contours=6,
        cmap=cmap,
        vlim=(
            vmin,
            vmax
        ),
        mask=mask,
        mask_params={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "linewidth": 0,
            "markersize": 10,
            "markeredgewidth": 2,
        },
    )


    ax.set_title(
        title,
        fontsize=11
    )


    return image


# ============================================================
# MAIN STIMULATION ANALYSIS
# ============================================================

for participant in PARTICIPANTS:

    participant_dir = (
        SUMMARY_DIR
        / f"participant_{participant}"
    )

    participant_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for modality in modalities:

        condition10 = (
            condition_pairs[
                modality
            ][10]
        )

        condition40 = (
            condition_pairs[
                modality
            ][40]
        )


        required = (
            electrodes_by_modality[
                modality
            ]
        )


        # ====================================================
        # CHECK EXCLUSIONS
        # ====================================================

        pair_excluded = False

        for condition in [
            condition10,
            condition40
        ]:

            if (
                participant,
                condition
            ) in EXCLUDE_BLOCKS:

                qc_rows.append({

                    "Participant":
                        participant,

                    "Condition":
                        condition,

                    "Modality":
                        modality,

                    "Issue":
                        EXCLUDE_BLOCKS[
                            (
                                participant,
                                condition
                            )
                        ]
                })

                pair_excluded = True


        # We still allow the non-excluded condition to enter
        # the AGR dataset later if it is valid, but we cannot
        # make a misleading paired 10-vs-40 summary PNG.

        try:

            if not pair_excluded:

                base10, stim10 = (
                    load_pair(
                        participant,
                        condition10
                    )
                )

                base40, stim40 = (
                    load_pair(
                        participant,
                        condition40
                    )
                )


                # ============================================
                # COMMON CHANNELS FOR SCALP MAPS
                # ============================================

                common_channels = (
                    common_eeg_channels(
                        [
                            base10,
                            stim10,
                            base40,
                            stim40,
                        ]
                    )
                )


                b10_topo = (
                    base10.copy()
                    .pick(common_channels)
                )

                s10_topo = (
                    stim10.copy()
                    .pick(common_channels)
                )

                b40_topo = (
                    base40.copy()
                    .pick(common_channels)
                )

                s40_topo = (
                    stim40.copy()
                    .pick(common_channels)
                )


                topo_info = (
                    b10_topo.info
                )


                # ============================================
                # 9.5–10.5 HZ POWER
                # ============================================

                b10_alpha = (
                    hilbert_channel_power(
                        b10_topo,
                        common_channels,
                        9.5,
                        10.5
                    )
                )

                s10_alpha = (
                    hilbert_channel_power(
                        s10_topo,
                        common_channels,
                        9.5,
                        10.5
                    )
                )

                b40_alpha = (
                    hilbert_channel_power(
                        b40_topo,
                        common_channels,
                        9.5,
                        10.5
                    )
                )

                s40_alpha = (
                    hilbert_channel_power(
                        s40_topo,
                        common_channels,
                        9.5,
                        10.5
                    )
                )


                # ============================================
                # 39.5–40.5 HZ POWER
                # ============================================

                b10_gamma = (
                    hilbert_channel_power(
                        b10_topo,
                        common_channels,
                        39.5,
                        40.5
                    )
                )

                s10_gamma = (
                    hilbert_channel_power(
                        s10_topo,
                        common_channels,
                        39.5,
                        40.5
                    )
                )

                b40_gamma = (
                    hilbert_channel_power(
                        b40_topo,
                        common_channels,
                        39.5,
                        40.5
                    )
                )

                s40_gamma = (
                    hilbert_channel_power(
                        s40_topo,
                        common_channels,
                        39.5,
                        40.5
                    )
                )


                # ============================================
                # AGR AT EVERY ELECTRODE
                # ============================================

                b10_agr = (
                    b10_alpha
                    / b10_gamma
                )

                s10_agr = (
                    s10_alpha
                    / s10_gamma
                )

                b40_agr = (
                    b40_alpha
                    / b40_gamma
                )

                s40_agr = (
                    s40_alpha
                    / s40_gamma
                )


                # ============================================
                # PREREGISTERED AGR DIFFERENCE MAP
                #
                # Stim AGR - Baseline AGR
                # ============================================

                agr_difference10 = (
                    s10_agr
                    - b10_agr
                )

                agr_difference40 = (
                    s40_agr
                    - b40_agr
                )


                # ============================================
                # ADDITIONAL DESCRIPTIVE AGR RATIO MAP
                #
                # Stim AGR / Baseline AGR
                # ============================================

                agr_ratio10 = (
                    s10_agr
                    / b10_agr
                )

                agr_ratio40 = (
                    s40_agr
                    / b40_agr
                )


                # ============================================
                # SAVE ELECTRODE-LEVEL MAP VALUES
                # ============================================

                for i, channel in enumerate(
                    common_channels
                ):

                    topomap_rows.append({

                        "Participant":
                            participant,

                        "Modality":
                            modality,

                        "Channel":
                            channel,

                        "Alpha_10HzStim":
                            s10_alpha[i],

                        "Alpha_40HzStim":
                            s40_alpha[i],

                        "Gamma_10HzStim":
                            s10_gamma[i],

                        "Gamma_40HzStim":
                            s40_gamma[i],

                        "AGR_Difference_10Hz":
                            agr_difference10[i],

                        "AGR_Difference_40Hz":
                            agr_difference40[i],

                        "AGR_Ratio_10Hz":
                            agr_ratio10[i],

                        "AGR_Ratio_40Hz":
                            agr_ratio40[i],
                    })


                # ============================================
                # SHARED COLOR SCALES
                # ============================================

                alpha_vmin = min(
                    s10_alpha.min(),
                    s40_alpha.min()
                )

                alpha_vmax = max(
                    s10_alpha.max(),
                    s40_alpha.max()
                )


                gamma_vmin = min(
                    s10_gamma.min(),
                    s40_gamma.min()
                )

                gamma_vmax = max(
                    s10_gamma.max(),
                    s40_gamma.max()
                )


                # AGR difference is centered at 0

                difference_limit = max(
                    np.nanmax(
                        np.abs(
                            agr_difference10
                        )
                    ),
                    np.nanmax(
                        np.abs(
                            agr_difference40
                        )
                    )
                )


                # AGR ratio is centered at 1

                ratio_distance = max(
                    np.nanmax(
                        np.abs(
                            agr_ratio10 - 1
                        )
                    ),
                    np.nanmax(
                        np.abs(
                            agr_ratio40 - 1
                        )
                    )
                )


                ratio_vmin = max(
                    0,
                    1 - ratio_distance
                )

                ratio_vmax = (
                    1 + ratio_distance
                )


                # ============================================
                # ONE SUMMARY PNG
                # ============================================

                fig, axes = plt.subplots(
                    5,
                    2,
                    figsize=(12, 20)
                )


                # --------------------------------------------
                # ROW 1 — PSD
                # --------------------------------------------

                freq10, base10_psd = (
                    psd_values(
                        base10
                    )
                )

                _, stim10_psd = (
                    psd_values(
                        stim10
                    )
                )


                freq40, base40_psd = (
                    psd_values(
                        base40
                    )
                )

                _, stim40_psd = (
                    psd_values(
                        stim40
                    )
                )


                axes[0, 0].plot(
                    freq10,
                    base10_psd,
                    linewidth=1.8,
                    label="Baseline"
                )

                axes[0, 0].plot(
                    freq10,
                    stim10_psd,
                    linewidth=1.8,
                    label="Stimulation"
                )


                axes[0, 1].plot(
                    freq40,
                    base40_psd,
                    linewidth=1.8,
                    label="Baseline"
                )

                axes[0, 1].plot(
                    freq40,
                    stim40_psd,
                    linewidth=1.8,
                    label="Stimulation"
                )


                for ax in axes[0, :]:

                    ax.axvspan(
                        9.5,
                        10.5,
                        alpha=0.1
                    )

                    ax.axvspan(
                        39.5,
                        40.5,
                        alpha=0.1
                    )

                    ax.set_xlim(
                        1,
                        50
                    )

                    ax.set_xlabel(
                        "Frequency (Hz)"
                    )

                    ax.set_ylabel(
                        "PSD (dB)"
                    )

                    ax.legend(
                        frameon=False
                    )

                    ax.spines[
                        "top"
                    ].set_visible(False)

                    ax.spines[
                        "right"
                    ].set_visible(False)


                axes[0, 0].set_title(
                    "10 Hz stimulation — PSD"
                )

                axes[0, 1].set_title(
                    "40 Hz stimulation — PSD"
                )


                # --------------------------------------------
                # ROW 2 — 10 HZ POWER
                # --------------------------------------------

                alpha10_img = plot_topomap(
                    axes[1, 0],
                    s10_alpha,
                    topo_info,
                    "10 Hz stimulation\n9.5–10.5 Hz power",
                    alpha_vmin,
                    alpha_vmax,
                    required,
                    "Reds"
                )


                alpha40_img = plot_topomap(
                    axes[1, 1],
                    s40_alpha,
                    topo_info,
                    "40 Hz stimulation\n9.5–10.5 Hz power",
                    alpha_vmin,
                    alpha_vmax,
                    required,
                    "Reds"
                )


                alpha_bar = fig.colorbar(
                    alpha40_img,
                    ax=axes[1, :],
                    shrink=0.72,
                    pad=0.03
                )

                alpha_bar.set_label(
                    "9.5–10.5 Hz power"
                )


                # --------------------------------------------
                # ROW 3 — 40 HZ POWER
                # --------------------------------------------

                gamma10_img = plot_topomap(
                    axes[2, 0],
                    s10_gamma,
                    topo_info,
                    "10 Hz stimulation\n39.5–40.5 Hz power",
                    gamma_vmin,
                    gamma_vmax,
                    required,
                    "Reds"
                )


                gamma40_img = plot_topomap(
                    axes[2, 1],
                    s40_gamma,
                    topo_info,
                    "40 Hz stimulation\n39.5–40.5 Hz power",
                    gamma_vmin,
                    gamma_vmax,
                    required,
                    "Reds"
                )


                gamma_bar = fig.colorbar(
                    gamma40_img,
                    ax=axes[2, :],
                    shrink=0.72,
                    pad=0.03
                )

                gamma_bar.set_label(
                    "39.5–40.5 Hz power"
                )


                # --------------------------------------------
                # ROW 4 — PREREGISTERED AGR DIFFERENCE
                # --------------------------------------------

                diff10_img = plot_topomap(
                    axes[3, 0],
                    agr_difference10,
                    topo_info,
                    "10 Hz stimulation\nAGR difference",
                    -difference_limit,
                    difference_limit,
                    required,
                    "RdBu_r"
                )


                diff40_img = plot_topomap(
                    axes[3, 1],
                    agr_difference40,
                    topo_info,
                    "40 Hz stimulation\nAGR difference",
                    -difference_limit,
                    difference_limit,
                    required,
                    "RdBu_r"
                )


                difference_bar = fig.colorbar(
                    diff40_img,
                    ax=axes[3, :],
                    shrink=0.72,
                    pad=0.03
                )

                difference_bar.set_label(
                    "Stim AGR − Baseline AGR\n"
                    "0 = no change"
                )


                # --------------------------------------------
                # ROW 5 — DESCRIPTIVE AGR RATIO
                # --------------------------------------------

                ratio10_img = plot_topomap(
                    axes[4, 0],
                    agr_ratio10,
                    topo_info,
                    "10 Hz stimulation\nAGR ratio",
                    ratio_vmin,
                    ratio_vmax,
                    required,
                    "RdBu_r"
                )


                ratio40_img = plot_topomap(
                    axes[4, 1],
                    agr_ratio40,
                    topo_info,
                    "40 Hz stimulation\nAGR ratio",
                    ratio_vmin,
                    ratio_vmax,
                    required,
                    "RdBu_r"
                )


                ratio_bar = fig.colorbar(
                    ratio40_img,
                    ax=axes[4, :],
                    shrink=0.72,
                    pad=0.03
                )

                ratio_bar.set_label(
                    "Stim AGR / Baseline AGR\n"
                    "1 = no change"
                )


                fig.suptitle(
                    f"Participant {participant} — {modality}\n"
                    f"Preregistered AGR electrodes: "
                    + ", ".join(
                        required
                    ),
                    fontsize=15
                )


                fig.text(
                    0.5,
                    0.012,
                    "Black-circled sensors are the preregistered electrodes used "
                    "for the formal H5A AGR analysis. "
                    "AGR difference is preregistered; AGR ratio is descriptive.",
                    ha="center",
                    fontsize=9
                )


                plt.subplots_adjust(
                    top=0.95,
                    bottom=0.05,
                    hspace=0.48,
                    wspace=0.25
                )


                safe_modality = (
                    modality
                    .lower()
                    .replace(
                        " + ",
                        "_plus_"
                    )
                    .replace(
                        " ",
                        "_"
                    )
                )


                plt.savefig(
                    participant_dir
                    / f"{safe_modality}_summary.png",
                    dpi=300,
                    bbox_inches="tight"
                )


                plt.close()


        except Exception as e:

            qc_rows.append({

                "Participant":
                    participant,

                "Condition":
                    f"{condition10},{condition40}",

                "Modality":
                    modality,

                "Issue":
                    f"Summary figure error: {e}"
            })


        # ====================================================
        # FORMAL AGR EXTRACTION
        #
        # PROCESS EACH CONDITION INDIVIDUALLY SO ONE BAD
        # CONDITION DOES NOT AUTOMATICALLY DELETE ITS PAIR.
        # ====================================================

        for frequency, condition in [

            (10, condition10),
            (40, condition40),

        ]:

            if (
                participant,
                condition
            ) in EXCLUDE_BLOCKS:

                continue


            try:

                baseline, stimulation = (
                    load_pair(
                        participant,
                        condition
                    )
                )


                missing = sorted(
                    set(
                        missing_channels(
                            baseline,
                            required
                        )
                        +
                        missing_channels(
                            stimulation,
                            required
                        )
                    )
                )


                if missing:

                    qc_rows.append({

                        "Participant":
                            participant,

                        "Condition":
                            condition,

                        "Modality":
                            modality,

                        "Issue":
                            "AGR skipped: missing "
                            + ", ".join(
                                missing
                            )
                    })

                    continue


                baseline_alpha = (
                    hilbert_mean_power(
                        baseline,
                        required,
                        9.5,
                        10.5
                    )
                )


                baseline_gamma = (
                    hilbert_mean_power(
                        baseline,
                        required,
                        39.5,
                        40.5
                    )
                )


                stim_alpha = (
                    hilbert_mean_power(
                        stimulation,
                        required,
                        9.5,
                        10.5
                    )
                )


                stim_gamma = (
                    hilbert_mean_power(
                        stimulation,
                        required,
                        39.5,
                        40.5
                    )
                )


                baseline_agr = (
                    baseline_alpha
                    / baseline_gamma
                )


                stim_agr = (
                    stim_alpha
                    / stim_gamma
                )


                baseline_corrected_agr = (
                    stim_agr
                    - baseline_agr
                )


                agr_rows.append({

                    "Participant":
                        participant,

                    "Condition":
                        condition,

                    "Condition_Name":
                        condition_names[
                            condition
                        ],

                    "Modality":
                        modality,

                    "Frequency":
                        frequency,

                    "Electrodes":
                        ", ".join(
                            required
                        ),

                    "Baseline_Alpha":
                        baseline_alpha,

                    "Stim_Alpha":
                        stim_alpha,

                    "Baseline_Gamma":
                        baseline_gamma,

                    "Stim_Gamma":
                        stim_gamma,

                    "Baseline_AGR":
                        baseline_agr,

                    "Stim_AGR":
                        stim_agr,

                    "Baseline_Corrected_AGR":
                        baseline_corrected_agr,

                    "AGR_Response_Ratio":
                        (
                            stim_agr
                            / baseline_agr
                        )
                })


            except Exception as e:

                qc_rows.append({

                    "Participant":
                        participant,

                    "Condition":
                        condition,

                    "Modality":
                        modality,

                    "Issue":
                        f"AGR extraction error: {e}"
                })


# ============================================================
# NO-STIM AGR
#
# SECONDARY ONLY.
#
# Condition 6 is calculated using each modality's electrode set
# so the no-stim value uses the same sensors as the comparison.
# ============================================================

for participant in PARTICIPANTS:

    try:

        baseline, nostim_period = (
            load_pair(
                participant,
                6
            )
        )

    except Exception as e:

        qc_rows.append({

            "Participant":
                participant,

            "Condition":
                6,

            "Modality":
                "No Stimulation",

            "Issue":
                f"No-stim load error: {e}"
        })

        continue


    for modality in modalities:

        required = (
            electrodes_by_modality[
                modality
            ]
        )


        missing = sorted(
            set(
                missing_channels(
                    baseline,
                    required
                )
                +
                missing_channels(
                    nostim_period,
                    required
                )
            )
        )


        if missing:

            qc_rows.append({

                "Participant":
                    participant,

                "Condition":
                    6,

                "Modality":
                    modality,

                "Issue":
                    "No-stim AGR skipped: missing "
                    + ", ".join(
                        missing
                    )
            })

            continue


        baseline_alpha = (
            hilbert_mean_power(
                baseline,
                required,
                9.5,
                10.5
            )
        )


        baseline_gamma = (
            hilbert_mean_power(
                baseline,
                required,
                39.5,
                40.5
            )
        )


        nostim_alpha = (
            hilbert_mean_power(
                nostim_period,
                required,
                9.5,
                10.5
            )
        )


        nostim_gamma = (
            hilbert_mean_power(
                nostim_period,
                required,
                39.5,
                40.5
            )
        )


        baseline_agr = (
            baseline_alpha
            / baseline_gamma
        )


        nostim_agr = (
            nostim_alpha
            / nostim_gamma
        )


        nostim_rows.append({

            "Participant":
                participant,

            "Modality":
                modality,

            "Electrodes":
                ", ".join(
                    required
                ),

            "Baseline_AGR":
                baseline_agr,

            "NoStim_AGR":
                nostim_agr,

            "Baseline_Corrected_AGR":
                (
                    nostim_agr
                    - baseline_agr
                )
        })


# ============================================================
# SAVE RAW OUTPUT TABLES
# ============================================================

agr = pd.DataFrame(
    agr_rows
)

nostim = pd.DataFrame(
    nostim_rows
)

topomap_data = pd.DataFrame(
    topomap_rows
)

qc = pd.DataFrame(
    qc_rows
)


agr.to_csv(
    OUTPUT
    / "h5a_agr_data.csv",
    index=False
)


nostim.to_csv(
    OUTPUT
    / "h5a_no_stim_agr_data.csv",
    index=False
)


topomap_data.to_csv(
    OUTPUT
    / "h5a_topomap_values.csv",
    index=False
)


qc.to_csv(
    OUTPUT
    / "qc_log.csv",
    index=False
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = (
    agr.groupby(
        [
            "Modality",
            "Frequency",
            "Electrodes"
        ]
    )[
        "Baseline_Corrected_AGR"
    ]
    .agg(
        Mean="mean",
        SD="std",
        N="count",
        SEM="sem"
    )
    .reset_index()
)


summary.to_csv(
    OUTPUT
    / "h5a_summary.csv",
    index=False
)


# ============================================================
# PRIMARY PREREGISTERED ANOVA
#
# 4 modality × 2 frequency repeated measures
# ============================================================

counts = (
    agr.groupby(
        "Participant"
    )
    .size()
)


complete_participants = (
    counts[
        counts == 8
    ]
    .index
)


anova_data = (
    agr[
        agr[
            "Participant"
        ].isin(
            complete_participants
        )
    ]
    .copy()
)


print(
    "\nComplete participants for 4 × 2 AGR ANOVA:",
    len(
        complete_participants
    )
)


if len(
    complete_participants
) > 1:

    anova = (
        AnovaRM(
            data=anova_data,
            depvar=
                "Baseline_Corrected_AGR",
            subject=
                "Participant",
            within=[
                "Modality",
                "Frequency"
            ]
        )
        .fit()
    )


    print(
        "\nH5A REPEATED-MEASURES ANOVA"
    )

    print(
        anova
    )


    with open(
        OUTPUT
        / "h5a_agr_anova.txt",
        "w"
    ) as f:

        f.write(
            str(anova)
        )


# ============================================================
# 10 HZ VS 40 HZ WITHIN EACH MODALITY
# ============================================================

followups = []


for modality in modalities:

    temp = (
        agr[
            agr[
                "Modality"
            ]
            == modality
        ]
    )


    wide = (
        temp.pivot(
            index="Participant",
            columns="Frequency",
            values=
                "Baseline_Corrected_AGR"
        )
        .dropna(
            subset=[
                10,
                40
            ]
        )
    )


    if len(wide) < 2:
        continue


    t_value, p_value = (
        ttest_rel(
            wide[10],
            wide[40]
        )
    )


    followups.append({

        "Modality":
            modality,

        "Electrodes":
            ", ".join(
                electrodes_by_modality[
                    modality
                ]
            ),

        "Mean_10Hz":
            wide[10].mean(),

        "Mean_40Hz":
            wide[40].mean(),

        "Difference_10_minus_40":
            (
                wide[10]
                - wide[40]
            ).mean(),

        "t":
            t_value,

        "p_uncorrected":
            p_value,

        "N":
            len(wide)
    })


followups = pd.DataFrame(
    followups
)


if len(followups) > 0:

    reject, corrected_p, _, _ = (
        multipletests(
            followups[
                "p_uncorrected"
            ],
            method="holm"
        )
    )


    followups[
        "p_holm"
    ] = corrected_p


    followups[
        "Significant_Holm"
    ] = reject


followups.to_csv(
    OUTPUT
    / "h5a_10_vs_40_followups.csv",
    index=False
)


# ============================================================
# PRIMARY GROUP FIGURE
#
# 10 HZ VS 40 HZ
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
    sharey=True
)


axes = axes.flatten()


rng = np.random.default_rng(
    42
)


for ax, modality in zip(
    axes,
    modalities
):

    temp = (
        agr[
            agr[
                "Modality"
            ]
            == modality
        ]
    )


    wide = (
        temp.pivot(
            index="Participant",
            columns="Frequency",
            values=
                "Baseline_Corrected_AGR"
        )
    )


    for _, row in wide.iterrows():

        if (
            pd.notna(
                row.get(10)
            )
            and
            pd.notna(
                row.get(40)
            )
        ):

            ax.plot(
                [0, 1],
                [
                    row[10],
                    row[40]
                ],
                alpha=0.20,
                linewidth=0.8
            )


    for x_position, frequency in enumerate(
        [10, 40]
    ):

        values = (
            temp[
                temp[
                    "Frequency"
                ]
                == frequency
            ][
                "Baseline_Corrected_AGR"
            ]
            .dropna()
        )


        if len(values) == 0:
            continue


        jitter = (
            rng.normal(
                0,
                0.035,
                len(values)
            )
        )


        ax.scatter(
            np.full(
                len(values),
                x_position
            )
            + jitter,
            values,
            s=30,
            alpha=0.55,
            zorder=3
        )


        ax.errorbar(
            x_position,
            values.mean(),
            yerr=values.sem(),
            fmt="o",
            markersize=9,
            capsize=5,
            linewidth=2,
            zorder=4
        )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )


    ax.set_xticks(
        [0, 1]
    )


    ax.set_xticklabels(
        [
            "10 Hz",
            "40 Hz"
        ]
    )


    ax.set_title(
        f"{modality}\n"
        f"Electrodes: "
        + ", ".join(
            electrodes_by_modality[
                modality
            ]
        )
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


axes[0].set_ylabel(
    "Baseline-Corrected AGR\n"
    "(Stim AGR − Baseline AGR)"
)

axes[2].set_ylabel(
    "Baseline-Corrected AGR\n"
    "(Stim AGR − Baseline AGR)"
)


fig.suptitle(
    "Hypothesis 5A: 10 Hz vs 40 Hz Alpha/Gamma Ratio",
    fontsize=15
)


fig.text(
    0.5,
    0.025,
    "Points = participants • "
    "lines connect paired observations • "
    "large point = mean ± SEM",
    ha="center",
    fontsize=9
)


plt.tight_layout(
    rect=[
        0,
        0.06,
        1,
        0.95
    ]
)


plt.savefig(
    OUTPUT
    / "FIGURE_1_H5A_10_vs_40_AGR.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SECONDARY FIGURE
#
# NO STIM VS 10 HZ VS 40 HZ
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
    sharey=True
)


axes = axes.flatten()


for ax, modality in zip(
    axes,
    modalities
):

    control = (
        nostim[
            nostim[
                "Modality"
            ]
            == modality
        ][
            [
                "Participant",
                "Baseline_Corrected_AGR"
            ]
        ]
        .rename(
            columns={
                "Baseline_Corrected_AGR":
                    "NoStim"
            }
        )
    )


    stim = (
        agr[
            agr[
                "Modality"
            ]
            == modality
        ]
    )


    ten = (
        stim[
            stim[
                "Frequency"
            ]
            == 10
        ][
            [
                "Participant",
                "Baseline_Corrected_AGR"
            ]
        ]
        .rename(
            columns={
                "Baseline_Corrected_AGR":
                    "TenHz"
            }
        )
    )


    forty = (
        stim[
            stim[
                "Frequency"
            ]
            == 40
        ][
            [
                "Participant",
                "Baseline_Corrected_AGR"
            ]
        ]
        .rename(
            columns={
                "Baseline_Corrected_AGR":
                    "FortyHz"
            }
        )
    )


    combined = (
        control
        .merge(
            ten,
            on="Participant",
            how="inner"
        )
        .merge(
            forty,
            on="Participant",
            how="inner"
        )
    )


    for _, row in combined.iterrows():

        ax.plot(
            [0, 1, 2],
            [
                row["NoStim"],
                row["TenHz"],
                row["FortyHz"]
            ],
            alpha=0.16,
            linewidth=0.8
        )


    for x_position, column in enumerate(
        [
            "NoStim",
            "TenHz",
            "FortyHz"
        ]
    ):

        values = (
            combined[
                column
            ]
            .dropna()
        )


        if len(values) == 0:
            continue


        jitter = (
            rng.normal(
                0,
                0.035,
                len(values)
            )
        )


        ax.scatter(
            np.full(
                len(values),
                x_position
            )
            + jitter,
            values,
            s=28,
            alpha=0.5,
            zorder=3
        )


        ax.errorbar(
            x_position,
            values.mean(),
            yerr=values.sem(),
            fmt="o",
            markersize=9,
            capsize=5,
            linewidth=2,
            zorder=4
        )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )


    ax.set_xticks(
        [0, 1, 2]
    )


    ax.set_xticklabels(
        [
            "No Stim",
            "10 Hz",
            "40 Hz"
        ]
    )


    ax.set_title(
        f"{modality}\n"
        f"Electrodes: "
        + ", ".join(
            electrodes_by_modality[
                modality
            ]
        )
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


axes[0].set_ylabel(
    "Baseline-Corrected AGR"
)

axes[2].set_ylabel(
    "Baseline-Corrected AGR"
)


fig.suptitle(
    "Secondary H5A Comparison: No Stimulation vs 10 Hz vs 40 Hz",
    fontsize=15
)


fig.text(
    0.5,
    0.025,
    "No-stimulation AGR is calculated using the same "
    "preregistered electrode set as each modality.",
    ha="center",
    fontsize=9
)


plt.tight_layout(
    rect=[
        0,
        0.06,
        1,
        0.95
    ]
)


plt.savefig(
    OUTPUT
    / "FIGURE_2_H5A_with_NoStim.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 72)
print("HYPOTHESIS 5A ANALYSIS FINISHED")
print("=" * 72)


print(
    "\nOPEN THESE FIRST:"
)

print(
    "FIGURE_1_H5A_10_vs_40_AGR.png"
)

print(
    "FIGURE_2_H5A_with_NoStim.png"
)

print(
    "participant_summary_figures/"
)


print(
    "\nEach participant folder contains up to:"
)

print(
    "sound_summary.png"
)

print(
    "vibration_summary.png"
)

print(
    "light_summary.png"
)

print(
    "sound_plus_light_summary.png"
)


print(
    "\nTABLES:"
)

print(
    "h5a_agr_data.csv"
)

print(
    "h5a_summary.csv"
)

print(
    "h5a_10_vs_40_followups.csv"
)

print(
    "h5a_agr_anova.txt"
)

print(
    "h5a_no_stim_agr_data.csv"
)

print(
    "h5a_topomap_values.csv"
)

print(
    "qc_log.csv"
)
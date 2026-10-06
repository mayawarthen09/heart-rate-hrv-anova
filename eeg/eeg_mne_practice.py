from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


print("MNE version:", mne.__version__)


participants = range(10, 34)
conditions = range(1, 10)

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

bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 45)
}


output_root = Path("eeg_outputs")
output_root.mkdir(exist_ok=True)

band_results = []


def analyze_participant(participant, condition):

    participant_folder = Path(f"en{participant}")

    pre_file = (
        participant_folder
        / f"{participant}-{condition}-a.set"
    )

    post_file = (
        participant_folder
        / f"{participant}-{condition}-b.set"
    )

    condition_name = condition_labels[condition]

    print("\n" + "=" * 70)

    print(
        f"Participant {participant}, "
        f"Condition {condition}: "
        f"{condition_name}"
    )

    print("=" * 70)


    if not pre_file.exists():
        print("Missing:", pre_file)
        return

    if not post_file.exists():
        print("Missing:", post_file)
        return


    condition_output = (
        output_root
        / f"participant_{participant}"
        / f"condition_{condition}"
    )

    condition_output.mkdir(
        parents=True,
        exist_ok=True
    )


    pre = mne.io.read_raw_eeglab(
        pre_file,
        preload=True
    )

    post = mne.io.read_raw_eeglab(
        post_file,
        preload=True
    )


    print("Pre duration:", pre.times[-1], "seconds")
    print("Post duration:", post.times[-1], "seconds")
    print("Sampling frequency:", pre.info["sfreq"])
    print("Channels:", pre.ch_names)
    print("Bad channels:", pre.info["bads"])


    pre_psd = pre.compute_psd(
        fmin=1,
        fmax=50
    )

    post_psd = post.compute_psd(
        fmin=1,
        fmax=50
    )


    pre_data, pre_freqs = pre_psd.get_data(
        return_freqs=True
    )

    post_data, post_freqs = post_psd.get_data(
        return_freqs=True
    )


    pre_db = 10 * np.log10(pre_data)
    post_db = 10 * np.log10(post_data)

    pre_mean = pre_db.mean(axis=0)
    post_mean = post_db.mean(axis=0)


    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        pre_freqs,
        pre_mean,
        label="Pre-stimulation"
    )

    ax.plot(
        post_freqs,
        post_mean,
        label="Post-stimulation"
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB)")

    ax.set_title(
        f"Participant {participant}: "
        f"{condition_name}\n"
        f"Pre vs Post PSD"
    )

    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        condition_output / "pre_post_psd.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    for band_name, (fmin, fmax) in bands.items():

        pre_mask = (
            (pre_freqs >= fmin)
            & (pre_freqs < fmax)
        )

        post_mask = (
            (post_freqs >= fmin)
            & (post_freqs < fmax)
        )

        pre_band = pre_mean[
            pre_mask
        ].mean()

        post_band = post_mean[
            post_mask
        ].mean()

        difference = (
            post_band - pre_band
        )

        band_results.append({
            "Participant": participant,
            "Condition": condition,
            "Condition_Name": condition_name,
            "Band": band_name,
            "Pre_dB": pre_band,
            "Post_dB": post_band,
            "Difference_dB": difference
        })

        print(
            band_name,
            "Pre:",
            round(pre_band, 3),
            "Post:",
            round(post_band, 3),
            "Difference:",
            round(difference, 3)
        )


    try:

        fig = pre_psd.plot_topomap(
            bands=bands,
            ch_type="eeg",
            normalize=False,
            show=False
        )

        fig.suptitle(
            f"Participant {participant}: "
            f"{condition_name} — Pre"
        )

        fig.savefig(
            condition_output
            / "pre_band_topomaps.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:

        print(
            "Could not create PRE topomap:",
            e
        )


    try:

        fig = post_psd.plot_topomap(
            bands=bands,
            ch_type="eeg",
            normalize=False,
            show=False
        )

        fig.suptitle(
            f"Participant {participant}: "
            f"{condition_name} — Post"
        )

        fig.savefig(
            condition_output
            / "post_band_topomaps.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:

        print(
            "Could not create POST topomap:",
            e
        )


    alpha_band = {
        "Alpha (8-12 Hz)": (8, 12)
    }


    try:

        fig = pre_psd.plot_topomap(
            bands=alpha_band,
            ch_type="eeg",
            normalize=False,
            show=False
        )

        fig.suptitle(
            f"Participant {participant}: "
            f"{condition_name} — Pre Alpha"
        )

        fig.savefig(
            condition_output
            / "pre_alpha_topomap.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:

        print(
            "Could not create PRE alpha topomap:",
            e
        )


    try:

        fig = post_psd.plot_topomap(
            bands=alpha_band,
            ch_type="eeg",
            normalize=False,
            show=False
        )

        fig.suptitle(
            f"Participant {participant}: "
            f"{condition_name} — Post Alpha"
        )

        fig.savefig(
            condition_output
            / "post_alpha_topomap.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:

        print(
            "Could not create POST alpha topomap:",
            e
        )


    print(
        f"Finished Participant {participant}, "
        f"Condition {condition}"
    )


for participant in participants:

    for condition in conditions:

        try:

            analyze_participant(
                participant,
                condition
            )

        except Exception as e:

            print(
                f"ERROR: Participant {participant}, "
                f"Condition {condition}: {e}"
            )


band_results_df = pd.DataFrame(
    band_results
)

band_results_df.to_csv(
    output_root
    / "all_participants_band_power.csv",
    index=False
)


alpha_results = band_results_df[
    band_results_df["Band"] == "Alpha"
].copy()

alpha_results.to_csv(
    output_root
    / "all_participants_alpha_power.csv",
    index=False
)


print("\nFinished all available participants and conditions.")

print(
    "Results saved in:",
    output_root.resolve()
)
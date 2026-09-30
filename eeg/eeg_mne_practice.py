import mne
import numpy as np
import matplotlib.pyplot as plt


print("MNE version:", mne.__version__)


def analyze_participant(participant, condition):
    pre_file = f"{participant}-{condition}-a.set"
    post_file = f"{participant}-{condition}-b.set"

    print("\n" + "=" * 60)
    print(f"Participant {participant}, Condition {condition}")
    print("=" * 60)

    pre = mne.io.read_raw_eeglab(
        pre_file,
        preload=True
    )

    post = mne.io.read_raw_eeglab(
        post_file,
        preload=True
    )

    print("\nPRE RECORDING")
    print(pre)

    print("\nPOST RECORDING")
    print(post)

    print("\nCHANNEL NAMES")
    print(pre.ch_names)

    print("\nSAMPLING FREQUENCY")
    print(pre.info["sfreq"])

    print("\nBAD CHANNELS")
    print(pre.info["bads"])

    print("\nPRE ANNOTATIONS")
    print(pre.annotations)

    print("\nPOST ANNOTATIONS")
    print(post.annotations)

    print("\nPRE DURATION")
    print(pre.times[-1], "seconds")

    print("\nPOST DURATION")
    print(post.times[-1], "seconds")


    pre.plot(
        duration=5,
        n_channels=min(18, len(pre.ch_names)),
        title=f"Participant {participant} Condition {condition} Pre"
    )

    post.plot(
        duration=5,
        n_channels=min(18, len(post.ch_names)),
        title=f"Participant {participant} Condition {condition} Post"
    )


    pre_clean = pre.copy()

    post_clean = post.copy()


    pre_clean.filter(
        l_freq=1,
        h_freq=50
    )

    post_clean.filter(
        l_freq=1,
        h_freq=50
    )


    pre_clean.set_eeg_reference(
        "average"
    )

    post_clean.set_eeg_reference(
        "average"
    )


    pre_psd = pre_clean.compute_psd(
        fmin=1,
        fmax=50
    )

    post_psd = post_clean.compute_psd(
        fmin=1,
        fmax=50
    )


    pre_psd.plot(
        picks="data",
        exclude="bads",
        amplitude=False
    )

    post_psd.plot(
        picks="data",
        exclude="bads",
        amplitude=False
    )


    pre_data, freqs = pre_psd.get_data(
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
        freqs,
        pre_mean,
        label="Pre-stimulation"
    )

    ax.plot(
        post_freqs,
        post_mean,
        label="Post-stimulation"
    )

    ax.set_xlabel("Frequency (Hz)")

    ax.set_ylabel(
        "Power Spectral Density (dB)"
    )

    ax.set_title(
        f"Participant {participant}, Condition {condition}: Pre vs Post PSD"
    )

    ax.legend()

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        f"participant{participant}_condition{condition}_pre_post_psd.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 45)
    }


    print("\nBAND POWER")

    for band_name, (fmin, fmax) in bands.items():

        pre_mask = (
            (freqs >= fmin) &
            (freqs < fmax)
        )

        post_mask = (
            (post_freqs >= fmin) &
            (post_freqs < fmax)
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

        events_pre, event_id_pre = (
            mne.events_from_annotations(
                pre
            )
        )

        print("\nPRE EVENT IDS")
        print(event_id_pre)

        print("\nPRE EVENTS")
        print(events_pre[:10])

    except Exception as e:

        print(
            "\nCould not extract pre events:",
            e
        )


    try:

        events_post, event_id_post = (
            mne.events_from_annotations(
                post
            )
        )

        print("\nPOST EVENT IDS")
        print(event_id_post)

        print("\nPOST EVENTS")
        print(events_post[:10])

    except Exception as e:

        print(
            "\nCould not extract post events:",
            e
        )


    print(
        f"\nFinished Participant {participant}, Condition {condition}"
    )


analyze_participant(
    participant=10,
    condition=1
)

analyze_participant(
    participant=11,
    condition=1
)


print("\nDone.")
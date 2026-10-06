import mne
import matplotlib.pyplot as plt


print("\nPARTICIPANT 11 CHANNELS")
print("=" * 60)

raw11 = mne.io.read_raw_eeglab(
    "en11/11-3-a.set",
    preload=True,
    verbose=False
)

print(raw11.ch_names)


print("\nPARTICIPANT 10 CONDITION 5")
print("=" * 60)

pre = mne.io.read_raw_eeglab(
    "en10/10-5-a.set",
    preload=True,
    verbose=False
)

post = mne.io.read_raw_eeglab(
    "en10/10-5-b.set",
    preload=True,
    verbose=False
)

channels = ["O1", "O2", "Oz", "Pz"]

pre.pick(channels)
post.pick(channels)

print("Baseline duration:", pre.times[-1])
print("Stim duration:", post.times[-1])

print("\nBaseline:")
print(pre.get_data().min(), pre.get_data().max())

print("\nStimulation:")
print(post.get_data().min(), post.get_data().max())


pre.compute_psd(
    fmin=1,
    fmax=50
).plot(
    average=True
)

plt.savefig(
    "p10_c5_baseline_psd.png",
    dpi=300
)

plt.close()


post.compute_psd(
    fmin=1,
    fmax=50
).plot(
    average=True
)

plt.savefig(
    "p10_c5_stimulation_psd.png",
    dpi=300
)

plt.close()


print("\nSaved PSD plots.")
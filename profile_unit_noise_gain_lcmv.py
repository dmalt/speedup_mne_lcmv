import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import sample

from mne.beamformer import apply_lcmv
from time import time  # dmalt
from make_lcmv_dmalt import make_lcmv_dmalt, make_lcmv_mne
from functools import partial

mne.set_log_level(verbose="ERROR")

print("MNE VERSION: ", mne.__version__)

data_path = sample.data_path()
raw_fname = data_path + "/MEG/sample/sample_audvis_raw.fif"
event_fname = data_path + "/MEG/sample/sample_audvis_raw-eve.fif"
fname_fwd = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
label_name = "Aud-lh"
fname_label = data_path + "/MEG/sample/labels/%s.label" % label_name
subjects_dir = data_path + "/subjects"

###############################################################################
# Get epochs
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info["bads"] = ["MEG 2443", "EEG 053"]  # 2 bads channels
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
picks = mne.pick_types(
    raw.info, meg=False, eeg=True, stim=True, eog=True, exclude="bads"
)
# picks = mne.pick_types(
#     raw.info, meg=True, eeg=False, stim=True, eog=True, exclude="bads"
# )

raw.set_eeg_reference(projection=True)

# Pick the channels of interest
raw.pick_channels([raw.ch_names[pick] for pick in picks])
# Re-normalize our empty-room projectors, so they are fine after subselection
raw.info.normalize_proj()

# Read epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(None, 0),
    preload=True,
    proj=True,
)
evoked = epochs.average()

forward = mne.read_forward_solution(fname_fwd)
forward = mne.convert_forward_solution(forward, surf_ori=True)
forward = mne.pick_types_forward(forward, meg=False, eeg=True)

# Compute regularized noise and data covariances
noise_cov = mne.compute_covariance(
    epochs, tmin=tmin, tmax=0, method="shrunk", rank=None
)
data_cov = mne.compute_covariance(
    epochs, tmin=0.04, tmax=0.15, method="shrunk", rank=None
)

pick_ori = "max-power"
descriptions = ["dmalt with jit", "dmalt without jit", "mne"]

make_lcmv_dmalt_jit = partial(make_lcmv_dmalt, is_use_jit=True)
make_lcmv_dmalt_nojit = partial(make_lcmv_dmalt, is_use_jit=False)
make_lcmv_funcs = [make_lcmv_dmalt_jit, make_lcmv_dmalt_nojit, make_lcmv_mne]

# -------- these parameters can be changed -------- #
n_runs = 10  # number of repetitions for better profiling
is_show_plots = True  # show stc on cortex for both solutions?
# ------------------------------------------------- #

stcs = []

print("=" * 80)
for make_lcmv, desc in zip(make_lcmv_funcs, descriptions):
    # compute unit-noise-gain beamformer with whitening of the leadfield and
    # data (enabled by passing a noise covariance matrix)

    make_lcmv_times = []
    compute_beamformer_times = []
    for i in range(n_runs):
        t1 = time()  # dmalt
        filters, compute_beamformer_dt = make_lcmv(
            evoked.info,
            forward,
            data_cov,
            reg=0.05,
            noise_cov=noise_cov,
            pick_ori=pick_ori,
            weight_norm="unit-noise-gain",
            rank=None,
        )
        t2 = time()

        make_lcmv_times.append((t2 - t1) * 1000)  # in milliseconds
        compute_beamformer_times.append(compute_beamformer_dt * 1000)

    # -------- report timings -------- #
    print("-" * 80)
    print(desc.upper())
    print()
    print(
        "%-18s: %-21s: %.2f ms"
        % ("make_lcmv", "first run", make_lcmv_times[0])
    )
    print(
        "%-18s: %-21s: mean = %7.2f ms; std = %5.2f"
        % (
            "make_lcmv",
            "2-nd and subseq. runs",
            np.mean(make_lcmv_times[1:]),
            np.std(make_lcmv_times[1:]),
        )
    )
    print(
        "%-18s: %-21s: mean = %7.2f ms; std = %5.2f"
        % (
            "compute_beamformer",
            "all times",
            np.mean(compute_beamformer_times),
            np.std(compute_beamformer_times),
        )
    )
    # -------------------------------- #

    # apply this spatial filter to source-reconstruct the evoked data
    stcs.append(apply_lcmv(evoked, filters, max_ori_out="signed"))
    print("-" * 80)
print("=" * 80)

if is_show_plots:
    fig, axes = plt.subplots(len(stcs), 1)
    for stc, ax, desc in zip(stcs, axes, descriptions):
        # Plot last stc in the brain in 3D with PySurfer if available
        brain = stc.plot(
            hemi="lh",
            views="lat",
            subjects_dir=subjects_dir,
            initial_time=0.1,
            time_unit="s",
            smoothing_steps=5,
            title=desc,
            size=800,
        )
        brain.add_text(0.4, 0.8, "Version: " + desc, "Version: ", font_size=12)
        im = brain.screenshot()
        brain.close()
        ax.imshow(im)

    fig.show()

    # make sure stc plots are not destroyed before we want them to
    while True:
        ans = input("Finish?")
        if ans:
            break

Description
-----------
Accelerated version of MNE-Python's
`mne.beamformer._compute_beamformer._compute_beamformer` and
profiling scripts to compare accelerated code vs original.

In `make_lcmv_dmalt.py` I've put two versions of the original
`make_lcmv` function:
1. `make_lcmv_mne`

    the same as the original `mne.beamformer.make_lcmv` execpt for it returns
    execution time of `_compute_beamformer`
2. `make_lcmv_dmalt`

    calls accelerated version of `_compute_beamformer` (compute_beamformer_dmalt._compute_beamformer_dmalt)
    



Instructions
------------

Clone the repository and run `python profile_unit_noise_gain_lcmv.py`.

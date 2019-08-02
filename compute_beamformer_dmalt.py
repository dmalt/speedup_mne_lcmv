import numpy as np
from numpy import linalg
from numba import jit
from mne.utils import _reg_pinv


def _beam_loop(n_sources, W, G, n_ori, G_x_Cm_inv_sq):
    """
    Compute 3x3 nominator and denominator matrices for max-power orientation
    estimation in unit-noise-gain vector beamformer (see 4.46 in [1])

    Matrices are stacked together into (n_sources, 3, 3) tensor and returned

    References
    ----------
    .. [1] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media

    """
    A = np.empty((n_sources, 3, 3))
    B = np.empty((n_sources, 3, 3))
    for k in range(n_sources):
        Wk = W[n_ori * k : n_ori * k + n_ori, :]
        Gk = G[:, n_ori * k : n_ori * k + n_ori]
        B[k, :, :] = np.dot(
            G_x_Cm_inv_sq[n_ori * k : n_ori * k + n_ori, :], Gk
        )
        A[k, :, :] = np.dot(Wk, Gk)

    return A, B


# Define the compiled version of beamformer loop
_beam_loop_jit = jit(_beam_loop, nopython=True, cache=True, fastmath=True)


def _compute_beamformer_dmalt(
    G,
    Cm,
    reg,
    n_orient,
    weight_norm,
    pick_ori,
    reduce_rank,
    rank,
    inversion,
    nn,
    orient_std,
    is_use_jit,
):
    """Works only for unit-noise-gain beamformer with max-power orientation

    Accelerated version of
    mne.beamformer._compute_beamformer._normalized_weights

    """
    Cm_inv, _, _ = _reg_pinv(Cm, reg, rank)

    Cm_inv_sq = np.matmul(Cm_inv, Cm_inv)
    W = np.matmul(G.T, Cm_inv)

    n_sources = G.shape[1] // n_orient
    G_x_Cm_inv_sq = np.matmul(G.T, Cm_inv_sq)

    if is_use_jit:
        A, B = _beam_loop_jit(n_sources, W, G, n_orient, G_x_Cm_inv_sq)
    else:
        A, B = _beam_loop(n_sources, W, G, n_orient, G_x_Cm_inv_sq)

    pwr_matr_stacked = linalg.solve(B, A)  # vectorized inv(B).dot(A)
    evals, evecs = linalg.eig(pwr_matr_stacked)

    max_eval_ind = evals.argmax(axis=1)
    max_ori = np.vstack(
        [evecs[i, :, max_eval_ind[i]] for i in range(len(evals))]
    )

    # set the (otherwise arbitrary) sign to match the normal
    sign = np.sign((max_ori * nn).sum(axis=1))
    sign[sign == 0] = 1
    max_ori *= sign[:, np.newaxis]

    # -------- filters in the orientation of max power -------- #
    W = W.reshape(n_sources, n_orient, -1)
    W = np.matmul(max_ori[:, np.newaxis, :], W).squeeze()
    # --------------------------------------------------------- #

    # -------- unit-noise-gain filters normalization -------- #
    tmp = np.matmul(B, max_ori[:, :, np.newaxis])
    pwr = np.matmul(max_ori[:, np.newaxis, :], tmp).ravel()
    denom = np.sqrt(pwr)
    W /= np.expand_dims(denom, axis=1)
    # -------------------------------------------------------- #

    return W

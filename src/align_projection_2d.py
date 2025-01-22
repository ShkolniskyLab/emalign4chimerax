import cmath
import math
import time
import numpy as np
from .commonline_R2 import cryo_normalize
from .common_finufft import cryo_pft
from .cryo_project_itay_finufft import cryo_project
from .genRotationsGrid import genRotationsGrid


def compute_commonlines_aux(Rj, candidate_rots, L):
    Nrot = candidate_rots.shape[2]
    Ckj = (-1) * np.ones(Nrot, dtype=int)
    Cjk = (-1) * np.ones(Nrot, dtype=int)
    Mkj = np.zeros(Nrot, dtype=int)

    for k in range(Nrot):
        Rk = np.transpose(candidate_rots[:, :, k])
        dot_RkRj = Rk[0, 2] * Rj[0, 2] + Rk[1, 2] * Rj[1, 2] + Rk[2, 2] * Rj[2, 2]
        if dot_RkRj < 0.999:
            Rk3 = Rk[2, :]
            Rj3 = Rj[2, :]

            clvec = np.array([[Rk3[1] * Rj3[2] - Rk3[2] * Rj3[1]],
                              [Rk3[2] * Rj3[0] - Rk3[0] * Rj3[2]],
                              [Rk3[0] * Rj3[1] - Rk3[1] * Rj3[0]]])

            cij = Rk @ clvec
            cji = Rj @ clvec

            alphaij = math.atan2(cij[1], cij[0])
            alphaji = math.atan2(cji[1], cji[0])

            alphaij = alphaij + np.pi
            alphaji = alphaji + np.pi

            l_ij = alphaij / (2 * np.pi) * L
            l_ji = alphaji / (2 * np.pi) * L

            ckj = int(round(l_ij) % L)
            cjk = int(round(l_ji) % L)

            Ckj[k] = ckj
            Cjk[k] = cjk
            Mkj[k] = 1

    return Ckj, Cjk, Mkj


def align_projection(projs, vol, inds_ref, starting_t, opt=None, log=None, show_log=False):
    """
    This function aligns given projection in a given volume.
    This is a secondary algorithm for cryo_align_vols.

    input:
        projs- projection images for the alignment.
        vol-reference volume.
        opt-Structure with optimizer options.

    output:
        Rots_est- size=3x3x(size(projs,3)). The estimated rotation matrices of
                  the projections. If we project the volume in these
                  orientations we will receive the same projections.
        Shifts- size=(size(projs,3))x2. The 2D estimated shift of the
                projections, first column contained the shift in the x-axis, and
                the secound colunm in the y-axis.
        corrs- size=size((projs,3))x2. Statistics of the alignment. The i'th
               entry of the first column contains the correlation of the common
               lines between the i'th image and all the reference images induced
               by the best matching rotation. The  i'th entry of the second
               column contains the mean matching correlation over all tested
               rotations.
        err_Rots- error calculation between the true rotations and the estimated
                  rotations.
        err_shifts- error calculation between the true shifts and the estimated
                    shifts, in x and y-axis.
    Options:
        opt.Nprojs- number of reference projections for the alignment. (default
                   is 30).
        opt.Rots - size=3x3x(size(Rots,3)). a set of candidate rotations.
    """
    # Check options:
    if hasattr(opt, 'Nprojs'):
        Nprojs = opt.Nprojs
    else:
        Nprojs = 50

    if hasattr(opt, 'Rots'):
        Rots = opt.Rots
    else:
        Rots = None

    # Define parameters:
    canrots = 1
    if Rots is None:
        canrots = 0
    n = np.size(vol, 0)
    n_r = math.ceil(n / 2)
    L = 360

    # Compute polar Fourier transform of projs:
    print_to_log(log, f"{get_time_stamp(starting_t)} Computing polar Fourier transform of unaligned query projections", show_log=show_log)
    projs_hat = cryo_pft(projs, n_r, L)[0]
    # Normalize polar Fourier transforms:
    print_to_log(log, f"{get_time_stamp(starting_t)} Normalizing the polar Fourier transform of unaligned query projections", show_log=show_log)
    projs_hat = cryo_normalize(projs_hat)
    n_projs = np.size(projs_hat, 2)

    # Generate candidate rotations and reference projections:
    print_to_log(log, f"{get_time_stamp(starting_t)} Generating {Nprojs} reference projections", show_log=show_log)
    if canrots == 0:
        Rots = genRotationsGrid(75)
    candidate_rots = Rots
    Nrot = np.size(candidate_rots, 2)
    print_to_log(log, f"{get_time_stamp(starting_t)} Using {Nrot} candidate rotations for the alignment", show_log=show_log)
    rots_ref = Rots[:, :, inds_ref]

    ref_projs = cryo_project(vol, rots_ref)
    ref_projs = np.transpose(ref_projs, (1, 0, 2))
    rots_ref = np.transpose(rots_ref, (1, 0, 2))  # the true rots

    # Compute polar Fourier transform of reference projections:
    print_to_log(log, f"{get_time_stamp(starting_t)} Computing polar Fourier transform of reference projections", show_log=show_log)
    refprojs_hat = cryo_pft(ref_projs, n_r, L)[0]

    # Normalize polar Fourier transforms:
    print_to_log(log, f"{get_time_stamp(starting_t)} Normalizing the polar Fourier transform of reference projections", show_log=show_log)
    refprojs_hat = cryo_normalize(refprojs_hat)

    # Compute the common lines between the candidate rotations and the references:
    print_to_log(log, f"{get_time_stamp(starting_t)} Computing the common lines between the projections", show_log=show_log)

    # t1 = time.perf_counter()
    Ckj = (-1) * np.ones((Nrot, Nprojs), dtype=int)
    Cjk = (-1) * np.ones((Nrot, Nprojs), dtype=int)
    Mkj = np.zeros((Nrot, Nprojs), dtype=int)
    for j in range(Nprojs):
        Rj = np.transpose(rots_ref[:, :, j])
        for k in range(Nrot):
            Rk = np.transpose(candidate_rots[:, :, k])

            # The following is an optimization of np.dot(Rk[:,2],Rj[:,2])
            dot_RkRj = Rk[0, 2] * Rj[0, 2] + Rk[1, 2] * Rj[1, 2] + Rk[2, 2] * Rj[2, 2]
            if dot_RkRj < 0.999:
                Rk3 = Rk[2, :]
                Rj3 = Rj[2, :]

                clvec = np.array([[Rk3[1] * Rj3[2] - Rk3[2] * Rj3[1]],
                                  [Rk3[2] * Rj3[0] - Rk3[0] * Rj3[2]],
                                  [Rk3[0] * Rj3[1] - Rk3[1] * Rj3[0]]])

                cij = Rk @ clvec
                cji = Rj @ clvec

                alphaij = math.atan2(cij[1], cij[0])
                alphaji = math.atan2(cji[1], cji[0])

                alphaij = alphaij + np.pi
                alphaji = alphaji + np.pi

                l_ij = alphaij / (2 * np.pi) * L
                l_ji = alphaji / (2 * np.pi) * L

                ckj = int(round(l_ij) % L)
                cjk = int(round(l_ji) % L)

                # Convert the returned indices ckj and cjk into 1-based
                Ckj[k, j] = ckj
                Cjk[k, j] = cjk
                Mkj[k, j] = 1

    # t2 = time.perf_counter()

    max_s = int(np.round(0.2 * np.size(projs_hat, 0)))  # set the maximum shift
    s_step = 0.5
    n_shifts = int((2 / s_step) * max_s + 1)  # always odd number (to have zero value without shift)
    max_r = np.size(projs_hat, 0)

    # The shifts in the r variable in the common lines:
    s_vec = np.linspace(-max_s, max_s, n_shifts).reshape((1, n_shifts))
    r_vec = np.arange(max_r).reshape((1, max_r))
    s_phases = np.exp((-2 * math.pi * cmath.sqrt(-1)) / (2 * max_r + 1) * (r_vec.conj().T @ s_vec))  # size (n_rXn_shift)

    # Main loop-compute the cross correlation:
    # computing the correlation between the common line, first choose the best shift, and then chose the best rotation.
    print_to_log(log, f"{get_time_stamp(starting_t)} Aligning unaligned projections using reference projections", show_log=show_log)
    Rots_est = np.zeros((3, 3, n_projs))
    corrs = np.zeros((n_projs, 2))  # statistics on common-lines matching
    shifts = np.zeros((2, n_projs))

    for projidx in range(n_projs):
        cross_corr_m = np.zeros((Nrot, Nprojs))
        for j in range(Nprojs):
            iidx = np.array(np.where(Mkj[:, j] != 0)).T
            conv_hat = (projs_hat[:, Ckj[iidx, j], projidx].conj() * refprojs_hat[:, Cjk[iidx, j], j]).reshape(
                (n_r, np.size(iidx, axis=0)))  # size of (n_rxsize(iidx))
            temp_corr = np.real(s_phases.conj().T @ conv_hat)
            cross_corr_m[iidx, j] = temp_corr.max(axis=0, initial=0).reshape(iidx.shape)
        # Calculating the mean of each row in cross_corr_m:
        cross_corr = (np.sum(cross_corr_m, axis=1) / np.sum(cross_corr_m > 0, axis=1)).reshape((Nrot, 1))
        # Find estimated rotation:
        bestRscore = np.amax(cross_corr)
        bestRidx = np.array(np.where(cross_corr == bestRscore))[0, 0]
        meanRscore = np.mean(cross_corr)
        corrs[projidx, 0] = bestRscore
        corrs[projidx, 1] = meanRscore
        Rots_est[:, :, projidx] = candidate_rots[:, :, bestRidx]

    return Rots_est, shifts, corrs


def print_to_log(log, msg, show_log=True):
    if show_log:
        log.info(msg)


def get_time_stamp(starting_t):
    full_t = (time.perf_counter() - starting_t) / 60
    t_minutes = math.floor(full_t)
    t_seconds = (full_t - t_minutes) * 60
    t_minutes_stamp = "0" + str(t_minutes) if t_minutes < 10 else str(t_minutes)
    t_seconds_stamp = str(t_seconds)[0:5] if t_seconds >= 10 else "0" + str(t_seconds)[0:4]
    time_stamp = t_minutes_stamp + ":" + t_seconds_stamp + " |  "
    return time_stamp

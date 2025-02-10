import math
import time
import numpy as np
from numpy import linalg as la
from .common_finufft import cryo_downsample
from .cryo_project_itay_finufft import cryo_project
from .genRotationsGrid import genRotationsGrid
from .align_projection_2d import align_projection
from .fastrotate3d import fastrotate3d
from . import fastrotate3d
from .register_translations_3d import register_translations_3d
from .reshift_vol import reshift_vol
from . import reshift_vol

MSG_PREFIX = "_______ | "


def align_volumes(vol1, vol2, starting_t=None, opt=None, show_log=True, session=None, original_vol1=None, original_vol2=None):
    """
    This function aligns vol2 according to vol1.
    Aligning vol2 to vol1 by finding the relative rotation, translation and reflection between vol1 and vol2,
    such that vol2 is best aligned with vol1.
    How to align the two volumes:
        The user should align vol2 according to vol1 using the parameters bestR, bestdx and reflect.
        If reflect=0 then there is no reflection between the volumes.
        In that case the user should first rotate vol2 by bestR and then reshift by bestdx.
        If reflect=1, then there is a reflection between the volumes.
        In that case the user should first reflect vol2 about the z axis using the flip function,
        then rotate the volume by bestR and finally reshift by bestdx.
    Input:
        vol1- 3D reference volume that vol2 should be aligned accordingly.
        vol2- 3D volume to be aligned.
        verbose - Set verbose to nonzero for verbose printouts (default is zero).
    Output:
        bestR - the estimated rotation between vol2 and vol1, such that bestR*vol2 will align vol2 to vol1.
        bestdx - size=3x1. the estimated translation between vol2 and vol1.
        reflect - indicator for reflection.
                  If reflect=1 then there is a reflection between vol1 and vol2, else reflect=0.
                  In order to align the volumes in the case of reflect=1, the user should first reflect vol2 about
                  the z axis, and then rotate by bestR.
        vol2aligned - vol2 after applying the est. transformation, so it's best aligned with vol1 (after optimization).
        bestcorr - the coorelation between vol1 and vol2aligned.
    Options:
        opt.downsample - Downsample the volume to this size (in pixels) for faster alignment. Default is 64.
                         Use larger value if alignment fails.
        opt.Nprojs - Number of projections to use for the alignment. Defult is 30.
    """
    # np.random.seed(114)

    # Check options:
    Nprojs, align_in_place, downsample, log, starting_t, mask = check_options(opt, session, starting_t)

    # Validate input:
    n_1, n_2 = validate_input(vol1, vol2)

    # Downsampling speeds up calculation, and does not seem to degrade accuracy:
    vol1_ds, vol2_ds = downsample_volumes(downsample, log, n_1, n_2, show_log, starting_t, vol1, vol2)

    # Aligning the downsampled volumes:
    R_est, estdx_ds, reflect, corr_v = align_ds_volumes(vol1_ds.copy(), vol2_ds.copy(), Nprojs, starting_t=starting_t, log=log, show_log=show_log)

    if mask:
        if original_vol1 is not None:
            vol1 = original_vol1.copy()
        if original_vol2 is not None:
            vol2 = original_vol2.copy()

    if reflect:
        vol2 = np.flip(vol2, axis=2)
        if align_in_place:
            print_to_log(log, f"{get_time_stamp(starting_t)} Flipping query volume before alignment", show_log=show_log)

    print_to_log(log, MSG_PREFIX + f"Correlation between downsampled aligned volumes: {corr_v:.4f}", show_log=show_log)

    bestR, bestdx, vol2aligned = align_original_volumes(R_est, align_in_place, log, show_log, starting_t, vol1, vol2)

    return bestR, bestdx, reflect, vol2aligned


def align_ds_volumes(vol1_ds, vol2_ds, Nprojs=25, starting_t=0.0, log=None, show_log=False, reselect_random=True):
    corr_ds_before = round(calculate_chimerax_correlation(vol1_ds, vol2_ds, center_data=False), 4)
    if reselect_random:
        print_to_log(log, MSG_PREFIX + f"Alignning downsampled volumes:", show_log=show_log)
        print_to_log(log, MSG_PREFIX + f"Correlation between downsampled volumes before alignment: {corr_ds_before:.4f}", show_log=show_log)

    # Generating 15236 rotation matrices (3x3) into Rots (3x3x15236):
    Rots = genRotationsGrid(75)

    # Extracting the number of rotations we generated into sz_Rots: (15236)
    sz_Rots = np.size(Rots, 2)

    print_to_log(log, f"{get_time_stamp(starting_t)} Generating {Nprojs} reference projections", show_log=show_log)
    rand_inds = np.random.randint(sz_Rots, size=Nprojs * 2)

    inds_to_align = rand_inds[0:Nprojs]
    inds_to_ref = rand_inds[Nprojs:]

    R_est, R_est_J = fast_alignment_3d(vol1_ds, vol2_ds, inds_to_align, inds_to_ref, Rots, Nprojs, starting_t, log=log, show_log=show_log)
    R_est, estdx_ds, reflect, corr_v = calculate_alignment_parameters(vol1_ds, vol2_ds, R_est, R_est_J)

    output_parameters = R_est, estdx_ds, reflect, corr_v

    if output_parameters[3] < corr_ds_before * 1.10:
        print_to_log(log, MSG_PREFIX + f"Switching assignment of rotation matrices and re-alignning", show_log=show_log)
        R_est_rev, R_est_J_rev = fast_alignment_3d(vol1_ds, vol2_ds, inds_to_ref, inds_to_align, Rots, Nprojs, starting_t, log=log, show_log=False)
        R_est_rev, estdx_ds_rev, reflect_rev, corr_v_rev = calculate_alignment_parameters(vol1_ds, vol2_ds, R_est_rev, R_est_J_rev)
        if corr_v_rev > corr_v:
            output_parameters = R_est_rev, estdx_ds_rev, reflect_rev, corr_v_rev

    if reselect_random and (output_parameters[3] < corr_ds_before * 1.10):
        # In the case where the results are still not well aligned, try another random selection and run just once more:
        print_to_log(log, MSG_PREFIX + f"Re-alignning downsampled volumes with different rotation matrices", show_log=show_log)
        R_est_1, estdx_ds_1, reflect_1, corr_v_1 = align_ds_volumes(vol1_ds, vol2_ds, starting_t=starting_t, Nprojs=Nprojs, log=log, show_log=show_log, reselect_random=False)
        if corr_v_1 > output_parameters[3]:
            output_parameters = R_est_1, estdx_ds_1, reflect_1, corr_v_1

    return output_parameters[0], output_parameters[1], output_parameters[2], output_parameters[3]


def fast_alignment_3d(vol1, vol2, inds_to_align, inds_to_ref, Rots, Nprojs, starting_t, log=None, show_log=False):
    """
    This function does the work for align_volumes.

    Input:
    vol1 - 3D reference volume that vol2 should be aligned accordingly.
    vol2 - 3D volume to be aligned.
    verbose - set verbose to nonzero for verbose printouts (default is zero).
    Nprojs - number of reference projections for the alignment.

    Output:
    Rest - the estimated rotation between vol_2 and vol_1 without reflection.
    Rest_J - the estimated rotation between vol_2 and vol_1 with reflection.
    """
    R_ref = Rots[:, :, inds_to_align]  # size (3,3,N_projs)
    ref_projs = cryo_project(vol2, R_ref)
    ref_projs = np.transpose(ref_projs, (1, 0, 2))
    R_ref = np.transpose(R_ref, (1, 0, 2))  # the true rotations

    # Align reference projections to vol1:
    class Struct:
        """
        Used to pass optimal paramters to the alignment function
        """
        pass

    opt = Struct()
    opt.Nprojs = Nprojs
    opt.Rots = Rots

    print_to_log(log, f"{get_time_stamp(starting_t)} Aligning reference projections of query map to reference map", show_log=show_log)
    R_tild = align_projection(ref_projs, vol1, inds_to_ref, starting_t, opt, log=log, show_log=show_log)  # size (3,3,N_projs)

    # Synchronization:
    # A synchronization algorithm is used in order to revel the symmetry elements of the reference projections.

    # Estimate X with or without reflection:
    R_tild = R_tild[0]
    X_mat = np.zeros((3, 3, Nprojs))
    X_mat_J = np.zeros((3, 3, Nprojs))
    J3 = np.diag([1, 1, -1])
    for i in range(Nprojs):
        X_mat[:, :, i] = R_ref[:, :, i] @ R_tild[:, :, i].T
        X_mat_J[:, :, i] = R_ref[:, :, i] @ (J3 @ R_tild[:, :, i] @ J3).T

    # Construct the synchronization matrix with and without reflection:
    X_ij = np.zeros((3 * Nprojs, 3 * Nprojs))
    X_ij_J = np.zeros((3 * Nprojs, 3 * Nprojs))
    for i in range(Nprojs):
        for j in range(i + 1, Nprojs):
            X_ij[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = X_mat[:, :, i].T @ X_mat[:, :, j]
            X_ij_J[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = X_mat_J[:, :, i].T @ X_mat_J[:, :, j]

    # Enforce symmetry:
    X_ij = X_ij + X_ij.T
    X_ij = X_ij + np.eye(np.size(X_ij, 0))
    X_ij_J = X_ij_J + X_ij_J.T
    X_ij_J = X_ij_J + np.eye(np.size(X_ij_J, 0))

    # Define v=[g_1.',..., g_N_projs.'].' (v is of size 3*N_projx3), then X_ij=v*v.', and Xij*v=N_projs*v.
    # Thus, v is an eigenvector of Xij. The matrix Xij should be of rank 3. Find the top 3 eigenvectors:
    # --> Without reflection:
    s, U = la.eigh(X_ij)
    ii = np.argsort(s, axis=0)[::-1]
    U = U[:, ii]
    V = U[:, 0:3]
    # --> With reflection:
    sJ, UJ = la.eigh(X_ij_J)
    iiJ = np.argsort(sJ, axis=0)[::-1]
    UJ = UJ[:, iiJ]
    VJ = UJ[:, 0:3]

    # Estimating G:
    # Estimate the group elemnts for each reference image.
    # G denotes the estimated group without reflection, and G_J with reflection.
    # This estimation is done from the eigenvector v by using a rounding algorithm over SO(3) for each 3x3 block of v.
    G = np.zeros((Nprojs, 3, 3))
    G_J = np.zeros((Nprojs, 3, 3))
    for i in range(Nprojs):
        B = V[3 * i:3 * (i + 1), :]
        u_tmp, _, v_tmp = la.svd(B)
        B_round = la.det(u_tmp @ v_tmp) * (u_tmp @ v_tmp)
        G[i, :, :] = B_round.T
        # With reflection:
        BJ = VJ[3 * i:3 * (i + 1), :]
        uJ_tmp, _, vJ_tmp = la.svd(BJ)
        BJ_round = la.det(uJ_tmp @ vJ_tmp) * (uJ_tmp @ vJ_tmp)
        G_J[i, :, :] = BJ_round.T

    # Set the global rotation to be an element from the symmetry group:
    # The global rotation from the synchronization can be any rotation matrix from SO(3).
    # So, in order to get the estimated symmetry elements to be from the symmetry group,
    # we set the global rotation to be also an element from the symmetry group.
    O1 = G[0, :, :].T
    O1_J = G_J[0, :, :].T
    G_est = np.zeros((Nprojs, 3, 3))
    G_J_est = np.zeros((Nprojs, 3, 3))
    for i in range(Nprojs):
        G_est[i, :, :] = O1 @ G[i, :, :]
        G_J_est[i, :, :] = O1_J @ G_J[i, :, :]

    # Estimating the rotation:
    # Estimate the two candidate orthogonal transformations.
    for i in range(Nprojs):
        X_mat[:, :, i] = X_mat[:, :, i] @ G_est[i, :, :].T
        X_mat_J[:, :, i] = X_mat_J[:, :, i] @ G_J_est[i, :, :].T
    X = np.mean(X_mat, axis=2)
    X_J = np.mean(X_mat_J, axis=2)
    # --> Without reflection:
    R = X
    U, _, V = la.svd(R)  # project R to the nearest rotation
    R_est = U @ V
    assert la.det(R_est) > 0
    R_est = R_est[:, [1, 0, 2]][[1, 0, 2]]
    R_est = R_est.T
    # --> With reflection:
    R_J = X_J
    U, _, V = la.svd(R_J)  # project R to the nearest rotation
    R_est_J = U @ V
    assert la.det(R_est_J) > 0
    R_est_J = R_est_J[:, [1, 0, 2]][[1, 0, 2]]
    R_est_J = R_est_J.T

    return R_est, R_est_J


def calculate_alignment_parameters(vol1_ds, vol2_ds, R_est, R_est_J):
    vol2_aligned_ds = fastrotate3d.fastrotate3d(vol2_ds, R_est)
    vol2_aligned_J_ds = fastrotate3d.fastrotate3d(vol2_ds, R_est_J)

    vol2_aligned_J_ds = np.flip(vol2_aligned_J_ds, axis=2)
    estdx_ds = register_translations_3d(vol1_ds, vol2_aligned_ds)
    estdx_J_ds = register_translations_3d(vol1_ds, vol2_aligned_J_ds)

    if np.size(estdx_ds) != 3 or np.size(estdx_J_ds) != 3:
        raise Warning("***** Translation estimation failed *****")

    # Prepare FFTW data to avoid unnecessary calaculations:
    vol2_aligned_ds = reshift_vol.reshift_vol_int(vol2_aligned_ds, estdx_ds)
    vol2_aligned_J_ds = reshift_vol.reshift_vol_int(vol2_aligned_J_ds, estdx_J_ds)

    no1 = round(calculate_chimerax_correlation(vol1_ds, vol2_aligned_ds, center_data=False), 4)
    no2 = round(calculate_chimerax_correlation(vol1_ds, vol2_aligned_J_ds, center_data=False), 4)

    # Check if volumes are reflected:
    reflect = 0
    corr_v = no1
    if no2 > no1:
        J3 = np.diag([1, 1, -1])
        corr_v = no2
        R_est = R_est_J
        R_est = J3 @ R_est @ J3
        estdx_ds = estdx_J_ds
        reflect = 1

    return R_est, estdx_ds, reflect, corr_v


def align_original_volumes(R_est, align_in_place, log, show_log, starting_t, vol1, vol2):
    print_to_log(log, f"{get_time_stamp(starting_t)} Applying the calculated rotation to the original volume", show_log=show_log)
    bestR = R_est  # refinement option is controled from emalign_cmd, so for now R_est is the best rotation
    vol2aligned = fastrotate3d.fastrotate3d(vol2, bestR)

    print_to_log(log, f"{get_time_stamp(starting_t)} Estimating shift for the original volume", show_log=show_log)
    bestdx = register_translations_3d(vol1, vol2aligned)

    if not align_in_place:
        # When original volumes are of different sizes, we adjust the translations and align the volumes in emalign_cmd:
        translation_msg = MSG_PREFIX + f"Shift before pixel adjustment: [{bestdx[0]:.3f}, {bestdx[1]:.3f}, {bestdx[2]:.3f}]"
        print_to_log(log, translation_msg, show_log=show_log)
        vol2aligned = vol2
    else:
        print_to_log(log, f"{get_time_stamp(starting_t)} Shifting the original volume\n", show_log=show_log)
        if (np.round(bestdx) == bestdx).all():
            # Use fast method:
            vol2aligned = reshift_vol.reshift_vol_int(vol2aligned, bestdx)
        else:
            vol2aligned = reshift_vol.reshift_vol(vol2aligned, bestdx)
    vol2aligned = vol2aligned.astype(np.float32)

    return bestR, bestdx, vol2aligned


# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------- Additional Functions: -------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def check_options(opt, session, starting_t):
    if session is not None:
        log = session.logger
    else:
        log = None

    class Struct:
        pass

    if opt is None:
        opt = Struct()

    if starting_t is None:
        starting_t = time.perf_counter()

    if hasattr(opt, 'Nprojs'):
        Nprojs = opt.Nprojs
    else:
        Nprojs = 25

    if hasattr(opt, 'downsample'):
        downsample = opt.downsample
    else:
        downsample = 64

    if hasattr(opt, 'options'):
        align_in_place = opt.options[0]
    else:
        align_in_place = True

    if hasattr(opt, 'masking'):
        mask = opt.masking
    else:
        mask = False

    return Nprojs, align_in_place, downsample, log, starting_t, mask


def validate_input(vol1, vol2):
    """
    Input volumes must be 3-dimensional, where all dimensions must be equal. This restriction can be removed,
    but then the calculation of nr (radial resolution in the Fourier domain) should be adjusted accordingly.
    Both vol_1 and vol_2 must have the same dimensions.
    This function ensures the input vol1 and vol2 are valid and returns their respective shapes - n_1, n_2.
    """
    n_1 = np.shape(vol1)
    n_2 = np.shape(vol2)
    assert np.size(n_1) == 3 and np.size(n_2) == 3, "Inputs must be 3D"
    assert n_1[0] == n_1[1] == n_1[2], "All dimensions of input volumes must be equal"
    assert n_2[0] == n_2[1] == n_2[2], "All dimensions of input volumes must be equal"
    return n_1, n_2


def downsample_volumes(downsample, log, n_1, n_2, show_log, starting_t, vol1, vol2):
    """
    This function downsamples the reference and query volumes according to the given parameter downsample.
    It ensures that we can't align volumes with different sizes and that we can't align volumes without downsampling,
    if the volumes are > 256 pixels.
    """
    if n_1[0] == n_2[0]:
        n = n_1[0]
        n_ds = downsample
        if downsample is None:
            assert n <= 256, "Input volume is more than 256 pixels, must downsample"
            print_to_log(log,
                         f"{get_time_stamp(starting_t)} Downsampling volumes from {n},{n},{n} to {n_ds},{n_ds},{n_ds}",
                         show_log=show_log)
            vol1_ds = cryo_downsample(vol1, (n_ds, n_ds, n_ds))
            vol2_ds = cryo_downsample(vol2, (n_ds, n_ds, n_ds))
        else:
            assert downsample <= n, "Downsample must be less than input volume size"
            print_to_log(log,
                         f"{get_time_stamp(starting_t)} Downsampling volumes from {n},{n},{n} to {n_ds},{n_ds},{n_ds}",
                         show_log=show_log)
            vol1_ds = cryo_downsample(vol1, (n_ds, n_ds, n_ds))
            vol2_ds = cryo_downsample(vol2, (n_ds, n_ds, n_ds))
    else:
        assert n_1[0] == n_2[0], "Input volumes have different sizes"
        vol1_ds = vol1
        vol2_ds = vol2

    return vol1_ds, vol2_ds


def calculate_chimerax_correlation(map1, map2, center_data=True):
    """
    Calculate the correlation between two density maps, with an option to calculate about the mean data value.

    Parameters:
    map1 (numpy.ndarray): The first density map.
    map2 (numpy.ndarray): The second density map.
    center_data (bool): If True, calculate the correlation about the mean data value. If False, use raw data.

    Returns:
    float: The correlation coefficient.
    """
    # Ensure the maps have the same shape:
    assert map1.shape == map2.shape, "Maps must have the same shape."

    # Flatten the maps to 1D arrays:
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()

    if center_data:
        # Subtract the mean from each map:
        map1_centered = map1_flat - np.mean(map1_flat)
        map2_centered = map2_flat - np.mean(map2_flat)
    else:
        # Use raw data without centering:
        map1_centered = map1_flat
        map2_centered = map2_flat

    # Calculate the dot product of the maps:
    dot_product = np.dot(map1_centered, map2_centered)

    # Calculate the norms of the maps:
    norm_map1 = np.linalg.norm(map1_centered)
    norm_map2 = np.linalg.norm(map2_centered)

    # Calculate the correlation as the cosine of the angle between the maps:
    correlation = dot_product / (norm_map1 * norm_map2)

    return correlation


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

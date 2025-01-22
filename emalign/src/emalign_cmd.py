import math
import time
import numpy as np
from chimerax.map_fit import fitcmd
from chimerax.map_fit.fitmap import map_overlap_and_correlation as calculate_stats
from chimerax.map_data import arraygrid
from chimerax.core.errors import UserError
from . import align_volumes_3d, reshift_vol, fastrotate3d
from .common_finufft import cryo_downsample, cryo_crop
from .utils import fuzzy_mask

MSG_PREFIX = "_______ | "


def register_emalign_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import IntArg
    from chimerax.map.mapargs import MapArg

    emalign_desc = CmdDesc(
        required=[
            ('ref_map', MapArg),
        ],
        keyword=[
            ('query_map', MapArg)

        ],
        optional=[
            ('downsample', IntArg),
            ('projections', IntArg)
        ],
        required_arguments=['ref_map', 'query_map'],
        synopsis='Perform EM-alignment of two density maps'
    )
    register('volume emalign', emalign_desc, emalign, logger=logger)


def emalign(session, ref_map, query_map, downsample=64, projections=50, mask=False, show_log=True, show_param=True,
            refine=False):
    log = session.logger

    # Calculate overlap and correlation (calculated using only data above contour level from first map):
    overlap_before, corr_before, corr_m_before = calculate_stats(query_map, ref_map, False)

    print_to_log(log, "-" * 17 + "\nDetailed Log:\n" + "-" * 17, show_log=show_log)

    ref = ref_map.data
    query = query_map.data

    # Save original parameters of ref_map and query_map: {origin, step, cell_angles, rotation, symmetries, name}
    ref_dict = {}
    query_dict = {}
    keys = ["origin", "step", "cell_angles", "rotation", "symmetries", "name"]
    ref_map_values = [ref.origin, ref.step, ref.cell_angles, ref.rotation, ref.symmetries, ref.name]
    query_map_values = [query.origin, query.step, query.cell_angles, query.rotation, query.symmetries, query.name]
    for i in range(len(keys)):
        ref_dict[keys[i]] = ref_map_values[i]
        query_dict[keys[i]] = query_map_values[i]

    grid_ref_map = ref_map.full_matrix().T
    grid_query_map = query_map.full_matrix().T

    ref_vol = np.ascontiguousarray(grid_ref_map)
    query_vol = np.ascontiguousarray(grid_query_map)

    ref_vol, query_vol = validate_input(ref_vol, query_vol)

    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = projections
    opt.downsample = downsample

    # Grid size:
    N_ref = np.shape(ref_vol)[0]
    N_query = np.shape(query_vol)[0]

    # Pixel:
    pixel_ref = ref_dict.get("step")[0]
    pixel_query = query_dict.get("step")[0]

    # Start the stopwatch to measure run-time:
    t1 = time.perf_counter()

    # Create copies of the volumes to run the alignment on:
    ref_vol_copy = ref_vol.copy()
    query_vol_copy = query_vol.copy()

    # Handling the case when volumes have same pixel size but different grid size:
    if (round(pixel_query, 3) == round(pixel_ref, 3)) and (N_query != N_ref):
        # Calculate new pixel size for the smaller volume:
        if N_ref < N_query:
            # In this case we want to enlarge pixel_ref:
            print_to_log(log, f"{get_time_stamp(t1)} Calculating new pixel size for reference volume",
                         show_log=show_log)
            pixel_ref = (N_query / N_ref) * pixel_ref
            ref_dict["step"] = (pixel_ref, pixel_ref, pixel_ref)
            # Create GridData object with aligned query_vol but with the original query_map parameters:
            ref_map_grid_data = arraygrid.ArrayGridData(ref_vol.T, origin=ref_dict.get("origin"),
                                                        step=ref_dict.get("step"),
                                                        cell_angles=ref_dict.get("cell_angles"),
                                                        rotation=ref_dict.get("rotation"),
                                                        symmetries=ref_dict.get("symmetries"),
                                                        name=ref_dict.get("name"))
            # Replace the data in the original query_map:
            ref_map.replace_data(ref_map_grid_data)
        elif N_query < N_ref:
            # In this case we want to enlarge pixel_query:
            print_to_log(log, f"{get_time_stamp(t1)} Calculating new pixel size for query volume", show_log=show_log)
            pixel_query = (N_ref / N_query) * pixel_query
            query_dict["step"] = (pixel_query, pixel_query, pixel_query)

        print_to_log(log, MSG_PREFIX + f"Updated pixel sizes: pixel_ref = {round(pixel_ref, 3)}, pixel_query = {round(pixel_query, 3)}", show_log=show_log)

    if pixel_query > pixel_ref:
        if mask:
            print_to_log(log, f"{get_time_stamp(t1)} Using masking to align center 90% of the volumes energy",
                         show_log=show_log)
            optimal_radius = calc_3D_radius(ref_vol)
            r0_factor = optimal_radius / N_ref

            m1 = fuzzy_mask([N_ref, N_ref, N_ref], dtype=np.float32, r0=r0_factor * N_ref)
            m2 = fuzzy_mask([N_query, N_query, N_query], dtype=np.float32, r0=r0_factor * N_query)

            ref_vol_copy = ref_vol_copy * m1
            query_vol_copy = query_vol_copy * m2

        # query_vol has the bigger pixel size ---> downsample ref_vol to N_ref_ds and then crop it to N_query:
        N_ref_ds = math.floor(N_ref * (pixel_ref / pixel_query))

        # Downsample ref_vol from N_ref to N_ref_ds:
        print_to_log(log,
                     f"{get_time_stamp(t1)} Downsampling the reference volume to grid size {N_ref_ds},{N_ref_ds},{N_ref_ds}",
                     show_log=show_log)
        ref_vol_ds = cryo_downsample(ref_vol_copy, (N_ref_ds, N_ref_ds, N_ref_ds))

        # Crop ref_vol from N_ref_ds to N_query:
        print_to_log(log,
                     f"{get_time_stamp(t1)} Cropping the reference volume to grid size {N_query},{N_query},{N_query}",
                     show_log=show_log)
        ref_vol_cropped = cryo_crop(ref_vol_ds.copy(), (N_query, N_query, N_query))

        opt.options = [False]

        # At this point both volumes are the same dimension, run the alignment:
        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol_cropped,
                                                                                   query_vol_copy,
                                                                                   starting_t=t1,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   session=session)
        if reflect:
            print_to_log(log, f"{get_time_stamp(t1)} Flipping query volume before alignment", show_log=show_log)
            query_vol = np.flip(query_vol, axis=2)

        # Rotate the volumes:
        print_to_log(log, f"{get_time_stamp(t1)} Applying the calculated rotation to the original volume",
                     show_log=show_log)
        query_vol_aligned = fastrotate3d.fastrotate3d(query_vol, bestR)

        # Translate the volumes:
        print_to_log(log, f"{get_time_stamp(t1)} Shifting the original volume\n", show_log=show_log)
        bestdx = (pixel_query / pixel_ref) * bestdx
        if (np.round(bestdx) == bestdx).all():
            # Use fast method:
            query_vol_aligned = reshift_vol.reshift_vol_int(query_vol_aligned, bestdx)
        else:
            query_vol_aligned = reshift_vol.reshift_vol(query_vol_aligned, bestdx)

        query_vol_aligned = query_vol_aligned.astype(np.float32)

    elif pixel_ref > pixel_query:  # ref_vol has the bigger pixel size and the smaller volume size
        if mask:
            print_to_log(log, f"{get_time_stamp(t1)} Using masking to align center 90% of the volumes energy",
                         show_log=show_log)
            optimal_radius = calc_3D_radius(query_vol)
            r0_factor = optimal_radius / N_query

            m1 = fuzzy_mask([N_ref, N_ref, N_ref], dtype=np.float32, r0=r0_factor * N_ref)
            m2 = fuzzy_mask([N_query, N_query, N_query], dtype=np.float32, r0=r0_factor * N_query)

            ref_vol_copy = ref_vol_copy * m1
            query_vol_copy = query_vol_copy * m2

        N_query_ds = math.floor(N_query * (pixel_query / pixel_ref))

        # Downsample query_vol_copy_ds from N_query to N_query_ds:
        print_to_log(log,
                     f"{get_time_stamp(t1)} Downsampling the reference volume to grid size {N_query_ds},{N_query_ds},{N_query_ds}",
                     show_log=show_log)
        query_vol_copy_ds = cryo_downsample(query_vol_copy, (N_query_ds, N_query_ds, N_query_ds))

        # Crop query_vol from N_ref_ds to N_ref:
        print_to_log(log, f"{get_time_stamp(t1)} Cropping the query volume to grid size {N_ref},{N_ref},{N_ref}",
                     show_log=show_log)
        query_vol_copy_cropped = cryo_crop(query_vol_copy_ds.copy(), (N_ref, N_ref, N_ref))

        opt.options = [False]

        bestR, bestdx, reflect, vol_aligned = align_volumes_3d.align_volumes(ref_vol_copy,
                                                                             query_vol_copy_cropped,
                                                                             starting_t=t1,
                                                                             opt=opt,
                                                                             show_log=show_log,
                                                                             session=session)

        if reflect:
            print_to_log(log, f"{get_time_stamp(t1)} Flipping query volume before alignment", show_log=show_log)
            query_vol = np.flip(query_vol, axis=2)

        # Rotate the volumes:
        print_to_log(log, f"{get_time_stamp(t1)} Applying the calculated rotation to the original volume",
                     show_log=show_log)
        query_vol_aligned = fastrotate3d.fastrotate3d(query_vol, bestR)

        # Translate the volumes:
        print_to_log(log, f"{get_time_stamp(t1)} Shifting the original volume\n", show_log=show_log)
        bestdx = (pixel_ref / pixel_query) * bestdx
        if (np.round(bestdx) == bestdx).all():
            # Use fast method:
            query_vol_aligned = reshift_vol.reshift_vol_int(query_vol_aligned, bestdx)
        else:
            query_vol_aligned = reshift_vol.reshift_vol(query_vol_aligned, bestdx)

        query_vol_aligned = query_vol_aligned.astype(np.float32)

    else:
        if mask:
            print_to_log(log, f"{get_time_stamp(t1)} Using masking to align center 90% of the volumes energy",
                         show_log=show_log)
            optimal_radius = calc_3D_radius(query_vol)
            r0_factor = optimal_radius / N_query

            m1 = fuzzy_mask([N_ref, N_ref, N_ref], dtype=np.float32, r0=r0_factor * N_ref)
            m2 = fuzzy_mask([N_query, N_query, N_query], dtype=np.float32, r0=r0_factor * N_query)

            ref_vol_copy = ref_vol_copy * m1
            query_vol_copy = query_vol_copy * m2

        opt.masking = mask
        original_ref_vol = ref_vol.copy()
        original_query_vol = query_vol.copy()

        bestR, bestdx, reflect, query_vol_aligned = align_volumes_3d.align_volumes(ref_vol_copy,
                                                                                   query_vol_copy,
                                                                                   starting_t=t1,
                                                                                   opt=opt,
                                                                                   show_log=show_log,
                                                                                   session=session,
                                                                                   original_vol1=original_ref_vol,
                                                                                   original_vol2=original_query_vol)

    t2 = time.perf_counter()
    print_to_log(log, f"Aligning the volumes using EMalign took {t2 - t1:.2f} seconds\n")
    print_param(log, bestR, bestdx, reflect, show_param)

    # Create GridData object with aligned query_vol but with the original query_map parameters:
    aligned_map_grid_data = arraygrid.ArrayGridData(query_vol_aligned.T, origin=query_dict.get("origin"),
                                                    step=query_dict.get("step"),
                                                    cell_angles=query_dict.get("cell_angles"),
                                                    rotation=query_dict.get("rotation"),
                                                    symmetries=query_dict.get("symmetries"),
                                                    name=query_dict.get("name"))

    # Replace the data in the original query_map:
    query_map.replace_data(aligned_map_grid_data)

    overlap, corr, corr_m = calculate_stats(query_map, ref_map, False)

    # Perform additional refinement with Fit in Map:
    if refine:
        print_to_log(log, "Applying 'Fit in Map' to perform additional refinement:")
        # Run fitmap to fit query_map in ref_map:
        fitcmd.fit_map_in_map(query_map, ref_map, metric="correlation", envelope=True, zeros=False, shift=True,
                              rotate=True, move_whole_molecules=True, map_atoms=None,
                              max_steps=2000, grid_step_min=0.01, grid_step_max=0.5, log=log)

    print_to_log(log, "-" * 88)
    print_to_log(log, "Stats before alignment:")
    print_to_log(log, f"correlation = {corr_before:.4f}, correlation about mean = {corr_m_before:.4f}, overlap = {overlap_before:.3f}")
    print_to_log(log, "-" * 88)
    print_to_log(log, "Stats after alignning with EMalign:")
    print_to_log(log, f"correlation = {corr:.4f}, correlation about mean = {corr_m:.4f}, overlap = {overlap:.3f}")
    print_to_log(log, "-" * 88)

    if refine:
        print_to_log(log, "Stats after refining with Fit in Map:")
        overlap, corr, corr_m = calculate_stats(query_map, ref_map, False)
        print_to_log(log, f"correlation = {corr:.4f}, correlation about mean = {corr_m:.4f}, overlap = {overlap:.3f}")
        print_to_log(log, "-" * 88)


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- Helper Functions: ---------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def generate_points_and_distances(shape):
    x, y, z = np.indices(shape)
    center = np.array(shape) // 2
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    distances = distances.flatten()
    return points, distances


def calc_3D_radius(density_map, energy_fraction=0.90):
    shape = density_map.shape
    # print("Create a list of all points and their distances from the center")
    points, distances = generate_points_and_distances(shape)

    # Compute the energy at each point
    # print("Compute the total energy")
    energy_values = density_map.flatten() ** 2

    # Sort distances and corresponding energy values
    # print("Sort distances and corresponding energy values")
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_energy_values = energy_values[sorted_indices]

    # Compute the cumulative energy
    # print("Compute cumulative energy")
    cumulative_energy = np.cumsum(sorted_energy_values)

    # Total energy
    total_energy = cumulative_energy[-1]

    # Find the radius where cumulative energy reaches the specified fraction
    # print("Find the radius where cumulative energy reaches the specified fraction")
    target_energy = total_energy * energy_fraction
    radius_index = np.searchsorted(cumulative_energy, target_energy)

    return sorted_distances[radius_index]


def validate_input(ref_vol, query_vol):
    # Handle the case where ref_vol is 4D:
    if (ref_vol.ndim == 4) and (ref_vol.shape[-1] == 1):
        ref_vol = np.squeeze(ref_vol)
    elif ref_vol.ndim != 3:
        raise UserError("Volumes must be three-dimensional or four-dimensional with singleton first dimension ")

    # Handle the case where query_vol is 4D:
    if (query_vol.ndim == 4) and (query_vol.shape[-1] == 1):
        query_vol = np.squeeze(query_vol)
    elif query_vol.ndim != 3:
        raise UserError("Volumes must be three-dimensional or four-dimensional with singleton first dimension ")

    if not ((ref_vol.shape[1] == ref_vol.shape[0]) and (ref_vol.shape[2] == ref_vol.shape[0])
            and (ref_vol.shape[0] % 2 == 0)):
        raise UserError("All three dimensions of input volumes must be equal and even")

    if not ((query_vol.shape[1] == query_vol.shape[0]) and (query_vol.shape[2] == query_vol.shape[0])
            and (query_vol.shape[0] % 2 == 0)):
        raise UserError("All three dimensions of input volumes must be equal and even")

    return ref_vol, query_vol


def print_param(log, bestR, bestdx, reflection, show_param):
    if show_param:
        log.info('Rotation:')
        log.info(f'[[{bestR[0, 0]:.3f} {bestR[0, 1]:.3f} {bestR[0, 2]:.3f}],')
        log.info(f'[{bestR[1, 0]:.3f} {bestR[1, 1]:.3f} {bestR[1, 2]:.3f}]')
        log.info(f'[{bestR[2, 0]:.3f} {bestR[2, 1]:.3f} {bestR[2, 2]:.3f}]]')
        log.info(f'Translation:\n [{bestdx[0]:.3f}, {bestdx[1]:.3f}, {bestdx[2]:.3f}]')
        log.info(f'Reflection:\n {reflection}\n')


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

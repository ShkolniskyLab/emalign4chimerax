import numpy as np
from scipy.special import erf


def cart2pol(x, y):
    """
    Convert Cartesian to Polar Coordinates. All input arguments must be the same shape.

    :param x: x-coordinate in Cartesian space
    :param y: y-coordinate in Cartesian space
    :return: A 2-tuple of values:
        theta: angular coordinate/azimuth
        r: radial distance from origin
    """
    return np.arctan2(y, x), np.hypot(x, y)


def cart2sph(x, y, z):
    """
    Transform cartesian coordinates to spherical. All input arguments must be the same shape.

    :param x: X-values of input coordinates.
    :param y: Y-values of input coordinates.
    :param z: Z-values of input coordinates.
    :return: A 3-tuple of values, all the same shape as the inputs.
        (<azimuth>, <elevation>, <radius>)
        azimuth and elevation are returned in radians.

    This function is equivalent to MATLAB's cart2sph function.
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def _mgrid_slice(n, shifted, normalized):
    """
    Util to generate a `slice` representing a 1d linspace
    as expected by `np.mgrid`.

    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: `slice` to be used by `np.mgrid`.
    """

    num_points = n * 1j
    start = -n // 2 + 1
    end = n // 2

    if shifted and n % 2 == 0:
        start -= 1 / 2
        end -= 1 / 2
    elif n % 2 == 0:
        start -= 1
        end -= 1

    if normalized:
        # Compute the denominator for normalization
        denom = n / 2
        if shifted and n % 2 == 0:
            denom -= 1 / 2

        # Apply the normalization
        start /= denom
        end /= denom

    return slice(start, end, num_points)


def grid_1d(n, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate one dimensional grid.

    :param n: the number of grid points.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :param dtype: the data type of the grid points.
    :return: the rectangular and polar coordinates of all grid points.
    """

    r = x = np.mgrid[_mgrid_slice(n, shifted, normalized)].astype(dtype)

    return {"x": x, "r": r}


def grid_2d(n, shifted=False, normalized=True, indexing="yx", dtype=np.float32):
    """
    Generate two-dimensional grid.

    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :param indexing: 'yx' (C) or 'xy' (F), defaulting to 'yx'.
        See https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    :param dtype: the data type of the grid points.
    :return: the rectangular and polar coordinates of all grid points.
    """

    grid = _mgrid_slice(n, shifted, normalized)
    y, x = np.mgrid[grid, grid].astype(dtype)
    if indexing == "xy":
        x, y = y, x
    elif indexing != "yx":
        raise RuntimeError(
            f"grid_2d indexing {indexing} not supported." "  Try 'xy' or 'yx'"
        )

    phi, r = cart2pol(x, y)

    return {"x": x, "y": y, "phi": phi, "r": r}


def grid_3d(n, shifted=False, normalized=True, indexing="zyx", dtype=np.float32):
    """
    Generate three-dimensional grid.

    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :param indexing: 'zyx' (C) or 'xyz' (F), defaulting to 'zyx'.
        See https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    :param dtype: the data type of the grid points.
    :return: the rectangular and spherical coordinates of all grid points.
    """

    grid = _mgrid_slice(n, shifted, normalized)
    z, y, x = np.mgrid[grid, grid, grid].astype(dtype)

    if indexing == "xyz":
        x, y, z = z, y, x
    elif indexing != "zyx":
        raise RuntimeError(
            f"grid_3d indexing {indexing} not supported." "  Try 'xyz' or 'zyx'"
        )

    phi, theta, r = cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside cart2sph?
    theta = np.pi / 2 - theta

    return {"x": x, "y": y, "z": z, "phi": phi, "theta": theta, "r": r}


def fuzzy_mask(L, dtype, r0=None, risetime=None):
    """
    Create a centered 1D to 3D fuzzy mask of radius r0.

    Made with an error function with effective rise time.

    :param L: The sizes of image in tuple structure. Must be 1D, 2D square,
        or 3D cube.
    :param dtype: dtype for fuzzy mask.
    :param r0: The specified radius. Defaults to floor(0.45 * L)
    :param risetime: The rise time for `erf` function. Defaults to floor(0.05 * L)

    :return: The desired fuzzy mask
    """
    # Note: default values for r0 and risetime are from Matlab common-lines code.
    if r0 is None:
        r0 = np.floor(0.45 * L[0])
    if risetime is None:
        risetime = np.floor(0.05 * L[0])

    dim = len(L)
    axes = ["x"]
    grid_kwargs = {"n": L[0], "shifted": False, "normalized": False, "dtype": dtype}

    if dim == 1:
        grid = grid_1d(**grid_kwargs)

    elif dim == 2:
        if not (L[0] == L[1]):
            raise ValueError(f"A 2D fuzzy_mask must be square, found L={L}.")
        grid = grid_2d(**grid_kwargs)
        axes.insert(0, "y")

    elif dim == 3:
        if not (L[0] == L[1] == L[2]):
            raise ValueError(f"A 3D fuzzy_mask must be cubic, found L={L}.")
        grid = grid_3d(**grid_kwargs)
        axes.insert(0, "y")
        axes.insert(0, "z")

    else:
        raise RuntimeError(
            f"Only 1D, 2D, or 3D fuzzy_mask supported. Found {dim}-dimensional `L`."
        )

    XYZ_sq = [grid[axis] ** 2 for axis in axes]
    R = np.sqrt(np.sum(XYZ_sq, axis=0))
    k = 1.782 / risetime
    m = 0.5 * (1 - erf(k * (R - r0)))

    return m

import numpy as np
from scipy import fft


def register_translations_3d(vol1, vol2):
    """
    REGISTER_TRANSLATIONS_3D  Estimate relative shift between two volumes.
    register_translations_3d(vol1,vol2,refdx)
      Estimate the relative shift between two volumes vol1 and vol2 to
      integral pixel accuracy. The function uses phase correlation to
      estimate the relative shift to within one pixel accuray.

      Input parameters:
      vol1,vol2 Two volumes to register. Volumes must be odd-sized.
      refidx    Two-dimensional vector with the true shift between the images,
                used for debugging purposes. (Optional)
      Output parameters
      estidx  A two-dimensional vector of how much to shift vol2 to aligh it
              with vol1. Returns -1 on error.
      err     Difference between estidx and refidx.
    """

    # Take Fourer transform of both volumes and compute the phase correlation factors:
    hats1 = fft.fftn(vol1)  # compute centered Fourier transform
    hats2 = fft.fftn(vol2)

    tmp1 = hats1 * np.conj(hats2)
    tmp2 = abs(tmp1)

    bool_idx = tmp2 < 2 * np.finfo(vol1.dtype).eps
    tmp2[bool_idx] = 1  # avoid division by zero

    # The numerator for these indices is small anyway:
    rhat = tmp1 / tmp2

    # Compute the relative shift between the images to within 1 pixel accuracy:
    r = fft.ifftn(rhat).real
    ii = np.argmax(r)

    # Find the center:
    n = np.size(vol1, 0)
    cX = np.fix(n / 2)
    cY = np.fix(n / 2)
    cZ = np.fix(n / 2)
    [sX, sY, sZ] = np.unravel_index(ii, np.shape(r))
    if sX > cX:
        sX = sX - n
    if sY > cY:
        sY = sY - n
    if sZ > cZ:
        sZ = sZ - n

    estdx = [-sX, -sY, -sZ]

    # No need to refine tranlations:
    return np.array(estdx)

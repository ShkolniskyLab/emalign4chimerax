import numpy as np
from scipy import fft
from numpy import linalg as LA


class fft_data_class:
    def __init__(self, in_data):
        n = in_data.shape[0]
        n2 = n // 2 + 1

        if in_data.dtype == np.float32:
            real_type = np.float32
            complex_type = np.complex64
        else:
            real_type = np.float64
            complex_type = np.complex128

        self.num_threads = 8
        self.in_array_f = np.empty((n, n, n), dtype=real_type)
        self.in_array_b = np.empty((n, n, n2), dtype=complex_type)
        self.fft_object_f = fft.rfftn(self.in_array_f)
        self.fft_object_b = fft.irfftn(self.in_array_b)

        self.n = n
        ll = np.fix(n / 2)
        freqrng = np.arange(-ll, n - ll)
        [omega_x, omega_y, omega_z] = np.meshgrid(freqrng, freqrng, freqrng, indexing='ij')
        self.omega_x = 2 * np.pi * omega_x / n
        self.omega_y = 2 * np.pi * omega_y / n
        self.omega_z = 2 * np.pi * omega_z / n


def reshift_vol(vol, s, fft_data=None):
    """
    Shift the volume given by im by the vector s using trigonometric interpolation.
    The volume im is of nxnxn, where n can be odi or even.

    Example: Shift the volume vol by 1 pixel in the x direction, 2 in the y direction, and 3 in the z direction:

          s = [1 2 3]
          vols = shift_vol(vol,s)

    NOTE: I don't know if s=[0 0 1 ] shifts up or down, but this can be easily checked. Same issue for the other directions.
    """

    if vol.ndim != 3:
        raise ValueError("Input must be a 3D volume")
    if (np.size(vol, 0) != np.size(vol, 1)) or (np.size(vol, 1) != np.size(vol, 2)):
        raise ValueError("All three dimension of the input must be equal")

    if fft_data is None:
        fft_data = fft_data_class(vol)

    n = np.size(vol, 0)
    if fft_data.n != n:  # cache is invalid, recreate.
        fft_data = fft_data_class(vol)

    phase_x = np.exp(1j * fft_data.omega_x * s[0])
    phase_y = np.exp(1j * fft_data.omega_y * s[1])
    phase_z = np.exp(1j * fft_data.omega_z * s[2])

    # Force conjugate symmetry:
    # (otherwise, this frequency component has no corresponding negative frequency to cancel out its imaginary part)
    if np.mod(n, 2) == 0:
        phase_x[0, :, :] = np.real(phase_x[0, :, :])
        phase_y[:, 0, :] = np.real(phase_y[:, 0, :])
        phase_z[:, :, 0] = np.real(phase_z[:, :, 0])
    phases = phase_x * phase_y * phase_z

    vol1 = fft.ifftshift(vol)
    pim1 = fft.rfftn(vol1)

    phases1 = fft.ifftshift(phases)
    n2 = n // 2 + 1
    phases1 = phases1[:, :, :n2]

    pim = pim1 * phases1
    svol = fft.irfftn(pim)

    svol = fft.fftshift(svol)

    if LA.norm(np.imag(svol[:])) / LA.norm(svol[:]) > 5.0e-7:
        raise ValueError("Large imaginary components")
    svol = np.real(svol)

    return svol


def reshift_vol_int(vol, s):
    """
    Shift a volume by the vector s (s must be a vector of integers, for non integer shifts use reshift_vol).
    """
    s = np.array(s)
    if not (np.round(s) == s).all():
        raise ValueError("s must be a vector of integers.")

    return np.roll(vol, (-s).astype(int), axis=[0, 1, 2])

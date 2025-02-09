import numpy as np
from scipy.fft import fft, ifft
import math
import cmath
from scipy.spatial.transform import Rotation


def fastrotate3d(vol, Rot):
    """
    FASTROTATE3D Rotate a 3D volume by a given rotation matrix.
    Input parameters:
     INPUT    Volume to rotate, can be odd or even.
     Rot        3x3 rotation matrix.
    Output parameters:
     OUTPUT   The rotated volume.
    Examples:
      Rot=rand_rots(1);
      rvol=fastrotate3d(vol,Rot);
    Yoel Shkolnisky, November 2013.
    """

    Rot_obj = Rotation.from_matrix(Rot)
    [psi, theta, phi] = Rot_obj.as_euler('xyz')

    psid = psi * 180 / np.pi
    thetad = theta * 180 / np.pi
    phid = phi * 180 / np.pi

    tmp = fastrotate3x(vol, psid)
    tmp = fastrotate3y(tmp, thetad)
    vol_out = fastrotate3z(tmp, phid)

    return vol_out


def adjustrotate(phi):
    """
    Decompose a rotation CCW by phi into a rotation of mult90 times 90
    degrees followed by a rotation by phi2, where phi2 is between -45 and 45.
    mult90 is an integer between 0 and 3 describing by how many multiples of
    90 degrees the image should be rotated so that an additional rotation by
    phi2 is equivalent to rotation by phi.
    """

    phi = np.mod(phi, 360)
    mult90 = 0
    phi2 = phi
    # Note that any two consecutive cases can be combined, but I decided to
    # leave them separated for clarity.
    if 45 <= phi < 90:
        mult90 = 1
        phi2 = -(90 - phi)
    elif 90 <= phi < 135:
        mult90 = 1
        phi2 = phi - 90
    elif 135 <= phi < 180:
        mult90 = 2
        phi2 = -(180 - phi)
    elif 180 <= phi < 225:
        mult90 = 2
        phi2 = phi - 180
    elif 215 <= phi < 270:
        mult90 = 3
        phi2 = -(270 - phi)
    elif 270 <= phi < 315:
        mult90 = 3
        phi2 = phi - 270
    elif 315 <= phi < 360:
        mult90 = 0
        phi2 = phi - 360

    return phi2, mult90


def fastrotateprecomp(SzX, SzY, phi):
    """
    Compute the interpolation tables required to rotate an image with SzX
    rows and SzY columns by an angle phi CCW.

    This function is used to accelerate fastrotate, in case many images are
    needed to be rotated by the same angle. In such a case it allows to
    precompute the interpolation tables only once instead of computing them
    for each image.

    M is a structure containing phi, Mx, My, where Mx and My are the
    interpolation tables used by fastrotate.
    """

    # Adjust the rotation angle to be between -45 and 45 degrees:
    [phi, mult90] = adjustrotate(phi)
    phi = np.pi * phi / 180
    phi = -phi  # to match Yaroslavsky's code which rotates CW
    if np.mod(SzY, 2) == 0:
        cy = SzY / 2 + 1
        sy = 1 / 2  # by how much should we shift the cy to get the center of the image
    else:
        cy = (SzY + 1) / 2
        sy = 0
    if np.mod(SzX, 2) == 0:
        cx = SzX / 2 + 1  # by how much should we shift the cy to get the center of the image
        sx = 1 / 2
    else:
        cx = (SzX + 1) / 2
        sx = 0

    # Precompte My and Mx:
    My = np.zeros((SzY, SzX)).astype(complex)
    r = np.arange(0, cy).astype(int)
    u = (1 - math.cos(phi)) / math.sin(phi + 2.2204e-16)
    alpha1 = 2 * np.pi * cmath.sqrt(-1) * r / SzY
    for x in range(SzX):
        Ux = u * (x + 1 - cx + sx)
        My[r, x] = np.exp(alpha1 * Ux)
        My[np.arange(SzY - 1, cy - 1, -1).astype(int), x] = np.conj(My[np.arange(1, cy - 2 * sy).astype(int), x])
    My = My.T  # remove when implementing using the loops below (NOTE: loop was removed, keeping comment just in case)
    Mx = np.zeros((SzX, SzY)).astype(complex)
    r = np.arange(0, cx).astype(int)
    u = -math.sin(phi)
    alpha2 = 2 * np.pi * cmath.sqrt(-1) * r / SzX
    for y in range(SzY):
        Uy = u * (y + 1 - cy + sy)
        Mx[r, y] = np.exp(alpha2 * Uy)
        Mx[np.arange(SzX - 1, cx - 1, -1).astype(int), y] = np.conj(Mx[np.arange(1, cx - 2 * sx).astype(int), y])

    class Struct:
        pass

    M = Struct()
    M.phi = phi
    M.Mx = Mx
    M.My = My
    M.mult90 = mult90
    return M


def fastrotate(vol, phi, M=None):
    """
    3-step image rotation by shearing.

    Input parameters:
     INPUT    Image to rotate, can be odd or even. If INPUT is a 3D array,
              each slice is rotated by phi.
     phi      Rotation angle in degrees CCW. Can be any angle (not limited
              like fastrotate_ref). Note that Yaroslavsky's code take phi CW.
     M        (Optional) Precomputed interpolation tables, as generated by
              fastrotateprecomp. If M is given than phi is ignored. This is
              useful if many images need to be rotated by the same angle,
              since then the computation of the same interpolation tables
              over and over again is avoided.
    Output parameters:
     OUTPUT   The rotated image.
    """

    SzX = np.size(vol, 0)
    SzY = np.size(vol, 1)
    SzZ = np.size(vol, 2)

    if M is None:
        M = fastrotateprecomp(SzX, SzY, phi)
    Mx = M.Mx
    My = M.My
    mult90 = M.mult90
    vol_out = np.zeros((SzX, SzY, SzZ))

    for k in range(SzZ):
        # Rotate by multiples of 90 degrees:
        if mult90 == 1:
            vol[:, :, k] = rot90(vol[:, :, k])
        elif mult90 == 2:
            vol[:, :, k] = rot180(vol[:, :, k])
        elif mult90 == 3:
            vol[:, :, k] = rot270(vol[:, :, k])
        elif mult90 != 0:
            TypeError('Invalid value for mult90')

        spinput = fft(vol[:, :, k], n=None, axis=1)
        spinput = spinput * My
        vol_out[:, :, k] = np.real(ifft(spinput, n=None, axis=1))

        spinput = fft(vol_out[:, :, k], n=None, axis=0)
        spinput = spinput * Mx
        vol_out[:, :, k] = np.real(ifft(spinput, n=None, axis=0))

        spinput = fft(vol_out[:, :, k], n=None, axis=1)
        spinput = spinput * My
        vol_out[:, :, k] = np.real(ifft(spinput, n=None, axis=1))

    return vol_out


def rot90(A):
    """
    Rotate the image A by 90 degrees CCW.
      B = rot90(A)
    """
    B = A.T
    B = np.flip(B, 0)
    return B


def rot180(A):
    """
    Rotate the image A by 180 degrees CCW.
      B = rot180(A)
    """
    B = np.flip(A, 0)
    B = np.flip(B, 1)
    return B


def rot270(A):
    """
    Rotate the image A by 270 degrees CCW.
      B = rot270(A)
    """
    B = A.T
    B = np.flip(B, 1)
    return B


def fastrotate3x(vol, phi):
    """
    FASTROTATE3X Rotate a 3D volume around the x-axis.
    Input parameters:
     INPUT    Volume to rotate, can be odd or even.
     phi      Rotation angle in degrees CCW.
     M        (Optional) Precomputed interpolation tables, as generated by
              fastrotateprecomp. If M is given than phi is ignored.

    Output parameters:
     OUTPUT   The rotated volume.

    Examples:

      rvol=fastrotate3x(vol,20);

      M=fastrotateprecomp(size(vol,2),size(vol,3),20);
      rvol=fastrotate(vol,[],M);
    """

    SzX = np.size(vol, 0)
    SzY = np.size(vol, 1)
    SzZ = np.size(vol, 2)

    # Precompte M:
    M = fastrotateprecomp(SzY, SzZ, phi)
    vol_out = np.zeros((SzX, SzY, SzZ), dtype=float)
    for k in range(SzX):
        im = (np.squeeze(vol[:, k, :]).reshape((SzX, SzZ, 1))).copy()
        rim = fastrotate(im, [], M)
        vol_out[:, k, :] = rim.reshape((SzX, SzZ))

    return vol_out


def fastrotate3y(vol, phi):
    """
    FASTROTATE3X Rotate a 3D volume around the x-axis.
    Input parameters:
     INPUT    Volume to rotate, can be odd or even.
     phi      Rotation angle in degrees CCW.
     M        (Optional) Precomputed interpolation tables, as generated by
              fastrotateprecomp. If M is given than phi is ignored.

    Output parameters:
     OUTPUT   The rotated volume.

    Examples:

      rvol=fastrotate3x(vol,20);

      M=fastrotateprecomp(size(vol,2),size(vol,3),20);
      rvol=fastrotate(vol,[],M);
    """

    SzX = np.size(vol, 0)
    SzY = np.size(vol, 1)
    SzZ = np.size(vol, 2)
    # Precompte M
    M = fastrotateprecomp(SzX, SzY, -phi)
    vol_out = np.zeros((SzX, SzY, SzZ), dtype=float)
    for k in range(SzY):
        im = (np.squeeze(vol[k, :, :]).reshape((SzY, SzZ, 1))).copy()
        rim = fastrotate(im, [], M)
        vol_out[k, :, :] = rim.reshape((SzY, SzZ))

    return vol_out


def fastrotate3z(vol, phi):
    """
    FASTROTATE3X Rotate a 3D volume around the x-axis.
    Input parameters:
     INPUT    Volume to rotate, can be odd or even.
     phi      Rotation angle in degrees CCW.
     M        (Optional) Precomputed interpolation tables, as generated by
              fastrotateprecomp. If M is given than phi is ignored.

    Output parameters:
     OUTPUT   The rotated volume.

    Examples:

      rvol=fastrotate3x(vol,20);

      M=fastrotateprecomp(size(vol,2),size(vol,3),20);
      rvol=fastrotate(vol,[],M);
    """

    SzX = np.size(vol, 0)
    SzY = np.size(vol, 1)
    SzZ = np.size(vol, 2)
    # Precompte M
    M = fastrotateprecomp(SzX, SzY, -phi)
    vol_out = np.zeros((SzX, SzY, SzZ), dtype=float)
    for k in range(SzZ):
        im = (np.squeeze(vol[:, :, k]).reshape((SzX, SzY, 1))).copy()
        rim = fastrotate(im, [], M)
        vol_out[:, :, k] = rim.reshape((SzX, SzY))

    return vol_out

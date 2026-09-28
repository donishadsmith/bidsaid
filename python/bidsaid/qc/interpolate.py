from pathlib import Path
from typing import Literal, TypeAlias

import nibabel as nib, numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from numpy.typing import NDArray

from bidsaid.io import load_nifti
from bidsaid.metadata import get_tr

ScipyInterpolatorClass: TypeAlias = type[CubicSpline] | type[PchipInterpolator]

_INTERPOLATORS: dict[Literal["cubic", "pchip"], ScipyInterpolatorClass] = {
    "cubic": CubicSpline,
    "pchip": PchipInterpolator,
}


def upsample_array(
    array: NDArray[np.floating],
    n: int,
    axis: int = -1,
    method: Literal["cubic", "pchip"] = "cubic",
    chunk_size: int = 20000,
) -> NDArray[np.floating]:
    """
    Temporally Upsample an Array.

    Inserts ``n`` interpolated frames in each gap between consecutive timepoints along
    ``axis``. For ``T`` original timepoints, there are ``T - 1`` gaps, so the new length is
    ``(T - 1) * n + T`` or ``(T - 1) * (n + 1) + 1``. The new TR will be
    ``TR / (n + 1)``. Original timepoints are retained and can be recovered with
    :func:`downsample_array` using the same ``n``.

    Parameters
    ----------
    array : :obj:`NDArray[np.floating]`
        The data to upsample (e.g. a 4D volume array or a 2D vertices x time surface array).

    n : :obj:`int`
        The number of interpolated frames inserted in each gap between two real timepoints.

    axis : :obj:`int`, default=-1
        The dimension containing the timepoints to be interpolated.

    method : :obj:`Literal["cubic", "pchip"]`, default="cubic"
        The interpolation method to use ("cubic" for cublic spline and "pcip" for shape-preserving cubic).

    chunk_size : :obj:`int`, default=20000
        The number of voxels or vertices interpolated at once. Lower values reduce memory use.

    Returns
    -------
    NDArray[np.floating]
        The upsampled array, with the same dimensions as ``array`` but as a length of
        ``(T - 1) * n + T`` in ``axis``.
    """
    assert method in ["cubic", "pchip"], "method must be 'cubic' or 'pchip'"

    array = np.moveaxis(array, axis, -1)
    interpolator = _INTERPOLATORS[method]

    n_timepoints = array.shape[-1]
    old_timepoints = np.arange(n_timepoints)
    new_timepoints = np.arange((n_timepoints - 1) * n + n_timepoints) / (n + 1)

    flat_array = array.reshape(-1, n_timepoints)
    out_array = np.empty((flat_array.shape[0], new_timepoints.size), dtype=array.dtype)
    for index in range(0, flat_array.shape[0], chunk_size):
        chunk = flat_array[index : index + chunk_size]
        out_array[index : index + chunk_size] = interpolator(
            old_timepoints, chunk, axis=-1
        )(new_timepoints)

    out_array[:, :: n + 1] = flat_array
    out_array = out_array.reshape(array.shape[:-1] + (new_timepoints.size,))

    return np.moveaxis(out_array, -1, axis)


def downsample_array(
    array: NDArray[np.floating], n: int, axis: int = -1
) -> NDArray[np.floating]:
    """
    Temporally Downsample an Array.

    Retains every ``(n + 1)``th frame along ``axis``, reversing :func:`upsample_array` when
    the same ``n`` is used. Note that ``n + 1`` is equivalent to
    ``original_tr / upsampled_tr``, and the new TR is ``TR * (n + 1)``.

    Parameters
    ----------
    array : :obj:`NDArray[np.floating]`
        The upsampled data (e.g. a 4D volume array or a 2D vertices x time surface array).

    n : :obj:`int`
        The number of interpolated frames that were inserted in each gap between two real
        timepoints.

    axis : :obj:`int`, default=-1
        The dimension containing the timepoints to be downsampled.

    Returns
    -------
    NDArray[np.floating]
        The downsampled array with the same dimensions as ``array``.
    """
    slicer = [slice(None)] * array.ndim
    slicer[axis] = slice(None, None, n + 1)

    return array[tuple(slicer)]


def _create_hdr(
    nifti_img,
    n: int,
    img_fdata: NDArray[np.floating],
    direction: Literal["upsample", "downsample"],
):
    assert direction in [
        "upsample",
        "downsample",
    ], "direction must be either 'upsample' or 'downsample'"

    new_tr = (
        get_tr(nifti_img) / (n + 1)
        if direction == "upsample"
        else get_tr(nifti_img) * (n + 1)
    )
    hdr = nifti_img.header.copy()
    hdr.set_data_dtype(img_fdata.dtype)
    hdr.set_zooms(hdr.get_zooms()[:3] + (new_tr,))

    return hdr


def upsample_img(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    n: int,
    method: Literal["cubic", "pchip"] = "cubic",
    chunk_size: int = 20000,
) -> nib.nifti1.Nifti1Image:
    """
    Temporally Upsample NIfTI Image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the functional NIfTI file or a functional NIfTI image.

    n : :obj:`int`
        The number of interpolated frames inserted in each gap between two real volumes.

    method : :obj:`Literal["cubic", "pchip"]`, default="cubic"
        The interpolation method to use ("cubic" for cublic spline and "pcip" for shape-preserving cubic).

    chunk_size : :obj:`int`, default=20000
        The number of voxels interpolated at once.

    Returns
    -------
    Nifti1Image
        The upsampled NIfTI image with the upsampled TR in the header.
    """
    nifti_img = load_nifti(nifti_file_or_img)
    img_data = np.asanyarray(nifti_img.dataobj, dtype=np.float32)
    upsampled_img_fdata = upsample_array(img_data, n, -1, method, chunk_size)

    return nib.Nifti1Image(
        upsampled_img_fdata,
        nifti_img.affine,
        _create_hdr(nifti_img, n, upsampled_img_fdata, "upsample"),
    )


def downsample_img(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image, n: int
) -> nib.nifti1.Nifti1Image:
    """
    Temporally Downsample NIfTI Image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the functional NIfTI file or a functional NIfTI image.

    n : :obj:`int`
        The number of interpolated frames inserted in each gap between two real volumes.

    Returns
    -------
    Nifti1Image
        The downsampled NIfTI image with the downsampled TR in the header.
    """
    nifti_img = load_nifti(nifti_file_or_img)
    downsampled_img_fdata = np.asanyarray(nifti_img.dataobj[..., :: n + 1])

    return nib.Nifti1Image(
        downsampled_img_fdata,
        nifti_img.affine,
        _create_hdr(nifti_img, n, downsampled_img_fdata, "downsample"),
    )


__all__ = ["downsample_array", "downsample_img", "upsample_array", "upsample_img"]

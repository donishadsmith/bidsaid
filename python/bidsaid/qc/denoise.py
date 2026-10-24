from pathlib import Path
from typing import Literal

import nibabel as nib, numpy as np, pandas as pd
from numpy.typing import NDArray

from bidsaid._helpers import is_path
from bidsaid.io import load_nifti, NiftiFileorImage

from .interpolate import interpolate_censored_array


def regress_confounds_array(
    array: NDArray[np.floating],
    confounds: pd.DataFrame | NDArray,
    censor_mask: str | Path | NDArray | None = None,
    interpolation_method: Literal["cubic", "pchip"] | None = None,
    axis: int = -1,
    chunk_size: int = 20000,
) -> NDArray[np.floating]:
    """
    Regress Confounds from an Array.

    Fits an ordinary least squares model with an intercept and ``confounds`` using only the retained timepoints (1 in
    ``censor_mask``), then removes the fitted confound signal from all timepoints along ``axis``. The output has the
    same length as ``array``.

    .. important::
        The intercept is estimated but not removed, so the output retains the mean of the retained timepoints. No
        detrending, demeaning, or standardization is applied to ``array``.

    .. important::
        If ``interpolation_method`` is not None, interpolation is performed after denoising, so censored timepoints are
        excluded from the fit and only censored timepoints are replaced. Censored timepoints before the first or after
        the last retained timepoint (e.g. dummy scans) are not interpolated since extrapolation is unstable, and instead
        use the value of the nearest retained timepoint. If None, censored timepoints contain the data minus the
        predicted confound signal.

    Parameters
    ----------
    array : :obj:`NDArray[np.floating]`
        The data to denoise (e.g. a 4D volume array or a 2D vertices x time surface array).

    confounds : :obj:`pandas.DataFrame` or :obj:`NDArray`
        The confound regressors with timepoints as rows. Must have the same number of rows as timepoints in ``axis`` and
        cannot contain NaN values. A constant column should not be included since an intercept is added.

    censor_mask : :obj:`str`, :obj:`Path`, :obj:`NDArray`, or :obj:`None`, default=None
        A 1D binary mask or a path to a text file containing the mask, where 1 indicates timepoints used to fit the
        model and 0 indicates timepoints excluded from the fit. If None, all timepoints are used.

    interpolation_method : :obj:`Literal["cubic", "pchip"]` or :obj:`None`, default=None
        The interpolation method used to replace censored timepoints after denoising ("cubic" for cubic spline and
        "pchip" for shape-preserving cubic). If None, no interpolation is performed.

    axis : :obj:`int`, default=-1
        The dimension containing the timepoints.

    chunk_size : :obj:`int`, default=20000
        The number of voxels or vertices denoised and interpolated at once.

    Returns
    -------
    NDArray[np.floating]
        The denoised array with the same dimensions as ``array``.
    """
    array = np.moveaxis(array, axis, -1)
    n_timepoints = array.shape[-1]

    confounds = (
        confounds.to_numpy()
        if isinstance(confounds, pd.DataFrame)
        else np.asarray(confounds)
    ).astype(np.float64)
    confounds = (
        confounds.reshape(n_timepoints, -1) if confounds.ndim == 1 else confounds
    )
    assert (
        confounds.shape[0] == n_timepoints
    ), f"confounds has {confounds.shape[0]} rows but the array has {n_timepoints} timepoints"
    assert np.isfinite(
        confounds
    ).all(), "confounds cannot contain NaN or infinite values"

    if censor_mask is None:
        censor_mask = np.ones(n_timepoints, dtype=bool)
    else:
        censor_mask = (
            np.loadtxt(censor_mask) if is_path(censor_mask) else np.asarray(censor_mask)
        ).astype(bool)

    assert (
        censor_mask.size == n_timepoints
    ), f"censor_mask has {censor_mask.size} values but the array has {n_timepoints} timepoints"
    assert censor_mask.any(), "censor_mask cannot censor all timepoints"
    assert interpolation_method in [
        "cubic",
        "pchip",
        None,
    ], "interpolation_method must be 'cubic', 'pchip', or None"

    # Centering on the retained timepoints keeps their mean in the output
    design_matrix = np.column_stack(
        [
            np.ones(n_timepoints),
            confounds - confounds[censor_mask].mean(axis=0),
        ]
    )
    out_dtype = np.result_type(array.dtype, np.float32)
    projector = (
        design_matrix[:, 1:] @ np.linalg.pinv(design_matrix[censor_mask])[1:]
    ).astype(out_dtype)

    flat_array = array.reshape(-1, n_timepoints)
    flat_out_array = np.empty(flat_array.shape, dtype=out_dtype)
    for index in range(0, flat_array.shape[0], chunk_size):
        chunk = flat_array[index : index + chunk_size].astype(out_dtype)
        flat_out_array[index : index + chunk_size] = (
            chunk - chunk[:, censor_mask] @ projector.T
        )

    if interpolation_method and not censor_mask.all():
        flat_out_array = interpolate_censored_array(
            flat_out_array, censor_mask, -1, interpolation_method, chunk_size
        )

    return np.moveaxis(flat_out_array.reshape(array.shape), -1, axis)


def regress_confounds(
    nifti_file_or_img: NiftiFileorImage,
    confounds: pd.DataFrame | NDArray,
    censor_mask: str | Path | NDArray | None = None,
    mask_img_or_file: NiftiFileorImage | None = None,
    interpolation_method: Literal["cubic", "pchip"] | None = None,
    chunk_size: int = 20000,
) -> nib.nifti1.Nifti1Image:
    """
    Regress Confounds from a NIfTI Image.

    Fits an ordinary least squares model with an intercept and ``confounds`` using only the retained volumes (1 in
    ``censor_mask``), then removes the fitted confound signal from all volumes. The output has the same number of
    volumes as the input image.

    .. important::
        The intercept is estimated but not removed, so the output retains the mean of the retained volumes. No
        detrending, demeaning, or standardization is applied to the image.

    .. important::
        If ``interpolation_method`` is not None, interpolation is performed after denoising, so censored volumes are
        excluded from the fit and only censored volumes are replaced. Censored volumes before the first or after the
        last retained volume (e.g. dummy scans) are not interpolated since extrapolation is unstable, and instead use
        the value of the nearest retained volume. If None, censored volumes contain the data minus the predicted confound signal.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the functional NIfTI file or a functional NIfTI image.

    confounds : :obj:`pandas.DataFrame` or :obj:`NDArray`
        The confound regressors with volumes as rows. Must have the same number of rows as volumes in the image and
        cannot contain NaN values. A constant column should not be included since an intercept is added.

    censor_mask : :obj:`str`, :obj:`Path`, :obj:`NDArray`, or :obj:`None`, default=None
        A 1D binary mask or a path to a text file containing the mask, where 1 indicates volumes used to fit the model
        and 0 indicates volumes excluded from the fit. If None, all volumes are used.

    mask_img_or_file : :obj:`str`, :obj:`Path`, :obj:`Nifti1Image`, or :obj:`None`, default=None
        Path to the mask NIfTI file or a mask NIfTI image. Voxels outside of the mask are set to zero. If None, all voxels are denoised.

    interpolation_method : :obj:`Literal["cubic", "pchip"]` or :obj:`None`, default=None
        The interpolation method used to replace censored volumes after denoising ("cubic" for cubic spline and "pchip"
        for shape-preserving cubic). If None, no interpolation is performed.

    chunk_size : :obj:`int`, default=20000
        The number of voxels denoised and interpolated at once.

    Returns
    -------
    Nifti1Image
        The denoised NIfTI image.
    """
    nifti_img = load_nifti(nifti_file_or_img)
    nifti_fdata = np.asanyarray(nifti_img.dataobj, dtype=np.float32)

    if mask_img_or_file is None:
        mask_data = np.ones(nifti_fdata.shape[:3], dtype=bool)
    else:
        mask_data = np.asanyarray(load_nifti(mask_img_or_file).dataobj).astype(bool)

    denoised_img_fdata = np.zeros(nifti_fdata.shape, dtype=np.float32)
    denoised_img_fdata[mask_data] = regress_confounds_array(
        nifti_fdata[mask_data],
        confounds,
        censor_mask,
        interpolation_method,
        -1,
        chunk_size,
    )

    hdr = nifti_img.header.copy()
    hdr.set_data_dtype(denoised_img_fdata.dtype)

    return nib.Nifti1Image(denoised_img_fdata, nifti_img.affine, hdr)


__all__ = ["regress_confounds_array", "regress_confounds"]

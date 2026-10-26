import nibabel as nib, numpy as np
from numpy.typing import NDArray

from bidsaid.io import load_nifti, NiftiFileorImage


def zscore_array(
    array: NDArray[np.floating],
    axis: int = -1,
) -> NDArray[np.floating]:
    """
    Z-Score an Array.

    .. important::
        Voxels or vertices that are constant in either array are set to zero.

    Parameters
    ----------
    array : :obj:`NDArray[np.floating]`
        The data to z-score (e.g. a 4D volume array or a 2D vertices x time surface array).

    axis : :obj:`int`, default=-1
        The dimension containing the timepoints.

    Returns
    -------
    NDArray[np.floating]
        The z-scored array with the same dimensions as ``array``.
    """
    array = np.moveaxis(array, axis, -1)
    out_dtype = np.result_type(array.dtype, np.float32)
    flat_array = array.reshape(-1, array.shape[-1]).astype(out_dtype)

    mean = flat_array.mean(axis=-1)
    std = flat_array.std(axis=-1, ddof=1)
    valid = flat_array.std(axis=-1) > 0
    assert valid.any(), "no voxels or vertices vary over time"

    flat_out_array = np.zeros(flat_array.shape, dtype=out_dtype)
    flat_out_array[valid] = (flat_array[valid] - mean[valid, np.newaxis]) / std[
        valid, np.newaxis
    ]

    return np.moveaxis(flat_out_array.reshape(array.shape), -1, axis)


def zscore_img(
    nifti_file_or_img: NiftiFileorImage,
) -> nib.nifti1.Nifti1Image:
    """
    Z-Score a NIfTI Image.

    .. important::
        Voxels that are constant in either array are set to zero.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the functional NIfTI file or a functional NIfTI image to z-score.

    Returns
    -------
    Nifti1Image
        The z-scored NIfTI image.
    """
    nifti_img = load_nifti(nifti_file_or_img)
    zscored_img_fdata = zscore_array(nifti_img.get_fdata())

    hdr = nifti_img.header.copy()
    hdr.set_data_dtype(zscored_img_fdata.dtype)

    return nib.Nifti1Image(zscored_img_fdata, nifti_img.affine, hdr)


def zscore_to_reference_array(
    array: NDArray[np.floating],
    reference_array: NDArray[np.floating],
    match_run_scale: bool = True,
    axis: int = -1,
) -> NDArray[np.floating]:
    """
    Z-Score an Array Relative to a Reference Array.

    For each voxel or vertex, subtracts the mean of ``reference_array`` from every timepoint of ``array`` and divides
    by the standard deviation of ``reference_array``. The output has the same length as ``array``, so each timepoint
    shows how far it is from the reference level in units of the reference variability (e.g. music relative to rest).

    .. important::
        Voxels or vertices that are constant in either array are set to zero.

    Parameters
    ----------
    array : :obj:`NDArray[np.floating]`
        The data to z-score (e.g. a 4D volume array or a 2D vertices x time surface array).

    reference_array : :obj:`NDArray[np.floating]`
        The reference data with the same dimensions as ``array`` except for ``axis``, which can have a different number of timepoints.

    match_run_scale : :obj:`bool`, default=True
        Whether to remove run-level intensity differences between ``array`` and ``reference_array`` before z-scoring.

    axis : :obj:`int`, default=-1
        The dimension containing the timepoints.

    Returns
    -------
    NDArray[np.floating]
        The z-scored array with the same dimensions as ``array``.
    """
    array = np.moveaxis(array, axis, -1)
    reference_array = np.moveaxis(reference_array, axis, -1)
    assert (
        array.shape[:-1] == reference_array.shape[:-1]
    ), f"array has shape {array.shape[:-1]} but reference_array has shape {reference_array.shape[:-1]} excluding axis"
    assert (
        reference_array.shape[-1] > 1
    ), "reference_array must have more than one timepoint"

    out_dtype = np.result_type(array.dtype, np.float32)
    flat_array = array.reshape(-1, array.shape[-1]).astype(out_dtype)
    flat_reference_array = reference_array.reshape(-1, reference_array.shape[-1])

    reference_mean = flat_reference_array.mean(axis=-1)
    reference_std = flat_reference_array.std(axis=-1, ddof=1)
    valid = (flat_array.std(axis=-1) > 0) & (reference_std > 0)
    assert valid.any(), "no voxels or vertices vary over time in both arrays"

    if match_run_scale:
        flat_array /= np.median(flat_array[valid].mean(axis=-1) / reference_mean[valid])

    flat_out_array = np.zeros(flat_array.shape, dtype=out_dtype)
    flat_out_array[valid] = (
        flat_array[valid] - reference_mean[valid, np.newaxis]
    ) / reference_std[valid, np.newaxis]

    return np.moveaxis(flat_out_array.reshape(array.shape), -1, axis)


def zscore_to_reference(
    nifti_file_or_img: NiftiFileorImage,
    reference_file_or_img: NiftiFileorImage,
    mask_file_or_img: NiftiFileorImage | None = None,
    match_run_scale: bool = True,
) -> nib.nifti1.Nifti1Image:
    """
    Z-Score a NIfTI Image Relative to a Reference NIfTI Image.

    For each voxel, subtracts the mean of the reference image from every volume of the image and divides by the
    standard deviation of the reference image. The output has the same number of volumes as the image, so each volume
    shows how far it is from the reference level in units of the reference variability (e.g. music relative to rest).

    .. important::
        Voxels that are constant in either image or outside of the mask are set to zero.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the functional NIfTI file or a functional NIfTI image to z-score.

    reference_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the reference functional NIfTI file or a reference functional NIfTI image. Must have the same spatial
        dimensions and affine as the image but can have a different number of volumes.

    mask_file_or_img : :obj:`str`, :obj:`Path`, :obj:`Nifti1Image`, or :obj:`None`, default=None
        Path to the mask NIfTI file or a mask NIfTI image. Only voxels in the mask are used to match the run scale and z-scored.
        If None, all voxels are used.

        .. important::
            Recommended to use `Nilearn's ``intersect_mask`` function <https://nilearn.github.io/dev/modules/generated/nilearn.masking.intersect_masks.html>_`
            to compute a "group" mask for the target and reference image to ensure

    match_run_scale : :obj:`bool`, default=True
        Whether to remove run-level intensity differences between the image and the reference image before z-scoring.

    Returns
    -------
    Nifti1Image
        The z-scored NIfTI image.
    """
    nifti_img = load_nifti(nifti_file_or_img)
    reference_img = load_nifti(reference_file_or_img)
    assert nifti_img.shape[:3] == reference_img.shape[:3] and np.allclose(
        nifti_img.affine, reference_img.affine, atol=1e-5
    ), "the image and the reference image must have the same spatial dimensions and affine"

    nifti_fdata = np.asanyarray(nifti_img.dataobj, dtype=np.float32)
    reference_fdata = np.asanyarray(reference_img.dataobj, dtype=np.float32)

    if mask_file_or_img is None:
        mask_data = np.ones(nifti_fdata.shape[:3], dtype=bool)
    else:
        mask_data = np.asanyarray(load_nifti(mask_file_or_img).dataobj).astype(bool)

    zscored_img_fdata = np.zeros(nifti_fdata.shape, dtype=np.float32)
    zscored_img_fdata[mask_data] = zscore_to_reference_array(
        nifti_fdata[mask_data], reference_fdata[mask_data], match_run_scale, -1
    )

    hdr = nifti_img.header.copy()
    hdr.set_data_dtype(zscored_img_fdata.dtype)

    return nib.Nifti1Image(zscored_img_fdata, nifti_img.affine, hdr)


__all__ = ["zscore_to_reference_array", "zscore_to_reference"]

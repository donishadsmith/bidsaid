"""Module for creating masks"""

import nibabel as nib, numpy as np

from nilearn.datasets import load_mni152_gm_template
from numpy.typing import NDArray

from .io import NiftiFileorImage, load_nifti


def create_sphere_mask(
    seed_coordinate: tuple[float, float, float] | NDArray[np.floating],
    sphere_radius: float | int = 5,
    nifti_file_or_img: NiftiFileorImage = None,
) -> nib.Nifti1Image:
    """
    Create a Sphere Mask.

    For a given seed coordinate, a binary sphere mask (with a radius equal to ``sphere_radius``) is created in the same space
    as ``nifti_file_or_img`` or the 2mm MNI152 template if ``nifti_file_or_img`` is None.

    Parameters
    ----------
    seed_coordinate:
        The center coordinate of a seed.

    sphere_radius
        The radius of the seed coordinate.

    nifti_file_or_img : :obj:`str`, :obj:`Path`, :obj:`Nifti1Image`, or :obj:`None`, default=None
        Path to a 3D NIfTI file or a 3D NIfTI image. If None, a 2mm MNI152 template is used.

    Return
    ------
    Nifti1Image
        A binary mask where "1" are the voxels inside the seed sphere and "0" is otherwise.
    """
    nifti_img = (
        load_nifti(nifti_file_or_img)
        if nifti_file_or_img
        else load_mni152_gm_template(resolution=2)
    )
    assert len(nifti_img.shape) == 3, "`nifti_file_or_img` must be a 3D NIfTI image."

    seed_coordinate = np.array(seed_coordinate)
    assert seed_coordinate.size == 3, "`seed_coordinate` must be a 3D coordinate."

    voxel_coordinates = nib.affines.apply_affine(
        nifti_img.affine, np.indices(nifti_img.shape).reshape(3, -1).T
    )
    voxel_distances = np.linalg.norm(voxel_coordinates - seed_coordinate, axis=1)

    sphere_mask = (
        (voxel_distances <= sphere_radius).reshape(nifti_img.shape).astype(np.uint8)
    )

    return nib.Nifti1Image(sphere_mask, nifti_img.affine, nifti_img.header)


__all__ = ["create_sphere_mask"]

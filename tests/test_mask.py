import nibabel as nib, numpy as np

from nilearn.datasets import load_mni152_gm_mask
from bidsaid.mask import create_sphere_mask


def test_create_sphere_mask():
    """Test for ``create_sphere_mask``"""
    nifti_img = load_mni152_gm_mask(1)
    sphere_mask = create_sphere_mask((0, 0, 0), nifti_file_or_img=nifti_img)
    voxel_coordinates = nib.affines.apply_affine(
        sphere_mask.affine, np.argwhere(sphere_mask.get_fdata() == 1)
    )

    voxel_distances = np.linalg.norm(voxel_coordinates - np.array((0, 0, 0)), axis=1)

    assert np.all(voxel_distances <= 5)

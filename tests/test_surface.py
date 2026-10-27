from pathlib import Path

import nibabel as nib, numpy as np, pytest

from bidsaid.surface import (
    get_cortex_masks,
    load_surface_data,
    load_surface_mesh,
    project_to_surface,
)


def _save_surface_gifti(tmp_dir, coordinates, faces):
    surface_file = Path(tmp_dir.name) / "hemi-L_pial.surf.gii"
    gifti = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                coordinates.astype(np.float32), intent="NIFTI_INTENT_POINTSET"
            ),
            nib.gifti.GiftiDataArray(
                faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"
            ),
        ]
    )

    nib.save(gifti, surface_file)

    return surface_file


def _save_data_gifti(tmp_dir, data, filename):
    data_file = Path(tmp_dir.name) / filename
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data)]), data_file)

    return data_file


def test_load_surface_mesh(tmp_dir, surface_mesh):
    "Test ``load_surface_mesh`` with a tuple and a GIFTI file."
    coordinates, faces = surface_mesh
    assert load_surface_mesh(surface_mesh) == surface_mesh

    surface_file = _save_surface_gifti(tmp_dir, coordinates, faces)
    loaded_coordinates, loaded_faces = load_surface_mesh(surface_file)
    assert np.array_equal(loaded_coordinates, coordinates)
    assert np.array_equal(loaded_faces, faces)


def test_load_surface_data(tmp_dir):
    "Test ``load_surface_data`` with an array and a GIFTI file."
    data = np.arange(16, dtype=np.float32)
    assert np.array_equal(load_surface_data(data), data)

    data_file = _save_data_gifti(tmp_dir, data, "hemi-L_sulc.shape.gii")
    assert np.array_equal(load_surface_data(data_file), data)


def test_get_cortex_masks(tmp_dir):
    "Test ``get_cortex_masks`` returns boolean masks for both hemispheres."
    left_label = np.array([1, 1, 0, 1], dtype=np.int32)
    right_label = np.array([0, 1, 1, 1], dtype=np.int32)
    medial_wall_files = (
        _save_data_gifti(tmp_dir, left_label, "hemi-L_desc-nomedialwall.label.gii"),
        _save_data_gifti(tmp_dir, right_label, "hemi-R_desc-nomedialwall.label.gii"),
    )

    cortex_masks = get_cortex_masks(medial_wall_files)

    assert len(cortex_masks) == 2
    assert all(cortex_mask.dtype == bool for cortex_mask in cortex_masks)
    assert np.array_equal(cortex_masks[0], left_label.astype(bool))
    assert np.array_equal(cortex_masks[1], right_label.astype(bool))


@pytest.mark.parametrize("use_white_surfaces", [True, False])
def test_project_to_surface(nifti_img_and_path, surface_mesh, use_white_surfaces):
    "Test ``project_to_surface`` returns one vertices x volumes array per hemisphere."
    nifti_img, nifti_img_path = nifti_img_and_path
    coordinates, faces = surface_mesh
    inner_coordinates = coordinates.copy()
    inner_coordinates[:, 2] -= 1
    pial_surfaces = (surface_mesh, surface_mesh)
    white_surfaces = ((inner_coordinates, faces), (inner_coordinates, faces))

    hemispheres = project_to_surface(
        nifti_img_path, pial_surfaces, white_surfaces if use_white_surfaces else None
    )

    assert len(hemispheres) == 2
    for hemisphere in hemispheres:
        assert hemisphere.shape == (coordinates.shape[0], nifti_img.shape[-1])
        assert hemisphere.dtype == np.float32
        assert not np.isnan(hemisphere).any()


def test_project_to_surface(nifti_img_and_path, surface_mesh):
    "Test ``project_to_surface`` with a 3D image returns 1D arrays."
    nifti_img, _ = nifti_img_and_path
    nifti_img_3d = nifti_img.slicer[..., 0]

    hemispheres = project_to_surface(nifti_img_3d, (surface_mesh, surface_mesh))

    assert hemispheres[0].shape == (surface_mesh[0].shape[0],)

import tempfile
from pathlib import Path

import nibabel as nib, numpy as np, pytest

from bidsaid.simulate import simulate_nifti_image


@pytest.fixture(autouse=False, scope="function")
def tmp_dir():
    """Create temporary directory for each test module."""
    temp_dir = tempfile.TemporaryDirectory()

    yield temp_dir

    temp_dir.cleanup()


@pytest.fixture(autouse=False, scope="function")
def nifti_img_and_path(tmp_dir):
    img = simulate_nifti_image((20, 20, 10, 5))
    img_path = Path(tmp_dir.name) / "img.nii"
    nib.save(img, img_path)

    yield img, img_path


@pytest.fixture(autouse=False, scope="function")
def surface_mesh():
    """
    Create a small triangulated grid mesh (16 vertices, 18 faces) as ``(coordinates, faces)``.
    """
    grid_x, grid_y = np.meshgrid(np.arange(2, 18, 4.0), np.arange(2, 18, 4.0))
    coordinates = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 5.0)]
    )

    n_columns = grid_x.shape[1]
    faces = []
    for row_index in range(grid_x.shape[0] - 1):
        for column_index in range(n_columns - 1):
            top_left = row_index * n_columns + column_index
            top_right, bottom_left = top_left + 1, top_left + n_columns
            bottom_right = bottom_left + 1
            faces.append([top_left, top_right, bottom_left])
            faces.append([top_right, bottom_right, bottom_left])

    yield coordinates, np.asarray(faces)

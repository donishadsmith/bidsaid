from pathlib import Path

import nibabel as nib, pytest

pytest.importorskip("pyvista")
imageio = pytest.importorskip("imageio.v2")
pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("neuromaps")

from bidsaid.movie import (
    MovieWriter,
    compute_vmax,
    create_surface_run,
    virtual_display,
)
from bidsaid.simulate import simulate_nifti_image
from bidsaid.surface import get_cortex_masks
from bidsaid.visual import SurfaceRenderer
from .test_visual import surfaces


@pytest.fixture(scope="module")
def renderer(surfaces):
    with (
        virtual_display(),
        SurfaceRenderer(
            *surfaces["inflated"], sulcal_depths=surfaces["sulc"], size=(80, 60)
        ) as surface_renderer,
    ):
        yield surface_renderer


def test_movie_writer(tmp_dir, surfaces, renderer):
    "Test projecting simulated images and writing the movies."
    dst_dir = Path(tmp_dir.name)
    cortex_masks = get_cortex_masks(surfaces["medial"])

    surface_runs = []
    for subject_id, task_name, n_volumes in (
        ("01", "rest", 4),
        ("01", "music", 3),
        ("02", "rest", 4),
    ):
        nifti_file = dst_dir / f"sub-{subject_id}_task-{task_name}_bold.nii.gz"
        nib.save(simulate_nifti_image((20, 20, 10, n_volumes)), nifti_file)
        surface_runs.append(
            create_surface_run(
                nifti_file,
                surfaces["pial"],
                surfaces["white"],
                cortex_masks,
                n_interpolated_frames=1,
            )
        )

    assert surface_runs[0].n_frames == 7
    assert surface_runs[0].display_name == "sub-01"

    vmax = compute_vmax(surface_runs)
    renderer.color_range = (-vmax, vmax)
    writer = MovieWriter(
        surface_runs, renderer, task_order=["rest", "music"], target_fps=24
    )
    movie_files = writer.write(dst_dir, per_subject=True, per_task=True, shared=True)

    assert len(movie_files) == 5
    assert all(movie_file.exists() for movie_file in movie_files)
    with imageio.get_reader(str(dst_dir / "group_desc-surface_movie.mp4")) as reader:
        assert reader.count_frames() == 5 * 12
        assert reader.get_data(0).shape[0] == 2 * 60

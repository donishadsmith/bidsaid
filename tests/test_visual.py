import numpy as np, pytest

pytest.importorskip("pyvista")
pytest.importorskip("neuromaps")

from bidsaid.movie import virtual_display
from bidsaid.simulate import simulate_nifti_img
from bidsaid.surface import (
    fetch_fsaverage_surfaces,
    get_cortex_masks,
    load_surface_mesh,
    project_to_surface,
)
from bidsaid.visual import SurfaceRenderer, create_polydata


@pytest.fixture(scope="session")
def surfaces():
    return fetch_fsaverage_surfaces("3k")


@pytest.fixture(scope="module")
def coordinates(surfaces):
    """The ``(left, right)`` vertex coordinates of the pial surfaces."""
    return (
        load_surface_mesh(surfaces["pial"][0])[0],
        load_surface_mesh(surfaces["pial"][1])[0],
    )


@pytest.fixture(scope="module")
def surface_data(surfaces):
    """A simulated image projected onto the surfaces as ``(lh, rh)`` arrays."""
    return project_to_surface(
        simulate_nifti_img((20, 20, 10, 3)),
        surfaces["pial"],
        surfaces["white"],
        get_cortex_masks(surfaces["medial"]),
    )


@pytest.fixture(scope="module")
def renderer(surfaces):
    with (
        virtual_display(),
        SurfaceRenderer(
            *surfaces["inflated"],
            sulcal_depths=surfaces["sulc"],
            size=(120, 90),
            color_range=(-1, 1),
            threshold=0.5,
        ) as surface_renderer,
    ):
        yield surface_renderer


def frames_match(a, b, max_diff=4, max_frac=1e-3):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))

    return diff.max() <= max_diff and np.mean(diff.any(axis=-1)) <= max_frac


def test_create_polydata(surface_mesh):
    "Test ``create_polydata`` keeps the vertices and faces."
    coordinates, faces = surface_mesh
    mesh = create_polydata(surface_mesh)

    assert mesh.n_points == coordinates.shape[0]
    assert mesh.n_cells == faces.shape[0]
    assert np.array_equal(mesh.faces.reshape(-1, 4)[:, 1:], faces)


def test_surface_renderer_updates_data(renderer, coordinates, surface_data):
    "Test consecutive frames reflect the data passed to ``render``."
    left_coordinates, right_coordinates = coordinates
    n_left, n_right = left_coordinates.shape[0], right_coordinates.shape[0]
    lh, rh = surface_data

    blank_frame = renderer.render()
    full_frame = renderer.render(np.full(n_left, 2.0), np.full(n_right, -2.0))
    half_frame = renderer.render(
        np.where(left_coordinates[:, 0] > np.median(left_coordinates[:, 0]), 2.0, 0.0),
        np.full(n_right, -2.0),
    )
    projected_frame = renderer.render(lh[:, 0], rh[:, 0])
    second_blank_frame = renderer.render(np.full(n_left, np.nan), None)

    assert not frames_match(blank_frame, full_frame)
    assert not frames_match(full_frame, half_frame)
    assert projected_frame.shape == blank_frame.shape
    assert frames_match(blank_frame, second_blank_frame)


def test_surface_renderer_transparency(renderer, coordinates):
    "Test zeros and values below the threshold are transparent."
    left_coordinates, right_coordinates = coordinates
    n_left, n_right = left_coordinates.shape[0], right_coordinates.shape[0]
    blank_frame = renderer.render()

    assert np.array_equal(
        blank_frame, renderer.render(np.zeros(n_left), np.zeros(n_right))
    )
    assert np.array_equal(
        blank_frame, renderer.render(np.full(n_left, 0.4), np.full(n_right, -0.4))
    )
    assert not np.array_equal(
        blank_frame, renderer.render(np.full(n_left, 0.6), np.full(n_right, -0.6))
    )


def test_surface_renderer_label(renderer):
    "Test the label changes the frame and is cleared when omitted."
    blank_frame = renderer.render()

    assert not np.array_equal(blank_frame, renderer.render(label="t = 1.0 s"))
    assert np.array_equal(blank_frame, renderer.render())


def test_surface_renderer_views(surfaces, surface_data):
    "Test named, custom and single-hemisphere views."
    _, rh = surface_data
    views = {"rh": ["dorsal", (0, 45, 0)]}

    with SurfaceRenderer(
        *surfaces["inflated"], views=views, size=(100, 80)
    ) as surface_renderer:
        frame = surface_renderer.render(rh=rh[:, 0])

    assert surface_renderer.panels == [("rh", "dorsal"), ("rh", (0, 45, 0))]
    assert frame.shape == (80, 200, 3)

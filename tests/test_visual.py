import numpy as np, pytest

pv = pytest.importorskip("pyvista")

from bidsaid.visual import (
    SurfaceRenderer,
    create_polydata,
)


@pytest.fixture(scope="module")
def sphere_surfaces():
    """Two spheres (left shifted -x, right shifted +x) as ``(coordinates, faces)``."""
    sphere = pv.Sphere(radius=50)
    coordinates = np.asarray(sphere.points)
    faces = sphere.faces.reshape(-1, 4)[:, 1:]

    return (coordinates - [60, 0, 0], faces), (coordinates + [60, 0, 0], faces)


@pytest.fixture(scope="module")
def renderer(sphere_surfaces):
    left_surface, right_surface = sphere_surfaces
    sulcal_depths = (left_surface[0][:, 2], right_surface[0][:, 2])

    with SurfaceRenderer(
        left_surface,
        right_surface,
        sulcal_depths=sulcal_depths,
        size=(120, 90),
        color_range=(-1, 1),
        threshold=0.5,
    ) as surface_renderer:
        yield surface_renderer


def test_create_polydata(surface_mesh):
    "Test ``create_polydata`` keeps the vertices and faces."
    coordinates, faces = surface_mesh
    mesh = create_polydata(surface_mesh)

    assert mesh.n_points == coordinates.shape[0]
    assert mesh.n_cells == faces.shape[0]
    assert np.array_equal(mesh.faces.reshape(-1, 4)[:, 1:], faces)


def test_surface_renderer_updates_data(renderer, sphere_surfaces):
    "Test consecutive frames reflect the data passed to ``render``."
    (left_coordinates, _), (right_coordinates, _) = sphere_surfaces
    n_left, n_right = left_coordinates.shape[0], right_coordinates.shape[0]

    blank_frame = renderer.render()
    full_frame = renderer.render(np.full(n_left, 2.0), np.full(n_right, -2.0))
    half_frame = renderer.render(
        np.where(left_coordinates[:, 0] > -60, 2.0, 0.0), np.full(n_right, -2.0)
    )
    second_blank_frame = renderer.render(np.full(n_left, np.nan), None)

    diff = np.abs(blank_frame.astype(np.int16) - second_blank_frame.astype(np.int16))
    n_bad = np.count_nonzero(diff.any(axis=-1))
    print(
        f"{n_bad} / {diff.shape[0] * diff.shape[1]} pixels differ, max diff {diff.max()}"
    )
    ys, xs = np.nonzero(diff.any(axis=-1))
    try:
        print("bbox:", ys.min(), ys.max(), xs.min(), xs.max())
    except:
        pass

    assert not np.array_equal(blank_frame, full_frame)
    assert not np.array_equal(full_frame, half_frame)
    assert np.array_equal(blank_frame, second_blank_frame)


def test_surface_renderer_transparency(renderer, sphere_surfaces):
    "Test zeros and values below the threshold are transparent."
    (left_coordinates, _), (right_coordinates, _) = sphere_surfaces
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


def test_surface_renderer_views(sphere_surfaces):
    "Test named, custom and single-hemisphere views."
    left_surface, right_surface = sphere_surfaces
    views = {"rh": ["dorsal", (0, 45, 0)]}

    with SurfaceRenderer(
        left_surface, right_surface, views=views, size=(100, 80)
    ) as surface_renderer:
        frame = surface_renderer.render(rh=np.ones(right_surface[0].shape[0]))

    assert surface_renderer.panels == [("rh", "dorsal"), ("rh", (0, 45, 0))]
    assert frame.shape == (80, 200, 3)

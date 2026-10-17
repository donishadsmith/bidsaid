"""
Module for rendering surface data as images.

.. important::
    Requires ``pyvista``, which is installed with ``pip install bidsaid[visual]``.
"""

from typing import Literal, TypeAlias

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
from numpy.typing import NDArray

try:
    import pyvista as pv
except ImportError as error:
    raise ImportError(
        "``pyvista`` is required for ``bidsaid.visual``. Install it with "
        "``pip install bidsaid[visual]``."
    ) from error

from bidsaid.logging import setup_logger
from bidsaid.surface import SurfaceMesh, load_surface_data, load_surface_mesh

LGR = setup_logger(__name__)

ViewName: TypeAlias = Literal[
    "lateral", "medial", "dorsal", "ventral", "anterior", "posterior"
]

View: TypeAlias = ViewName | tuple[float, float, float]
"""A named view or a custom ``(x, y, z)`` rotation of the surface in degrees."""

VIEW_ORIENTATIONS: dict[ViewName, tuple[float, float, float]] = {
    "lateral": (0, 90, 90),
    "medial": (0, -90, -90),
    "dorsal": (0, 0, 0),
    "ventral": (0, 180, 0),
    "anterior": (90, 90, 90),
    "posterior": (270, 90, 90),
}

_RIGHT_HEMISPHERE_VIEWS: dict[ViewName, ViewName] = {
    "lateral": "medial",
    "medial": "lateral",
}

_SHADING = {"ambient": 0.05, "diffuse": 1.0, "specular": 0.1, "specular_power": 1}

_SULCAL_LAYER, _DATA_LAYER = "sulcal_depth", "surface_data"

DARK_SULCAL_CMAP = LinearSegmentedColormap.from_list(
    "dark_sulcal_depth", ["#4d4d4d", "#1f1f1f"]
)
"""Dark gray shading of the folding pattern, gyri slightly lighter than sulci."""


def get_colormap(cmap: str | Colormap) -> Colormap:
    """
    Get a Colormap by Name.

    Parameters
    ----------
    cmap : :obj:`str` or :obj:`Colormap`
        The name of a nilearn colormap (e.g. "cold_hot", "cyan_orange") or a matplotlib colormap
        (e.g. "RdBu_r").

    Returns
    -------
    Colormap
        The colormap.
    """
    if isinstance(cmap, Colormap):
        return cmap

    import matplotlib
    from nilearn.plotting import cm as nilearn_cm

    nilearn_cmap = getattr(nilearn_cm, cmap, None)
    if isinstance(nilearn_cmap, Colormap):
        return nilearn_cmap

    return matplotlib.colormaps[cmap]


def create_polydata(surface_mesh: SurfaceMesh) -> "pv.PolyData":
    """
    Create a pyvista Mesh From a Surface.

    Parameters
    ----------
    surface_mesh : :obj:`str`, :obj:`Path`, or :obj:`tuple[NDArray, NDArray]`
        Path to a GIFTI surface file or a ``(coordinates, faces)`` tuple.

    Returns
    -------
    pyvista.PolyData
        The triangulated mesh.
    """
    coordinates, faces = load_surface_mesh(surface_mesh)
    # [n_vertices_in_face, vertex_1, vertex_2, ...] for each face
    padded_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    return pv.PolyData(np.asarray(coordinates, dtype=float), padded_faces.ravel())


def _get_orientation(view: View, hemisphere: Literal["lh", "rh"]):
    """Convert a view into the rotation applied to a hemisphere's mesh."""
    if isinstance(view, str):
        if view not in VIEW_ORIENTATIONS:
            raise ValueError(
                f"'{view}' is not a valid view. Must be one of "
                f"{list(VIEW_ORIENTATIONS)} or a tuple of three angles."
            )

        if hemisphere == "rh":
            view = _RIGHT_HEMISPHERE_VIEWS.get(view, view)

        return VIEW_ORIENTATIONS[view]

    if len(view) != 3:
        raise ValueError("A custom view must be a tuple of three angles (x, y, z).")

    return tuple(view) if hemisphere == "lh" else (view[0], -view[1], -view[2])


class SurfaceRenderer:
    """
    Render Per-Vertex Data on Cortical Surfaces.

    Builds the scene (meshes, views, lighting, colormaps) once and re-uses it for every call to :meth:`render`,
    so rendering a frame only updates the vertex data. Each view of each hemisphere is a panel and ``layout``
    controls how the panels are arranged.

    Parameters
    ----------
    left_surface : :obj:`str`, :obj:`Path`, or :obj:`tuple[NDArray, NDArray]`
        The left hemisphere surface to display (e.g. the inflated surface), as a GIfTI file path or
        a ``(coordinates, faces)`` tuple.

    right_surface : :obj:`str`, :obj:`Path`, or :obj:`tuple[NDArray, NDArray]`
        The right hemisphere surface to display.

    views : :obj:`dict[str, list[View]]` or :obj:`None`, default=None
        The views to render for each hemisphere, with the keys "lh" and "rh". A hemisphere can be omitted
        to exclude it. Views are "lateral", "medial", "dorsal", "ventral", "anterior", "posterior" or a
        custom ``(x, y, z)`` rotation in degrees. If None, ``{"lh": ["lateral", "medial"], "rh": ["lateral", "medial"]}`` is used.

    layout : :obj:`Literal["row", "grid"]`, default="row"
        How the panels are arranged. "row" places all panels side by side in a single row, left hemisphere first. "grid"
        places the hemispheres in columns (left, right) and the views in rows, in the order given in ``views``.

    sulcal_depths : :obj:`tuple[str | NDArray, str | NDArray]` or :obj:`None`, default=None
        The ``(left, right)`` sulcal depth maps drawn underneath the data, as GIfTI file paths or arrays. If None,
        a flat backdrop is drawn instead.

    size : :obj:`tuple[int, int]`, default=(400, 300)
        Width and height of each panel in pixels.

    zoom : :obj:`float`, default=1.2
        Camera zoom applied to each panel.

    cmap : :obj:`str` or :obj:`Colormap`, default="cold_hot"
        Colormap of the data. Nilearn and matplotlib colormap names are accepted.

    color_range : :obj:`tuple[float, float]`, default=(-3.0, 3.0)
        The ``(vmin, vmax)`` of the colormap. Kept fixed across frames to avoid flickering.

    threshold : :obj:`float`, default=0.0
        Vertices whose absolute value is below this threshold are transparent.

    zero_transparent : :obj:`bool`, default=True
        Whether vertices with a value of exactly 0 are transparent (e.g. the medial wall after :func:`bidsaid.surface.project_to_surface`).

    background : :obj:`str` or :obj:`tuple[float, float, float]`, default="black"
        Background color of the panels.

    sulcal_cmap : :obj:`Colormap`, default=DARK_SULCAL_CMAP
        Colormap of the sulcal depth backdrop.

    label_font_size : :obj:`int`, default=10
        Font size of the optional label drawn in the first panel.

    antialiasing : :obj:`bool`, default=True
        Whether to use multisample antialiasing.

    Examples
    --------
    >>> from bidsaid.surface import fetch_fsaverage_surfaces, project_to_surface
    >>> from bidsaid.visual import SurfaceRenderer
    >>> surfaces = fetch_fsaverage_surfaces("10k")
    >>> lh, rh = project_to_surface(
    ...     "sub-01_task-rest_desc-denoised_bold.nii.gz",
    ...     surfaces["pial"],
    ...     surfaces["white"],
    ... )
    >>> with SurfaceRenderer(
    ...     *surfaces["inflated"], sulcal_depths=surfaces["sulc"], threshold=1.0
    ... ) as renderer:
    ...     frames = [
    ...         renderer.render(lh[:, index], right[:, index], label=f"t = {index}")
    ...         for index in range(lh.shape[1])
    ...     ]
    """

    def __init__(
        self,
        left_surface: SurfaceMesh,
        right_surface: SurfaceMesh,
        views: dict[str, list[View]] | None = None,
        layout: Literal["row", "grid"] = "row",
        sulcal_depths: tuple[str | NDArray, str | NDArray] | None = None,
        size: tuple[int, int] = (400, 300),
        zoom: float = 1.2,
        cmap: str | Colormap = "cold_hot",
        color_range: tuple[float, float] = (-3.0, 3.0),
        threshold: float = 0.0,
        zero_transparent: bool = True,
        background: str | tuple[float, float, float] = "black",
        sulcal_cmap: Colormap = DARK_SULCAL_CMAP,
        label_font_size: int = 10,
        antialiasing: bool = True,
    ):
        views = views or {"lh": ["lateral", "medial"], "rh": ["lateral", "medial"]}
        if not set(views).issubset({"lh", "rh"}) or not views:
            raise ValueError(
                "``views`` must be a dictionary with the keys 'lh' and/or 'rh'."
            )

        if layout not in ("row", "grid"):
            raise ValueError("``layout`` must be 'row' or 'grid'.")

        self.threshold = threshold
        self.zero_transparent = zero_transparent
        self.layout = layout
        self.meshes = {
            "lh": create_polydata(left_surface),
            "rh": create_polydata(right_surface),
        }
        self.panels = [
            (hemisphere, view)
            for hemisphere in ("lh", "rh")
            for view in views.get(hemisphere, [])
        ]
        self.positions = self._get_panel_positions(views)
        n_rows = max(row for row, _ in self.positions.values()) + 1
        n_columns = max(column for _, column in self.positions.values()) + 1

        for hemisphere_index, hemisphere in enumerate(("lh", "rh")):
            mesh = self.meshes[hemisphere]
            if sulcal_depths is not None:
                sulcal_depth = load_surface_data(sulcal_depths[hemisphere_index])
                if sulcal_depth.shape[0] != mesh.n_points:
                    raise ValueError(
                        f"The {hemisphere} sulcal depth map has {sulcal_depth.shape[0]} "
                        f"values but the surface has {mesh.n_points} vertices."
                    )
            else:
                sulcal_depth = np.zeros(mesh.n_points)

            mesh[_SULCAL_LAYER] = sulcal_depth.astype(float)
            mesh[_DATA_LAYER] = np.full(mesh.n_points, np.nan)

        self.plotter = pv.Plotter(
            off_screen=True,
            shape=(n_rows, n_columns),
            window_size=(size[0] * n_columns, size[1] * n_rows),
            border=False,
            lighting="none",
        )
        if antialiasing:
            self.plotter.render_window.SetMultiSamples(8)

        self.plotter.set_background(background)
        data_cmap = get_colormap(cmap)
        for panel_index, (hemisphere, view) in enumerate(self.panels):
            self._add_panel(
                self.positions[panel_index],
                hemisphere,
                view,
                data_cmap,
                color_range,
                sulcal_cmap,
                zoom,
            )

        self.plotter.subplot(0, 0)
        self._label = self.plotter.add_text(
            "", position="upper_left", font_size=label_font_size, color="white"
        )

    def _get_panel_positions(
        self, views: dict[str, list[View]]
    ) -> dict[int, tuple[int, int]]:
        """Map each panel index to its ``(row, column)`` in the plotter."""
        if self.layout == "row":
            return {
                panel_index: (0, panel_index) for panel_index in range(len(self.panels))
            }

        hemispheres = [
            hemisphere for hemisphere in ("lh", "rh") if views.get(hemisphere)
        ]
        positions, panel_index = {}, 0
        for column, hemisphere in enumerate(hemispheres):
            for row in range(len(views[hemisphere])):
                positions[panel_index] = (row, column)
                panel_index += 1

        return positions

    def _add_panel(
        self,
        position: tuple[int, int],
        hemisphere: Literal["lh", "rh"],
        view: Literal[
            "lateral", "medial", "dorsal", "ventral", "anterior", "posterior"
        ],
        data_cmap: Colormap,
        color_range: tuple[float, float],
        sulcal_cmap: Colormap,
        zoom: float,
    ):
        """Add the two layers (sulcal depth and data) of one hemisphere to a panel."""
        mesh = self.meshes[hemisphere]
        orientation = _get_orientation(view, hemisphere)

        self.plotter.subplot(*position)
        self.plotter.add_light(pv.Light(light_type="headlight"), only_active=True)

        sulcal_depth = mesh[_SULCAL_LAYER]
        sulcal_actor = self.plotter.add_mesh(
            mesh,
            scalars=_SULCAL_LAYER,
            cmap=sulcal_cmap,
            clim=(sulcal_depth.min(), max(sulcal_depth.max(), sulcal_depth.min() + 1)),
            show_scalar_bar=False,
            interpolate_before_map=True,
            **_SHADING,
        )
        sulcal_actor.orientation = orientation

        data_actor = self.plotter.add_mesh(
            mesh,
            scalars=_DATA_LAYER,
            cmap=data_cmap,
            clim=color_range,
            nan_opacity=0.0,
            show_scalar_bar=False,
            interpolate_before_map=True,
            **_SHADING,
        )
        data_actor.orientation = orientation
        data_actor.force_opaque = True

        self.plotter.camera_position = "xy"
        self.plotter.camera.parallel_projection = True
        self.plotter.reset_camera()
        self.plotter.camera.zoom(zoom)

    @property
    def n_panels(self) -> int:
        """The number of panels (hemisphere and view combinations) in a frame."""
        return len(self.panels)

    def _mask_transparent(self, data: NDArray, hemisphere: str) -> NDArray:
        """Return a float copy of the data with transparent vertices set to NaN."""
        data = np.array(data, dtype=float)
        n_points = self.meshes[hemisphere].n_points
        if data.shape != (n_points,):
            raise ValueError(
                f"The {hemisphere} data must have shape ({n_points},) to match the "
                f"surface, not {data.shape}."
            )

        transparent = np.isnan(data)
        if self.zero_transparent:
            transparent |= data == 0

        if self.threshold > 0:
            transparent |= np.abs(data) < self.threshold

        data[transparent] = np.nan

        return data

    def render(
        self,
        lh: NDArray | None = None,
        rh: NDArray | None = None,
        label: str | None = None,
    ) -> NDArray[np.uint8]:
        """
        Render a Frame.

        Parameters
        ----------
        lh : :obj:`NDArray` or :obj:`None`, default=None
            One value per vertex of the left surface. If None, the left hemisphere shows
            only its backdrop.

        rh : :obj:`NDArray` or :obj:`None`, default=None
            One value per vertex of the right surface. If None, the right hemisphere shows
            only its backdrop.

        label : :obj:`str` or :obj:`None`, default=None
            Text drawn in the upper left corner of the first panel.

        Returns
        -------
        NDArray[np.uint8]
            An RGB image with shape (height, width, 3).
        """
        for hemisphere, data in (("lh", lh), ("rh", rh)):
            mesh = self.meshes[hemisphere]
            if data is None:
                mesh[_DATA_LAYER] = np.full(mesh.n_points, np.nan)
            else:
                mesh[_DATA_LAYER] = self._mask_transparent(data, hemisphere)

        self._label.set_text("upper_left", label or "")

        self.plotter.render()

        return self.plotter.screenshot(return_img=True)

    def close(self) -> None:
        """Release the render window."""
        self.plotter.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


__all__ = [
    "DARK_SULCAL_CMAP",
    "VIEW_ORIENTATIONS",
    "SurfaceRenderer",
    "create_polydata",
    "get_colormap",
]

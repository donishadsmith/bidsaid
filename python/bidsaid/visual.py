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
    from vtkmodules.vtkRenderingCore import vtkRenderer
except ImportError as error:
    raise ImportError(
        "``pyvista`` is required for ``bidsaid.visual``. Install it with ``pip install bidsaid[visual]``."
    ) from error

from bidsaid.logging import setup_logger
from bidsaid.surface import SurfaceMesh, load_surface_data, load_surface_mesh

LGR = setup_logger(__name__)

ViewName: TypeAlias = Literal[
    "lateral", "medial", "dorsal", "ventral", "anterior", "posterior"
]

View: TypeAlias = ViewName | tuple[float, float, float]
"""A named view or a custom ``(x, y, z)`` rotation of the surface in degrees."""

LabelAnchor: TypeAlias = tuple[
    Literal["left", "center", "right"], Literal["bottom", "center", "top"]
]
"""The ``(horizontal, vertical)`` point of the label text placed at ``label_position``."""

_HORIZONTAL_ANCHORS, _VERTICAL_ANCHORS = ("left", "center", "right"), (
    "bottom",
    "center",
    "top",
)

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
"""Dark gray shading of the folding pattern of gyri and sulci, respectively."""


def _validate_label_position(label_position: tuple[float, float], param_name: str):
    if not all(0 <= value <= 1 for value in label_position):
        raise ValueError(
            f"Both 'x' and 'y' of ``{param_name}`` must be between 0 and 1."
        )


def _validate_anchor(anchor: LabelAnchor, name: str) -> None:
    """Raise if ``anchor`` is not a valid ``(horizontal, vertical)`` pair."""
    if anchor[0] not in _HORIZONTAL_ANCHORS or anchor[1] not in _VERTICAL_ANCHORS:
        raise ValueError(
            f"``{name}`` must be a (horizontal, vertical) tuple with horizontal in "
            f"{_HORIZONTAL_ANCHORS} and vertical in {_VERTICAL_ANCHORS}."
        )


def _create_text(
    text: str,
    position: tuple[float, float],
    anchor: LabelAnchor,
    font_size: int,
    color: str | tuple[float, float, float],
) -> "pv.Text":
    """
    Create a text actor positioned in normalized viewport coordinates.

    ``position`` is a fraction of the viewport of whichever renderer the actor is added to
    (a panel or the full-frame overlay) and ``anchor`` is the point of the text placed there.
    """
    text_actor = pv.Text(text, position=(0, 0))
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.position = tuple(position)
    text_actor.prop.justification_horizontal, text_actor.prop.justification_vertical = (
        anchor
    )
    text_actor.prop.font_size = font_size
    text_actor.prop.color = color

    return text_actor


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

    Builds the scene (meshes, views, lighting, colormaps) once and re-uses it for every call to :meth:`render`, so rendering a frame
    only updates the vertex data. Each view of each hemisphere is a panel and ``layout`` Scontrols how the panels are arranged.

    Parameters
    ----------
    left_surface : :obj:`str`, :obj:`Path`, or :obj:`tuple[NDArray, NDArray]`
        The left hemisphere surface to display (e.g. the inflated surface), as a GIfTI file path or
        a ``(coordinates, faces)`` tuple.

    right_surface : :obj:`str`, :obj:`Path`, or :obj:`tuple[NDArray, NDArray]`
        The right hemisphere surface to display.

    views : :obj:`dict[str, list[View]]` or :obj:`None`, default=None
        The views to render for each hemisphere, with the keys "lh" and "rh". A hemisphere can be omitted to exclude it. Views are "lateral", "medial", "dorsal",
        "ventral", "anterior", "posterior" or a custom ``(x, y, z)`` rotation in degrees. If None, ``{"lh": ["lateral", "medial"], "rh": ["lateral", "medial"]}`` is used.

    layout : :obj:`Literal["row", "grid"]`, default="row"
        How the panels are arranged. "row" places all panels side by side in a single row, left hemisphere first. "grid"
        places the hemispheres in columns (left, right) and the views in rows, in the order given in ``views``.

    sulcal_depths : :obj:`tuple[str | NDArray, str | NDArray]` or :obj:`None`, default=None
        The ``(left, right)`` sulcal depth maps drawn underneath the data, as GIfTI file paths or arrays. If None, a flat backdrop is drawn instead.

    size : :obj:`tuple[int, int]`, default=(400, 300)
        Width and height of each panel in pixels.

    zoom : :obj:`float`, default=1.2
        Camera zoom applied to each panel.

    hemisphere_spacing : :obj:`int`, default=0
        Width in pixels of the empty space between the left and right hemisphere panels (between the last left panel and the first right panel in "row" layout,
        or between the two columns in "grid" layout). Brains maintain there size but the spacing between the panels change.

    top_spacing : :obj:`int`, default=0
        Height in pixels of an header space above all panels. Brains maintain there size but the spacing from the top of the panels change.

    cmap : :obj:`str` or :obj:`Colormap`, default="cold_hot"
        Colormap of the data. Nilearn and matplotlib colormap names are accepted.

    color_range : :obj:`tuple[float, float]`, default=(-3.0, 3.0)
        The ``(vmin, vmax)`` of the colormap.

    threshold : :obj:`float`, default=0.0
        Vertices whose absolute value is below this threshold are transparent.

    zero_transparent : :obj:`bool`, default=True
        Whether vertices with a value of exactly 0 are transparent.

    background : :obj:`str` or :obj:`tuple[float, float, float]`, default="black"
        Background color of the panels.

    sulcal_cmap : :obj:`Colormap`, default=DARK_SULCAL_CMAP
        Colormap of the sulcal depth backdrop.

    label_font_size : :obj:`int`, default=30
        Font size of the optional label drawn over the frame.

    label_color : :obj:`str` or :obj:`tuple[float, float, float]`, default="white"
        Color of the optional label drawn over the frame.

    panel_labels : :obj:`dict[Literal["lh", "rh"], str]` or :obj:`None`, default=None
        Static text drawn in every panel of a hemisphere (e.g. ``{"lh": "L", "rh": "R"}``).

    panel_label_position : :obj:`tuple[float, float]`, default=(0.5, 0.02)
        The ``(x, y)`` position of the panel labels as fractions of the panel's width and height, with ``(0, 0)`` the lower left corner of the panel and ``(1, 1)`` its upper right.

    panel_label_anchor : :obj:`tuple[Literal["left", "center", "right"], Literal["bottom", "center", "top"]]`, default=("center", "bottom")
        The ``(horizontal, vertical)`` point of the panel label text placed at ``panel_label_position``.

    panel_label_font_size : :obj:`int`, default=20
        Font size of the panel labels.

    panel_label_color : :obj:`str` or :obj:`tuple[float, float, float]`, default="white"
        Color of the panel labels.

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
        hemisphere_spacing: int = 0,
        top_spacing: int = 0,
        cmap: str | Colormap = "cold_hot",
        color_range: tuple[float, float] = (-3.0, 3.0),
        threshold: float = 0.0,
        zero_transparent: bool = True,
        background: str | tuple[float, float, float] = "black",
        sulcal_cmap: Colormap = DARK_SULCAL_CMAP,
        label_font_size: int = 30,
        label_color: str | tuple[float, float, float] = "white",
        panel_labels: dict[Literal["lh", "rh"], str] | None = None,
        panel_label_position: tuple[float, float] = (0.5, 0.02),
        panel_label_anchor: LabelAnchor = ("center", "bottom"),
        panel_label_font_size: int = 20,
        panel_label_color: str | tuple[float, float, float] = "white",
        antialiasing: bool = True,
    ):
        views = views or {"lh": ["lateral", "medial"], "rh": ["lateral", "medial"]}
        if not set(views).issubset({"lh", "rh"}) or not views:
            raise ValueError(
                "``views`` must be a dictionary with the keys 'lh' and/or 'rh'."
            )

        if layout not in ("row", "grid"):
            raise ValueError("``layout`` must be 'row' or 'grid'.")

        if hemisphere_spacing < 0 or top_spacing < 0:
            raise ValueError(
                "``hemisphere_spacing`` and ``top_spacing`` must be 0 or greater."
            )

        _validate_anchor(panel_label_anchor, "panel_label_anchor")

        self.threshold = threshold
        self.zero_transparent = zero_transparent
        self.layout = layout
        self._color_range = tuple(color_range)
        self._data_actors = []
        self.meshes = {
            "lh": create_polydata(left_surface),
            "rh": create_polydata(right_surface),
        }
        self.panels = [
            (hemisphere, view)
            for hemisphere in ("lh", "rh")
            for view in views.get(hemisphere, [])
        ]

        self.hemisphere_spacing = (
            hemisphere_spacing
            if all(views.get(hemisphere) for hemisphere in ("lh", "rh"))
            else 0
        )

        self.top_spacing = top_spacing
        self.positions, self._spacer_column = self._get_panel_positions(views)
        n_rows = max(row for row, _ in self.positions.values()) + 1
        n_columns = max(column for _, column in self.positions.values()) + 1
        column_widths = [size[0]] * n_columns
        if self.hemisphere_spacing:
            column_widths[self._spacer_column] = self.hemisphere_spacing

        row_heights = [size[1]] * n_rows
        if self.top_spacing:
            row_heights[0] = self.top_spacing

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
            row_weights=row_heights,
            col_weights=column_widths,
            window_size=(sum(column_widths), sum(row_heights)),
            border=False,
            lighting="none",
        )
        self._column_widths, self._row_heights = column_widths, row_heights

        if antialiasing:
            self.plotter.render_window.SetMultiSamples(8)

        self.plotter.set_background(background)
        self.background_rgb = tuple(pv.Color(background).int_rgb)
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

        _validate_label_position(panel_label_position, "panel_label_position")
        for panel_index, (hemisphere, _) in enumerate(self.panels):
            if text := (panel_labels or {}).get(hemisphere):
                self.plotter.subplot(*self.positions[panel_index])
                self.plotter.renderer.AddActor(
                    _create_text(
                        text,
                        panel_label_position,
                        panel_label_anchor,
                        panel_label_font_size,
                        panel_label_color,
                    )
                )

        self._overlay = vtkRenderer()
        self._overlay.SetViewport(0, 0, 1, 1)
        self._overlay.SetLayer(1)
        self._overlay.InteractiveOff()
        self.plotter.render_window.SetNumberOfLayers(2)
        self.plotter.render_window.AddRenderer(self._overlay)

        self._label = _create_text(
            "", (0.02, 0.98), ("left", "top"), label_font_size, label_color
        )
        self._overlay.AddActor(self._label)

    def _get_panel_positions(
        self, views: dict[str, list[View]]
    ) -> dict[int, tuple[int, int]]:
        """
        Map each panel index to its ``(row, column)`` in the plotter.

        When ``hemisphere_spacing`` is used, the right hemisphere panels are shifted one column to the right to leave the spacer column
        (``_spacer_column``) empty. When ``top_spacing`` is used, all panels are shifted one row down to leave the first row empty.
        """
        n_left = len(views.get("lh", []))
        first_row = int(bool(self.top_spacing))
        if self.layout == "row":
            return (
                {
                    panel_index: (
                        first_row,
                        panel_index
                        + (panel_index >= n_left) * bool(self.hemisphere_spacing),
                    )
                    for panel_index in range(len(self.panels))
                },
                n_left,
            )

        hemispheres = [
            hemisphere for hemisphere in ("lh", "rh") if views.get(hemisphere)
        ]
        positions, panel_index = {}, 0
        for column, hemisphere in enumerate(hemispheres):
            column += (hemisphere == "rh") * bool(self.hemisphere_spacing)
            for row in range(len(views[hemisphere])):
                positions[panel_index] = (first_row + row, column)
                panel_index += 1

        return positions, 1

    @property
    def hemisphere_spacing_center(self) -> tuple[float, float]:
        """
        The ``(x, y)`` center of the gutter between the hemispheres.

        Pass it as ``label_position`` with ``label_anchor=("center", "center")`` to draw the label in the gutter. Equals ``(0.5, 0.5)`` when
        both hemispheres have the same number of views.
        """
        if not self.hemisphere_spacing:
            return (0.5, 0.5)

        left_width = sum(self._column_widths[: self._spacer_column])
        center = left_width + self.hemisphere_spacing / 2

        return (center / sum(self._column_widths), 0.5)

    @property
    def top_spacing_center(self) -> tuple[float, float]:
        """
        The ``(x, y)`` center of the header above the panel.

        Pass it as ``label_position`` with ``label_anchor=("center", "center")`` to draw the label centered above the brains. Equals ``(0.5, 0.98)``
        when there is no ``top_spacing``.
        """
        if not self.top_spacing:
            return (0.5, 0.98)

        return (0.5, 1 - self.top_spacing / 2 / sum(self._row_heights))

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
        self._data_actors.append(data_actor)

        self.plotter.camera_position = "xy"
        self.plotter.camera.parallel_projection = True
        self.plotter.reset_camera()
        self.plotter.camera.zoom(zoom)

    @property
    def n_panels(self) -> int:
        """The number of panels (hemisphere and view combinations) in a frame."""
        return len(self.panels)

    @property
    def color_range(self) -> tuple[float, float]:
        """
        The ``(vmin, vmax)`` of the data colormap.

        Can be reassigned between calls to :meth:`render` (e.g. to give each task of a movie its own color range) without rebuilding the scene.
        """
        return self._color_range

    @color_range.setter
    def color_range(self, color_range: tuple[float, float]) -> None:
        vmin, vmax = color_range
        if vmin >= vmax:
            raise ValueError(
                "``color_range`` must be a (vmin, vmax) tuple with vmin < vmax."
            )

        self._color_range = (float(vmin), float(vmax))
        for data_actor in self._data_actors:
            data_actor.mapper.scalar_range = self._color_range

    def _mask_transparent(self, data: NDArray, hemisphere: str) -> NDArray:
        """Return a float copy of the data with transparent vertices set to NaN."""
        data = np.array(data, dtype=float)
        n_points = self.meshes[hemisphere].n_points
        if data.shape != (n_points,):
            raise ValueError(
                f"The {hemisphere} data must have shape ({n_points},) to match the surface, not {data.shape}."
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
        label_position: tuple[float, float] = (0.02, 0.98),
        label_anchor: LabelAnchor = ("left", "top"),
    ) -> NDArray[np.uint8]:
        """
        Render a Frame.

        Parameters
        ----------
        lh : :obj:`NDArray` or :obj:`None`, default=None
            One value per vertex of the left surface. If None, the left hemisphere shows only its backdrop.

        rh : :obj:`NDArray` or :obj:`None`, default=None
            One value per vertex of the right surface. If None, the right hemisphere shows only its backdrop.

        label : :obj:`str` or :obj:`None`, default=None
            Text drawn over the frame.

        label_position : :obj:`tuple[float, float]`, default=(0.02, 0.98)
            The ``(x, y)`` position of the label as fractions of the frame's width and height, with ``(0, 0)`` the lower left corner
            of the frame and ``(1, 1)`` its upper right (increasing ``y`` places the label closer to the top of the frame).

        label_anchor : :obj:`tuple[Literal["left", "center", "right"], Literal["bottom", "center", "top"]]`, default=("left", "top")
            The ``(horizontal, vertical)`` point of the text placed at ``label_position``. For instance,
            ``("center", "top")`` with ``label_position=(0.5, 0.98)`` centers the label at the top of the image.

        Returns
        -------
        NDArray[np.uint8]
            An RGB image with shape (height, width, 3).
        """
        for hemisphere, data in (("lh", lh), ("rh", rh)):
            if data is None:
                self.meshes[hemisphere][_DATA_LAYER] = np.full(
                    self.meshes[hemisphere].n_points, np.nan
                )
            else:
                self.meshes[hemisphere][_DATA_LAYER] = self._mask_transparent(
                    data, hemisphere
                )

        _validate_label_position(label_position, "label_position")
        _validate_anchor(label_anchor, "label_anchor")
        self._label.position = tuple(label_position)
        (
            self._label.prop.justification_horizontal,
            self._label.prop.justification_vertical,
        ) = label_anchor
        self._label.input = label or ""

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
    "LabelAnchor",
    "VIEW_ORIENTATIONS",
    "SurfaceRenderer",
    "create_polydata",
    "get_colormap",
]

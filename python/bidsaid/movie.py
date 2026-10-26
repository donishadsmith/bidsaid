"""
Module for writing movies of surface data.

.. important::
    Requires ``imageio`` with ffmpeg support and ``pyvista``, which are installed with
    ``pip install bidsaid[movie]``.
"""

import os, string, subprocess, sys, time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError as error:
    raise ImportError(
        "``imageio`` is required for ``bidsaid.movie``. Install it with  ``pip install bidsaid[movie]``."
    ) from error

from bidsaid._helpers import iterable_to_str
from bidsaid.files import get_entity_value
from bidsaid.io import NiftiFileorImage
from bidsaid.logging import setup_logger
from bidsaid.metadata import get_tr
from bidsaid.qc.interpolate import upsample_array
from bidsaid.surface import SurfaceMesh, project_to_surface
from bidsaid.visual import (
    LabelAnchor,
    SurfaceRenderer,
    _validate_anchor,
    _validate_label_position,
)

LGR = setup_logger(__name__)

LABEL_FIELDS = ("subject_id", "subject_name", "task_name", "frame_index", "frame_time")
"""The fields that can be used in the ``label_format`` of :class:`MovieWriter`."""

DEFAULT_LABEL_FORMAT = "{subject_name}    {task_name}    t = {frame_time:.1f} s"
"""The default ``label_format`` of :class:`MovieWriter`."""


def compute_frame_timing(
    tr: float,
    n_interpolated_frames: int,
    target_fps: float,
    playback_speed: float | None = None,
) -> tuple[float, int]:
    """
    Compute Frame Timing.

    Returns the output frame rate of a movie and the number of times each rendered frame must be written to play at ``playback_speed``.
    Rendered frames occur every ``tr / (n_interpolated_frames + 1)`` seconds of real time, so playing them at ``playback_speed`` requires a frame rate of
    ``playback_speed * (n_interpolated_frames + 1) / tr``. Each rendered frame is repeated so the output frame rate is a multiple of that rate close to ``target_fps``.

    Parameters
    ----------
    tr : :obj:`float`
        The repetition time (seconds) of the original (not upsampled) data.

    n_interpolated_frames : :obj:`int`
        The number of interpolated frames inserted in each gap between two real volumes.

    target_fps : :obj:`float`
        The targeted frames per second of the movie (e.g. 24 or 30).

    playback_speed : :obj:`float` or :obj:`None`, default=None
        The speed relative to real time (e.g. 1 for real time and 10 for 10x faster than real time). If None, each rendered frame is
        one output frame at exactly ``target_fps``.

    Returns
    -------
    tuple[float, int]
        The output frames per second and the number of times each rendered frame must be
        repeated to play at ``playback_speed`` at that frame rate.

    Examples
    --------
    >>> from bidsaid.movie import compute_frame_timing
    >>> compute_frame_timing(tr=2.0, n_interpolated_frames=1, target_fps=24, playback_speed=1)
    (24.0, 24)
    >>> compute_frame_timing(tr=2.0, n_interpolated_frames=1, target_fps=24, playback_speed=None)
    (24, 1)
    """
    if playback_speed is None:
        return target_fps, 1

    effective_fps = playback_speed * (n_interpolated_frames + 1) / tr
    n_repeats_per_frame = max(1, round(target_fps / effective_fps))

    return effective_fps * n_repeats_per_frame, n_repeats_per_frame


def _add_entity_prefix(value: str, entity: str) -> str:
    return f"{entity}-{value.removeprefix(f'{entity}-')}"


class SurfaceRun(NamedTuple):
    """
    Per-Vertex Timeseries of One Participant and One Task.

    Created directly from arrays or from a NIfTI file with :func:`create_surface_run`.

    Parameters
    ----------
    subject_id : :obj:`str`
        The participant label.

    task_name : :obj:`str`
        The task label (e.g. "rest", "music_minus_rest").

    lh : :obj:`NDArray[np.floating]`
        The left hemisphere data with shape (n_vertices, n_frames).

    rh : :obj:`NDArray[np.floating]`
        The right hemisphere data with shape (n_vertices, n_frames).

    tr : :obj:`float`
        The repetition time (seconds) of the original (not upsampled) data.

    n_interpolated_frames : :obj:`int`, default=0
        The number of interpolated frames inserted in each gap between two real volumes (see :func:`bidsaid.qc.upsample_array`). Used with ``tr`` to compute the time of each frame (``frame_tr``).

    name : :obj:`str` or :obj:`None`, default=None
        The name shown in the frame label. If None, ``subject_id`` is used.

    color_range : :obj:`tuple[float, float]` or :obj:`None`, default=None
        The ``(vmin, vmax)`` of the colormap for this run. If None, the color range of the renderer is used.
    """

    subject_id: str
    task_name: str
    lh: NDArray[np.floating]
    rh: NDArray[np.floating]
    tr: float
    n_interpolated_frames: int = 0
    name: str | None = None
    color_range: tuple[float, float] | None = None

    @property
    def n_frames(self) -> int:
        """The number of frames (timepoints) in the run."""
        return self.lh.shape[1]

    @property
    def frame_tr(self) -> float:
        """The time (seconds) between consecutive frames, accounting for interpolation."""
        return self.tr / (self.n_interpolated_frames + 1)

    @property
    def display_name(self) -> str:
        """The ``name`` if given else ``subject_id`` is returned."""
        return self.name if self.name else self.subject_id


def create_surface_run(
    nifti_file_or_img: NiftiFileorImage,
    pial_surfaces: tuple[SurfaceMesh, SurfaceMesh],
    white_surfaces: tuple[SurfaceMesh, SurfaceMesh] | None = None,
    cortex_masks: list[NDArray[np.bool_]] | None = None,
    subject_id: str | None = None,
    task_name: str | None = None,
    name: str | None = None,
    tr: float | None = None,
    n_interpolated_frames: int = 0,
    upsample_interpolation_method: Literal["cubic", "pchip"] = "pchip",
    fwhm: float | int | None = None,
    temporal_sigma: float | int | None = None,
    surface_interpolation_method: Literal["linear", "nearest"] = "linear",
    max_volumes: int | None = None,
    color_range: tuple[float, float] | None = None,
    dtype: type = np.float32,
) -> SurfaceRun:
    """
    Create a Surface Run From a NIfTI Image.

    Projects the image onto the surfaces with :func:`bidsaid.surface.project_to_surface` and
    optionally upsamples it in time with :func:`bidsaid.qc.upsample_array`.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the 4D functional NIfTI file or a 4D functional NIfTI image.

    pial_surfaces : :obj:`tuple[SurfaceMesh, SurfaceMesh]`
        The ``(left, right)`` pial surfaces, as GIFTI file paths or ``(coordinates, faces)`` tuples.

    white_surfaces : :obj:`tuple[SurfaceMesh, SurfaceMesh]` or :obj:`None`, default=None
        The ``(left, right)`` white matter surfaces used as the inner mesh.

    cortex_masks : :obj:`list[NDArray[np.bool_]]` or :obj:`None`, default=None
        The ``[left, right]`` cortex masks from :func:`bidsaid.surface.get_cortex_masks`. Vertices outside the mask (the medial wall) are set to 0.

    subject_id : :obj:`str` or :obj:`None`, default=None
        The participant label without the "sub-" prefix (e.g. "01"). If None, it is read from the "sub" entity of the filename.

    task_name : :obj:`str` or :obj:`None`, default=None
        The task label without the "task-" prefix (e.g. "rest"). If None, it is read from the "task" entity of the filename.

    name : :obj:`str` or :obj:`None`, default=None
        The name shown in the frame label. If None, defaults to ``subject_id``.

    tr : :obj:`float` or :obj:`None`, default=None
        The repetition time (seconds). If None, it is read from the NIfTI header.

    n_interpolated_frames : :obj:`int`, default=0
        The number of interpolated frames inserted in each gap between two real volumes.

    upsample_interpolation_method : :obj:`Literal["cubic", "pchip"]`, default="cubic"
        The interpolation method used when ``n_interpolated_frames`` is greater than 0.

    fwhm : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Full width at half maximum (mm) of the Gaussian smoothing applied before projection.

    temporal_sigma : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Sigma value passed to `Scipy's ``gaussian_filter1d`` function for temporal smoothing
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html>_`

    surface_interpolation_method : :obj:`Literal["linear", "nearest"]`, default="linear"
        Interpolation used when sampling the image.

    max_volumes : :obj:`int` or :obj:`None`, default=None
        Only use the first N volumes.

    color_range : :obj:`tuple[float, float]` or :obj:`None`, default=None
        The ``(vmin, vmax)`` of the colormap for this run. If None, the color range of the renderer is used.

    dtype : :obj:`type`, default=np.float32
        Data type of the surface arrays.

    Returns
    -------
    SurfaceRun
        The surface run.
    """
    is_file = isinstance(nifti_file_or_img, (str, Path))
    if not is_file and not all((subject_id, task_name)):
        raise ValueError(
            "``subject_id`` and ``task_name`` must be specified when ``nifti_file_or_img`` is an image."
        )

    if subject_id is None and is_file:
        subject_id = get_entity_value(nifti_file_or_img, "sub")

    if task_name is None and is_file:
        task_name = get_entity_value(nifti_file_or_img, "task")

    if subject_id is None or task_name is None:
        raise ValueError(
            "``subject_id`` and ``task_name`` must be specified when they cannot be read from the filename."
        )

    source = Path(nifti_file_or_img).name if is_file else f"{subject_id} {task_name}"

    LGR.info(f"Projecting to surface: {source}")
    lh, rh = project_to_surface(
        nifti_file_or_img,
        pial_surfaces,
        white_surfaces,
        cortex_masks,
        fwhm=fwhm,
        temporal_sigma=temporal_sigma,
        max_volumes=max_volumes,
        interpolation_method=surface_interpolation_method,
        dtype=dtype,
    )

    if n_interpolated_frames > 0:
        lh = upsample_array(
            lh, n_interpolated_frames, axis=1, method=upsample_interpolation_method
        )
        rh = upsample_array(
            rh, n_interpolated_frames, axis=1, method=upsample_interpolation_method
        )

    return SurfaceRun(
        subject_id=subject_id,
        task_name=task_name,
        lh=lh.astype(dtype, copy=False),
        rh=rh.astype(dtype, copy=False),
        tr=tr if tr is not None else get_tr(nifti_file_or_img),
        n_interpolated_frames=n_interpolated_frames,
        name=name,
        color_range=color_range,
    )


def compute_vmax(
    arrays: list[NDArray[np.floating] | SurfaceRun],
    percentile: float = 98,
    threshold: float = 0.0,
) -> float:
    """
    Compute a Symmetric Color Maximum.

    Computes a percentile of the absolute values across all arrays, ignoring NaNs, zeros (e.g. the medial wall) and values below
    ``threshold`` (values that are transparent when rendered).

    Parameters
    ----------
    arrays : :obj:`list[NDArray[np.floating] | SurfaceRun]`
        The arrays (or surface runs, whose hemispheres are both used) to compute the maximum from.

    percentile : :obj:`float`, default=98
        The percentile of the absolute values to use.

    threshold : :obj:`float`, default=0.0
        Values with an absolute value below this threshold are ignored.

    Returns
    -------
    float
        The value to use as ``vmax`` (with ``-vmax`` as ``vmin``). Returns 1.0 if no values remain after masking.
    """
    hemispheres = []
    for array in arrays:
        hemispheres.extend(
            (array.lh, array.rh) if isinstance(array, SurfaceRun) else (array,)
        )

    absolute_values = np.concatenate(
        [
            np.abs(np.asarray(hemisphere, dtype=float).ravel())
            for hemisphere in hemispheres
        ]
    )
    absolute_values = absolute_values[
        ~np.isnan(absolute_values)
        & (absolute_values != 0)
        & (absolute_values >= threshold)
    ]

    return (
        float(np.percentile(absolute_values, percentile))
        if absolute_values.size
        else 1.0
    )


@contextmanager
def virtual_display(
    display: str = ":99", screen_size: tuple[int, int] = (1920, 1080), wait: float = 1.0
) -> Generator[subprocess.Popen | None, None, None]:
    """
    Start an X Virtual Framebuffer While Rendering Headless on Linux.

    A context manager that starts ``Xvfb`` and sets the ``DISPLAY`` environment variable for headless environment.

    .. important::
        Must be called before the :class:`bidsaid.visual.SurfaceRenderer` is created.

    Parameters
    ----------
    display : :obj:`str`, default=":99"
        The X display to start ``Xvfb`` on.

    screen_size : :obj:`tuple[int, int]`, default=(1920, 1080)
        Width and height of the virtual screen in pixels.

    wait : :obj:`float`, default=1.0
        Seconds to wait for ``Xvfb`` to start before checking that it is running.

    Yields
    ------
    subprocess.Popen or None
        The ``Xvfb`` process, or None when no virtual display was started.

    Examples
    --------
    >>> from bidsaid.movie import virtual_display
    >>> from bidsaid.visual import SurfaceRenderer
    >>> with virtual_display(), SurfaceRenderer(left_surface, right_surface) as renderer:
    ...     frame = renderer.render(lh, rh)
    """
    if not sys.platform.startswith("linux") or os.environ.get("DISPLAY"):
        yield None

        return

    process = None
    try:
        process = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", f"{screen_size[0]}x{screen_size[1]}x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(wait)
        if process.poll() is not None:
            raise OSError(f"Xvfb exited with code {process.returncode}")

        os.environ["DISPLAY"] = display
        LGR.info(
            f"No display found, started a virtual framebuffer (Xvfb on {display})."
        )
    except (FileNotFoundError, OSError) as error:
        LGR.warning(
            f"Could not start Xvfb, trying to render without a display: {error}"
        )
        process = None

    try:
        yield process
    finally:
        if process is not None:
            process.terminate()
            process.wait()
            os.environ.pop("DISPLAY", None)


def _pad_to_even(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pad the bottom and right of a frame with black so both dimensions are even."""
    # Most mp4 need even frame dimensions
    pad_height, pad_width = frame.shape[0] % 2, frame.shape[1] % 2
    if not (pad_height or pad_width):
        return frame

    return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), constant_values=0)


def _get_label_fields(label_format: str) -> set[str]:
    """Return the replacement fields of a format string."""
    fields = {
        field_name.split(".")[0].split("[")[0]
        for _, field_name, _, _ in string.Formatter().parse(label_format)
        if field_name is not None
    }
    if invalid_fields := {field for field in fields if field not in LABEL_FIELDS}:
        raise ValueError(
            f"The following fields in ``label_format`` are invalid: "
            f"{iterable_to_str(sorted(invalid_fields))}. Must be one of "
            f"{iterable_to_str(LABEL_FIELDS)}."
        )

    return fields


class _MovieFile:
    """A movie file being written, repeating each rendered frame ``n_repeats`` times."""

    def __init__(
        self,
        movie_file: Path,
        n_frames: int,
        timing_run: SurfaceRun,
        target_fps: float,
        playback_speed: float | None,
        quality: int,
        ffmpeg_kwargs: list[str] | None,
    ):
        self.movie_file = movie_file
        self.n_frames = n_frames
        fps, self.n_repeats = compute_frame_timing(
            timing_run.tr, timing_run.n_interpolated_frames, target_fps, playback_speed
        )
        actual_speed = fps / self.n_repeats * timing_run.frame_tr
        LGR.info(
            f"Creating {movie_file.name} with {n_frames} rendered frames at {fps:.2f} fps "
            f"(each frame shown {self.n_repeats}x, playback speed={actual_speed:.2f}x real time)."
        )

        self.writer = imageio.get_writer(
            str(movie_file),
            fps=fps,
            quality=quality,
            macro_block_size=1,
            output_params=ffmpeg_kwargs or [],
        )

    def append(self, frame: NDArray[np.uint8]) -> None:
        frame = _pad_to_even(frame)
        for _ in range(self.n_repeats):
            self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()
        LGR.info(f"Saved movie to: {self.movie_file}")


class MovieWriter:
    """
    Write Movies of Surface Runs.

    Arranges the runs in a grid with one row per participant and one column per task. A participant missing a task leaves a blank panel in the background color.
    For each frame index, every run is rendered once with ``renderer`` and the panel is written to each movie it belongs to:

    - ``per_subject``: one movie per participant with their tasks side by side.
    - ``per_task``: one movie per task with the participants stacked vertically.
    - ``shared``: a single movie with the full grid, so all participants and tasks play in sync.

    Parameters
    ----------
    surface_runs : :obj:`list[SurfaceRun]`
        The runs to write. Each ``(subject_id, task_name)`` pair must be unique. Participants are
        ordered by first appearance.

    renderer : :obj:`SurfaceRenderer`
        The renderer used to draw each panel. Its ``color_range`` when the writer is created is
        used for runs without their own ``color_range``.

    task_order : :obj:`list[str]` or :obj:`None`, default=None
        The order of the task columns. Tasks not listed are placed after the listed ones in order of first appearance.
        If None, tasks are ordered by first appearance.

    target_fps : :obj:`float`, default=24
        The target frame rate of the movies. When ``playback_speed`` is None, each rendered frame is one output frame at exactly this rate.

    playback_speed : :obj:`float` or :obj:`None`, default=1.0
        The speed relative to real time (e.g. 1 for real time and 10 for 10x faster than real time). If None, each rendered frame is
        one output frame at exactly ``target_fps``.

    label_format : :obj:`str` or :obj:`None`, default="{subject_name}    {task_name}    t = {frame_time:.1f} s"
        Format string of the label drawn over each panel. The valid fields are "subject_id", "subject_name" (the ``display_name`` of the run),
        "task_name", "frame_index" and "frame_time" (seconds) and format specifications (e.g. "{frame_time:.1f}") is allowed.
        If None, no label is drawn.

    label_position : :obj:`tuple[float, float]`, default=None
        The ``(x, y)`` position of the label as fractions of the panel's width and height, with ``(0, 0)`` the lower left corner
        of the panel and ``(1, 1)`` its upper right (increasing ``y`` places the label closer to the top of the panel). If None,
        label is centered in the header section if ``render`` specified ``top_spacing`` and centered between the hemispheres if
        ``render`` specified ``hemisphere_spacing``. If ``top_spacing`` and ``hemisphere_spacing`` arent specified then ``(0.02, 0.98)`` is used.

    label_anchor : :obj:`tuple[Literal["left", "center", "right"], Literal["bottom", "center", "top"]]` or :obj:`None`, default=None
        The ``(horizontal, vertical)`` point of the label text placed at ``label_position``. If None, ``("center", "center")`` is used when ``renderer``
        has a ``top_spacing`` or ``hemisphere_spacing``, else ``("left", "top")`` is used.

    task_spacing : :obj:`int`, default=0
        Width in pixels of the empty space between the task columns of a row, filled with the background color of the renderer.

    subject_spacing : :obj:`int`, default=0
        Height in pixels of the empty space between the participant rows of a movie, filled with the background color of the renderer.

    quality : :obj:`int`, default=8
        Video quality between 0 and 10 passed to ``imageio`` (higher is better and larger).

    ffmpeg_kwargs : :obj:`list[str]` or :obj:`None`, default=None
        Extra output arguments passed to ``ffmpeg`` through ``imageio`` (its ``output_params``), as a flat list of command line
        tokens (e.g. ``["-preset", "ultrafast"]``). See the `ffmpeg documentation <https://ffmpeg.org/ffmpeg.html#Options>`_ for
        the available options and the `libx264 encoder options <https://ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb>`_ for the
        default codec. Note that ``["-preset", "ultrafast", "-tune", "zerolatency"]`` can be used to reduce memory usage when working
        with larger files.

    Examples
    --------
    >>> from bidsaid.movie import MovieWriter, compute_vmax, create_surface_run, virtual_display
    >>> from bidsaid.surface import fetch_fsaverage_surfaces, get_cortex_masks
    >>> from bidsaid.visual import SurfaceRenderer
    >>> surfaces = fetch_fsaverage_surfaces("10k")
    >>> cortex_masks = get_cortex_masks(surfaces["medial"])
    >>> surface_runs = [
    ...     create_surface_run(
    ...         denoised_file,
    ...         surfaces["pial"],
    ...         surfaces["white"],
    ...         cortex_masks,
    ...         n_interpolated_frames=1,
    ...     )
    ...     for denoised_file in denoised_files
    ... ]
    >>> vmax = compute_vmax(surface_runs, threshold=1.0)
    >>> with virtual_display(), SurfaceRenderer(
    ...     *surfaces["inflated"],
    ...     sulcal_depths=surfaces["sulc"],
    ...     color_range=(-vmax, vmax),
    ...     threshold=1.0,
    ... ) as renderer:
    ...     movie_files = MovieWriter(
    ...         surface_runs, renderer, task_order=["rest", "music"]
    ...     ).write("movies", per_subject=True, per_task=True, shared=True)
    """

    def __init__(
        self,
        surface_runs: list[SurfaceRun],
        renderer: SurfaceRenderer,
        task_order: list[str] | None = None,
        target_fps: float = 24,
        playback_speed: float | None = 1.0,
        label_format: str | None = DEFAULT_LABEL_FORMAT,
        label_position: tuple[float, float] | None = None,
        label_anchor: LabelAnchor | None = None,
        task_spacing: int = 0,
        subject_spacing: int = 0,
        quality: int = 8,
        ffmpeg_kwargs: list[str] | None = None,
    ):
        if not surface_runs:
            raise ValueError("``surface_runs`` must contain at least one run.")

        if target_fps <= 0:
            raise ValueError("``target_fps`` must be greater than 0.")

        if task_spacing < 0 or subject_spacing < 0:
            raise ValueError(
                "``task_spacing`` and ``subject_spacing`` must be 0 or greater."
            )

        if not 0 <= quality <= 10:
            raise ValueError("``quality`` must be between 0 and 10.")

        if label_position is None:
            if renderer.top_spacing:
                label_position = renderer.top_spacing_center
            elif renderer.hemisphere_spacing:
                label_position = renderer.hemisphere_spacing_center
            else:
                label_position = (0.02, 0.98)

        if label_anchor is None:
            in_spacing = bool(renderer.top_spacing or renderer.hemisphere_spacing)
            label_anchor = ("center", "center") if in_spacing else ("left", "top")

        _validate_label_position(label_position, "label_position")
        _validate_anchor(label_anchor, "label_anchor")

        self.renderer = renderer
        self.default_color_range = renderer.color_range
        self.target_fps = target_fps
        self.playback_speed = playback_speed
        self.label_format = label_format
        self.label_position = tuple(label_position)
        self.label_anchor = tuple(label_anchor)
        self.task_spacing = task_spacing
        self.subject_spacing = subject_spacing
        self.quality = quality
        self.ffmpeg_kwargs = list(ffmpeg_kwargs) if ffmpeg_kwargs else None
        self._label_fields = _get_label_fields(label_format) if label_format else set()

        self.grid: dict[str, dict[str, SurfaceRun]] = {}
        for surface_run in surface_runs:
            self._validate_run(surface_run)
            tasks = self.grid.setdefault(surface_run.subject_id, {})
            if surface_run.task_name in tasks:
                raise ValueError(
                    f"Duplicate run for {_add_entity_prefix(surface_run.subject_id, 'sub')} and task "
                    f"{surface_run.task_name}."
                )

            tasks[surface_run.task_name] = surface_run

        self.subject_ids = list(self.grid)
        self.task_names = self._order_tasks(surface_runs, task_order)

    @staticmethod
    def _validate_run(surface_run: SurfaceRun) -> None:
        """Raise if the hemispheres of a run are not 2D arrays with the same number of frames."""
        if surface_run.lh.ndim != 2 or surface_run.rh.ndim != 2:
            raise ValueError(
                f"The hemispheres of {_add_entity_prefix(surface_run.subject_id, 'sub')} task {surface_run.task_name} "
                "must be 2D (n_vertices, n_frames) arrays."
            )

        if surface_run.lh.shape[1] != surface_run.rh.shape[1]:
            raise ValueError(
                f"The hemispheres of {_add_entity_prefix(surface_run.subject_id, 'sub')} task {surface_run.task_name} "
                f"have a different number of frames ({surface_run.lh.shape[1]} and "
                f"{surface_run.rh.shape[1]})."
            )

    @staticmethod
    def _order_tasks(
        surface_runs: list[SurfaceRun], task_order: list[str] | None
    ) -> list[str]:
        """Return the task columns in the requested order, then by first appearance."""
        task_names = list(dict.fromkeys(run.task_name for run in surface_runs))
        if not task_order:
            return task_names

        if unknown_tasks := [task for task in task_order if task not in task_names]:
            raise ValueError(
                "The following tasks in ``task_order`` have no surface runs: "
                f"{iterable_to_str(unknown_tasks)}."
            )

        return list(task_order) + [
            task for task in task_names if task not in task_order
        ]

    @property
    def surface_runs(self) -> list[SurfaceRun]:
        """The runs in grid order (by participant, then by task column)."""
        return [
            self.grid[subject_id][task_name]
            for subject_id in self.subject_ids
            for task_name in self.task_names
            if task_name in self.grid[subject_id]
        ]

    @property
    def n_subjects(self) -> int:
        """The number of participants (rows)."""
        return len(self.subject_ids)

    @property
    def n_tasks(self) -> int:
        """The number of tasks (columns)."""
        return len(self.task_names)

    def get_label(self, surface_run: SurfaceRun, frame_index: int) -> str | None:
        """
        Get the Label of a Panel.

        Parameters
        ----------
        surface_run : :obj:`SurfaceRun`
            The run being rendered.

        frame_index : :obj:`int`
            The index of the frame being rendered.

        Returns
        -------
        str or None
            The ``label_format`` filled in for the run and frame, or None when ``label_format``
            is None.
        """
        if self.label_format is None:
            return None

        return self.label_format.format(
            subject_id=surface_run.subject_id,
            subject_name=surface_run.display_name,
            task_name=surface_run.task_name,
            frame_index=frame_index,
            frame_time=frame_index * surface_run.frame_tr,
        )

    def render_panel(
        self, surface_run: SurfaceRun, frame_index: int
    ) -> NDArray[np.uint8]:
        """
        Render One Frame of a Run.

        Sets the ``color_range`` of the renderer to that of the run (or ``default_color_range`` when the run has none) before rendering.

        Parameters
        ----------
        surface_run : :obj:`SurfaceRun`
            The run to render.

        frame_index : :obj:`int`
            The index of the frame to render.

        Returns
        -------
        NDArray[np.uint8]
            An RGB image with shape (height, width, 3).
        """
        self.renderer.color_range = surface_run.color_range or self.default_color_range

        return self.renderer.render(
            surface_run.lh[:, frame_index],
            surface_run.rh[:, frame_index],
            label=self.get_label(surface_run, frame_index),
            label_position=self.label_position,
            label_anchor=self.label_anchor,
        )

    @staticmethod
    def _get_n_frames(surface_runs: list[SurfaceRun], description: str) -> int:
        """Return the shortest length of the runs, warning when they differ."""
        n_frames = min(run.n_frames for run in surface_runs)
        if len({run.n_frames for run in surface_runs}) > 1:
            LGR.warning(
                f"Runs of {description} differ in length, truncating the movie to {n_frames} frames."
            )

        if len({run.frame_tr for run in surface_runs}) > 1:
            LGR.warning(
                f"Runs of {description} differ in TR or ``n_interpolated_frames``, the movie uses the timing of {_add_entity_prefix(surface_runs[0].subject_id, 'sub')} "
                f"task  {surface_runs[0].task_name} so panels will play at different speeds relative to real time."
            )

        return n_frames

    def _open_movie(
        self, movie_file: Path, surface_runs: list[SurfaceRun], description: str
    ) -> _MovieFile:
        """Open a movie file for a set of runs, timed by the first run."""
        return _MovieFile(
            movie_file,
            self._get_n_frames(surface_runs, description),
            surface_runs[0],
            self.target_fps,
            self.playback_speed,
            self.quality,
            self.ffmpeg_kwargs,
        )

    def write(
        self,
        dst_dir: str | Path,
        per_subject: bool = True,
        per_task: bool = False,
        shared: bool = True,
        desc: str = "surface",
    ) -> list[Path]:
        """
        Render the Frames and Write the Movies.

        Each run is rendered once per frame index and written to every movie it belongs to. Note that movies containing runs of different lengths are truncated to the shortest run.

        Parameters
        ----------
        dst_dir : :obj:`str` or :obj:`Path`
            The destination directory of the movies. Created if it does not exist.

        per_subject : :obj:`bool`, default=True
            Write one movie per participant ("sub-{subject_id}_desc-{desc}_movie.mp4") with their tasks side by side.

        per_task : :obj:`bool`, default=False
            Write one movie per task ("task-{task_name}_desc-{desc}_movie.mp4") with the participants stacked vertically.

        shared : :obj:`bool`, default=True
            Write a single movie ("group_desc-{desc}_movie.mp4") with the full grid of participants (rows) and tasks (columns).

        desc : :obj:`str`, default="surface"
            The "desc" entity of the filenames.

        Returns
        -------
        list[Path]
            The movie files that were written.
        """
        if not (per_subject or per_task or shared):
            raise ValueError(
                "At least one of ``per_subject``, ``per_task`` or ``shared`` must be True."
            )

        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        subject_movies: dict[str, _MovieFile] = {}
        task_movies: dict[str, _MovieFile] = {}
        shared_movie = None
        try:
            if per_subject:
                for subject_id, tasks in self.grid.items():
                    subject_movies[subject_id] = self._open_movie(
                        dst_dir
                        / f"{_add_entity_prefix(subject_id, 'sub')}_desc-{desc}_movie.mp4",
                        list(tasks.values()),
                        _add_entity_prefix(subject_id, "sub"),
                    )

            if per_task:
                for task_name in self.task_names:
                    task_movies[task_name] = self._open_movie(
                        dst_dir
                        / f"{_add_entity_prefix(task_name, 'task')}_desc-{desc}_movie.mp4",
                        [
                            tasks[task_name]
                            for tasks in self.grid.values()
                            if task_name in tasks
                        ],
                        f"task {task_name}",
                    )

            if shared and len(self.surface_runs) > 1:
                shared_movie = self._open_movie(
                    dst_dir / f"group_desc-{desc}_movie.mp4",
                    self.surface_runs,
                    "all participants",
                )
            elif shared:
                LGR.warning("Only one run available, skipping the shared movie.")

            self._write_frames(subject_movies, task_movies, shared_movie)
        finally:
            self.renderer.color_range = self.default_color_range
            movies = [*subject_movies.values(), *task_movies.values()]

            if shared_movie:
                movies.append(shared_movie)

            for movie in movies:
                movie.close()

        return [movie.movie_file for movie in movies]

    def _write_frames(
        self,
        subject_movies: dict[str, _MovieFile],
        task_movies: dict[str, _MovieFile],
        shared_movie: _MovieFile | None,
    ) -> None:
        """Render each run once per frame index and append the panels to the open movies."""
        movies = [*subject_movies.values(), *task_movies.values(), shared_movie]
        n_frames = max(
            (movie.n_frames for movie in movies if movie is not None), default=0
        )
        if not n_frames:
            LGR.warning("No movies to write.")
            return

        blank_panel = np.full_like(self.renderer.render(), self.renderer.background_rgb)
        for frame_index in tqdm(range(n_frames), desc="Rendering frames", unit="frame"):
            panels: dict[tuple[str, str], NDArray[np.uint8]] = {}
            for subject_id, tasks in self.grid.items():
                for task_name, surface_run in tasks.items():
                    if frame_index < self._n_frames_needed(
                        subject_id, task_name, subject_movies, task_movies, shared_movie
                    ):
                        panels[(subject_id, task_name)] = self.render_panel(
                            surface_run, frame_index
                        )

            for subject_id, movie in subject_movies.items():
                if frame_index < movie.n_frames:
                    movie.append(self._get_row(panels, subject_id, blank_panel))

            for task_name, movie in task_movies.items():
                if frame_index < movie.n_frames:
                    movie.append(
                        self._stack(
                            [
                                panels[(subject_id, task_name)]
                                for subject_id in self.subject_ids
                                if (subject_id, task_name) in panels
                            ],
                            axis=0,
                        )
                    )

            if shared_movie and frame_index < shared_movie.n_frames:
                shared_movie.append(
                    self._stack(
                        [
                            self._get_row(panels, subject_id, blank_panel)
                            for subject_id in self.subject_ids
                        ],
                        axis=0,
                    )
                )

    def _n_frames_needed(
        self,
        subject_id: str,
        task_name: str,
        subject_movies: dict[str, _MovieFile],
        task_movies: dict[str, _MovieFile],
        shared_movie: _MovieFile | None,
    ) -> int:
        """Return the number of frames of a run that are written to at least one movie."""
        n_frames = [
            movie.n_frames
            for movie in (
                subject_movies.get(subject_id),
                task_movies.get(task_name),
                shared_movie,
            )
            if movie is not None
        ]

        return max(n_frames, default=0)

    def _get_row(
        self,
        panels: dict[tuple[str, str], NDArray[np.uint8]],
        subject_id: str,
        blank_panel: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Place the panels of a participant side by side in task order, blank when missing."""
        return self._stack(
            [
                panels.get((subject_id, task_name), blank_panel)
                for task_name in self.task_names
            ],
            axis=1,
        )

    def _stack(
        self, images: list[NDArray[np.uint8]], axis: Literal[0, 1]
    ) -> NDArray[np.uint8]:
        """
        Stack images vertically (``axis=0``, rows) or horizontally (``axis=1``, columns).

        ``subject_spacing`` or ``task_spacing`` pixels of the renderer's background color are inserted between consecutive images.
        """
        spacing = self.subject_spacing if axis == 0 else self.task_spacing
        if spacing and len(images) > 1:
            shape = list(images[0].shape)
            shape[axis] = spacing
            spacer = np.full(shape, self.renderer.background_rgb, dtype=np.uint8)
            images = [
                image for pair in zip(images, [spacer] * len(images)) for image in pair
            ][:-1]

        return np.concatenate(images, axis=axis)


def mp4_to_gif(
    mp4_file: str | Path,
    gif_file: str | Path | None = None,
    every_n_frames: int = 1,
    **pillow_kwargs: Any,
) -> Path:
    """
    Convert a Movie to a GIF.

    Reads every ``every_n_frames``-th frame of the movie and writes them as an animated GIF with Pillow. Unless a timing is given in ``pillow_kwargs``,
    the GIF plays at the speed of the movie (each kept frame lasts ``every_n_frames / fps`` seconds).

    Parameters
    ----------
    mp4_file : :obj:`str` or :obj:`Path`
        Path to the mp4 movie.

    gif_file : :obj:`str`, :obj:`Path`, or :obj:`None`, default=None
        Path of the GIF to write. If None, the movie's path with a ".gif" extension is used.

    every_n_frames : :obj:`int`, default=1
        Keep one frame out of every ``every_n_frames`` frames of the movie (1 keeps every frame).

    **pillow_kwargs : :obj:`Any`
        Keyword arguments passed to ``imageio.mimsave`` for the Pillow GIF writer, such as  ``duration`` (milliseconds per frame) or ``loop`` (0 to loop forever, the default). See the
        `Pillow GIF documentation <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif>`_ for the available options.

    Returns
    -------
    Path
        The GIF file that was written.
    """
    mp4_file = Path(mp4_file)
    if (ext := mp4_file.suffix) != ".mp4":
        raise ValueError(f"``mp4_file`` does not have an '.mp4' extension: '{ext}'")

    if every_n_frames < 1:
        raise ValueError("``every_n_frames`` must be 1 or greater.")

    gif_file = Path(gif_file) if gif_file is not None else mp4_file.with_suffix(".gif")

    with imageio.get_reader(str(mp4_file)) as reader:
        if not {"duration", "fps"} & set(pillow_kwargs):
            pillow_kwargs["duration"] = (
                every_n_frames / reader.get_meta_data()["fps"] * 1000
            )

        frames = [
            frame for index, frame in enumerate(reader) if index % every_n_frames == 0
        ]

    imageio.mimsave(str(gif_file), frames, **pillow_kwargs)
    LGR.info(f"Saved GIF with {len(frames)} frames to: {gif_file}")

    return gif_file


__all__ = [
    "DEFAULT_LABEL_FORMAT",
    "LABEL_FIELDS",
    "MovieWriter",
    "SurfaceRun",
    "compute_frame_timing",
    "compute_vmax",
    "create_surface_run",
    "mp4_to_gif",
    "virtual_display",
]

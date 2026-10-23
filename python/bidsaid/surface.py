"""
Module for projecting volumetric NIfTI data onto cortical surfaces.

.. important::
    ``fetch_fsaverage_surfaces`` requires ``pip install bidsaid[surface]``.

"""

from pathlib import Path
from typing import Literal, TypeAlias

import nibabel as nib, numpy as np
from nilearn.image import smooth_img
from nilearn.surface import vol_to_surf
from numpy.typing import NDArray

from bidsaid.io import load_nifti, NiftiFileorImage
from bidsaid.logging import setup_logger

LGR = setup_logger(__name__)

SurfaceMesh: TypeAlias = str | Path | tuple[NDArray, NDArray]
"""A path to a GIFTI surface file or a ``(coordinates, faces)`` tuple of arrays."""

_FSAVERAGE_KEYS = ("pial", "white", "inflated", "sulc", "medial")


# https://mvpa.blogspot.com/2025/01/intro-to-working-with-volume-and.html
def fetch_fsaverage_surfaces(
    density: Literal["3k", "10k", "41k", "164k"] = "10k",
    data_dir: str | Path | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Fetch fsaverage Surface Files.

    Downloads (or loads from cache) the fsaverage surfaces distributed by ``neuromaps``.

    Parameters
    ----------
    density : :obj:`Literal["3k", "10k", "41k", "164k"]`, default="10k"
        Vertex density of the surface. "10k" corresponds to fsaverage5.

    data_dir : :obj:`str`, :obj:`Path`, or :obj:`None`, default=None
        Directory where the surfaces are cached. If None, the ``neuromaps`` default is used.

    Returns
    -------
    dict[str, tuple[str, str]]
        A dictionary with the keys "pial", "white", "inflated", "sulc", and "medial". Each value is a ``(left, right)``
        tuple of file paths. The "medial" files are "desc-nomedialwall" labels where True indicates cortex (see :func:`get_cortex_masks`).

    .. important::
        Requires ``neuromaps``, which is installed with ``pip install bidsaid[surface]``.
    """
    try:
        from neuromaps.datasets import fetch_fsaverage
    except ImportError as error:
        raise ImportError(
            "``neuromaps`` is required to fetch fsaverage surfaces. Install it with "
            "``pip install bidsaid[surface]``."
        ) from error

    surfaces = fetch_fsaverage(density=density, data_dir=data_dir)

    return {
        key: tuple(str(surface_file) for surface_file in surfaces[key])
        for key in _FSAVERAGE_KEYS
    }


def load_surface_mesh(surface_mesh: SurfaceMesh) -> tuple[NDArray, NDArray]:
    """
    Load a Surface Mesh.

    Parameters
    ----------
    surface_mesh : :obj:`str`, :obj:`Path`, or :obj:`tuple[NDArray, NDArray]`
        Path to a GIFTI surface file (".surf.gii") or a ``(coordinates, faces)`` tuple.

    Returns
    -------
    tuple[NDArray, NDArray]
        The vertex coordinates (n_vertices x 3) and triangular faces (n_faces x 3).
    """
    if isinstance(surface_mesh, (str, Path)):
        coordinates, faces = nib.load(surface_mesh).agg_data(("pointset", "triangle"))
    else:
        coordinates, faces = surface_mesh

    return np.asarray(coordinates), np.asarray(faces)


def load_surface_data(surface_data: str | Path) -> NDArray:
    """
    Load Per-Vertex Surface Data.

    Parameters
    ----------
    surface_data : :obj:`str`, :obj:`Path`, or :obj:`NDArray`
        Path to a GIFTI data file (e.g. a ".shape.gii" sulcal depth map or a ".label.gii" file).

    Returns
    -------
    NDArray
        A 1D array with one value per vertex.
    """
    if isinstance(surface_data, (str, Path)):
        return np.asarray(nib.load(surface_data).agg_data())

    return np.asarray(surface_data)


def get_cortex_masks(medial_wall_files: tuple[str, str]) -> list[NDArray[np.bool_]]:
    """
    Get Cortex Masks From Medial Wall Labels.

    Parameters
    ----------
    medial_wall_files : :obj:`tuple[str, str]`
        The ``(left, right)`` "desc-nomedialwall" label files returned under the "medial" key of
        :func:`fetch_fsaverage_surfaces`.

    Returns
    -------
    list[NDArray[np.bool_]]
        A list of ``[left, right]`` boolean arrays where True indicates a cortical vertex
        and False indicates the medial wall.
    """
    return [
        load_surface_data(medial_wall_file).astype(bool)
        for medial_wall_file in medial_wall_files
    ]


def project_to_surface(
    nifti_file_or_img: NiftiFileorImage,
    pial_surfaces: tuple[SurfaceMesh, SurfaceMesh],
    white_surfaces: tuple[SurfaceMesh, SurfaceMesh] | None = None,
    cortex_masks: list[NDArray[np.bool_]] | None = None,
    fwhm: float | int | None = None,
    max_volumes: int | None = None,
    interpolation: Literal["linear", "nearest"] = "linear",
    radius: float = 3.0,
    dtype: type = np.float32,
) -> list[NDArray[np.floating]]:
    """
    Project a Volumetric Image onto Cortical Surfaces.

    Samples the image at each vertex of the pial surface with :func:`nilearn.surface.vol_to_surf`. When ``white_surfaces``
    is given, it is passed as the inner mesh.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the functional NIfTI file or a functional NIfTI image (3D or 4D).

    pial_surfaces : :obj:`tuple[SurfaceMesh, SurfaceMesh]`
        The ``(left, right)`` pial surfaces, as GIFTI file paths or  ``(coordinates, faces)`` tuples.

    white_surfaces : :obj:`tuple[SurfaceMesh, SurfaceMesh]` or :obj:`None`, default=None
        The ``(left, right)`` white matter surfaces used as the inner mesh. Must have the same number of vertices as ``pial_surfaces``.

    cortex_masks : :obj:`list[NDArray[np.bool_]]` or :obj:`None`, default=None
        The ``[left, right]`` cortex masks from :func:`get_cortex_masks`. Vertices outside the mask (the medial wall) are set to 0.

    fwhm : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Full width at half maximum (mm) of the Gaussian smoothing applied to the image before projection.

    max_volumes : :obj:`int` or :obj:`None`, default=None
        Only project the first N volumes of a 4D image

    interpolation : :obj:`Literal["linear", "nearest"]`, default="linear"
        Interpolation used when sampling the image.

    radius : :obj:`float`, default=3.0
        Radius (mm) of the sampling sphere around each vertex. Ignored when ``white_surfaces`` is given.

    dtype : :obj:`type`, default=np.float32
        Data type of the returned arrays.

    Returns
    -------
    list[NDArray[np.floating]]
        A list of ``[left, right]`` arrays. Each has shape (n_vertices, n_volumes) for a 4D image or (n_vertices,) for a 3D image.
        Also, NaNs are replaced with 0.
    """
    nifti_img = load_nifti(nifti_file_or_img)
    if max_volumes and nifti_img.ndim == 4:
        nifti_img = nifti_img.slicer[..., :max_volumes]

    if fwhm:
        nifti_img = smooth_img(nifti_img, fwhm)

    hemispheres = []
    for hemisphere_index in range(2):
        pial_surface = load_surface_mesh(pial_surfaces[hemisphere_index])
        white_surface = (
            load_surface_mesh(white_surfaces[hemisphere_index])
            if white_surfaces
            else None
        )
        surface_data = vol_to_surf(
            nifti_img,
            pial_surface,
            inner_mesh=white_surface,
            interpolation=interpolation,
            radius=radius,
        )
        surface_data = np.nan_to_num(surface_data).astype(dtype, copy=False)

        if cortex_masks is not None:
            surface_data[~cortex_masks[hemisphere_index]] = 0

        hemispheres.append(surface_data)

    return hemispheres


__all__ = [
    "fetch_fsaverage_surfaces",
    "load_surface_mesh",
    "load_surface_data",
    "get_cortex_masks",
    "project_to_surface",
]

"""Module for input/output operations."""

import gzip, shutil, re, math
from pathlib import Path
from typing import Iterator

import nibabel as nib, numpy as np
from numpy.typing import NDArray

from bidsaid._helpers import is_path
from bidsaid.logging import setup_logger
from bidsaid._rust import compress_file

LGR = setup_logger(__name__)


def load_nifti(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
) -> nib.nifti1.Nifti1Image:
    """
    Loads a NIfTI image.

    Loads NIfTI image when not a ``Nifti1Image`` object or
    returns the image if already loaded in.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    Returns
    -------
    Nifti1Image
        The loaded in NIfTI image.
    """
    nifti_img = (
        nifti_file_or_img
        if isinstance(nifti_file_or_img, nib.nifti1.Nifti1Image)
        else nib.load(nifti_file_or_img)
    )

    return nifti_img


def compress_image(
    nifti_file: str | Path,
    dst_dir: str | Path | None = None,
    remove_src_file: bool = False,
    compression_level: int = 6,
) -> Path:
    """
    Compresses a ".nii" image to a ".nii.gz" image.

    Parameters
    ----------
    nifti_file : :obj:`str` or :obj:`Path`
        Path to the NIfTI image.

    dst_dir : :obj:`str` or :obj:`Path`, default=None
        Destination directory for the NIfTI image. If None, image is saved in the
        source directory.

    remove_src_file : :obj:`bool`, default=False
        Deletes the original source image file.

    compression_level : :obj:`int`, default=6
        Level of compression from 0 to 9. Higher levels are slower but
        produce smaller file sizes.

    Returns
    -------
    Path
        Path to compressed file.
    """
    nifti_file = Path(nifti_file)
    dst_dir = Path(dst_dir) if dst_dir else nifti_file.parent

    dst_file = dst_dir / str(nifti_file.name).replace(".nii", ".nii.gz")

    compress_file(str(nifti_file), str(dst_file), compression_level, remove_src_file)

    return dst_file


def regex_glob(
    src_dir: str | Path, pattern: str, recursive: bool = False
) -> Iterator[Path]:
    """
    Use regex to get content in the source directory with specific patterns.

    Parameters
    ----------
    src_dir : :obj:`str` or :obj:`Path`
        The source directory.

    pattern : :obj:`str`
        The regex pattern.

    recursive : :obj:`bool`, default=False
        If True, regex pattern is applied to content in the top-level directory
        (i.e., sub-101.log) and nested directories (i.e. logs/sub-101.log). If
        False, regex pattern is only applied to content in the top-level directory.

    Yields
    ------
    Path
        Paths in the directory whose names match ``pattern``.

    Example
    -------
    >>> from bidsaid.io import regex_glob
    >>> # Get any file ending in pdf or txt
    >>> regex_glob(r"path/to/directory", pattern=r"^.*.(pdf|txt)$")
    """
    all_contents = Path(src_dir).rglob("*") if recursive else Path(src_dir).glob("*")

    compiled = re.compile(pattern)
    for path in all_contents:
        if compiled.match(path.name):
            yield path


def get_nifti_header(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
) -> nib.nifti1.Nifti1Header:
    """
    Get header from a NIfTI image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    Returns
    -------
    Nifti1Header
        The header from a NIfTI image.
    """
    return load_nifti(nifti_file_or_img).header


def compute_uncompressed_image_size(nifti_file: str | Path) -> int:
    """
    Compute Uncompressed NiFTI Image Size

    Parameters
    ----------
    nifti_file: :obj:`str` or :obj:`Path`
        Path to the NIfTI file or a NIfTI image.

    Return
    ------
    The total number of bytes in the image.
    """
    nifti_file = Path(nifti_file)

    hdr = get_nifti_header(nifti_file)

    if nifti_file.suffix.lower() != ".gz":
        return nifti_file.stat().st_size

    total_bytes = 0
    with gzip.open(nifti_file, "rb") as f:
        while chunk := f.read(1024 * 1024):
            total_bytes += len(chunk)

        return total_bytes - int(hdr.get_data_offset())


def is_nifti_truncated(nifti_file: str | Path) -> bool:
    """
    Checks expected byte size and actual byte size of uncompressed NIfTI images.

    Parameters
    ----------
    nifti_file : :obj:`str`or :obj:`Path`
        Path to the NIfTI file.

    Returns
    -------
    bool
        True if the NIfTI image is truncated (expected byte size < actual byte size)
        and False otherwise.
    """
    if not is_path(nifti_file):
        raise ValueError("`nifti_file` must be a Path or string.")

    nifti_file = Path(nifti_file)
    if nifti_file.name.endswith(".gz"):
        raise ValueError("Only uncompressed files are allowed.")

    hdr = get_nifti_header(nifti_file)
    n_voxels = int(np.prod(hdr.get_data_shape()))
    bytes_per_voxel = hdr.get_data_dtype().itemsize
    expected_data_bytes = n_voxels * bytes_per_voxel

    actual_data_bytes = compute_uncompressed_image_size(nifti_file)

    if actual_data_bytes < expected_data_bytes:
        LGR.warning(
            f"Truncated NIfTI: {nifti_file} header expects {expected_data_bytes} data bytes "
            f"({n_voxels} voxels * {bytes_per_voxel} bytes) but only {actual_data_bytes} are present."
        )
        return True
    else:
        return False


def get_nifti_affine(nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image) -> NDArray:
    """
    Get the affine matrix from a NIfTI image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    Returns
    -------
    NDArray
        The affine matrix from a NIfTI image.
    """
    return load_nifti(nifti_file_or_img).affine


def _copy_file(
    src_file: str | Path, dst_file: str | Path, remove_src_file: bool = False
) -> None:
    """
    Copy a file and optionally remove the source file.

    Parameters
    ----------
    src_file : :obj:`str` or :obj:`Path`
        The source file to be copied.

    dst_file : :obj:`str` or :obj:`Path`
        The new destination file.

    remove_src_file : :obj:`bool`, default=False
        Delete the source file if True.
    """
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(src_file, dst_file)

    if remove_src_file:
        Path(src_file).unlink(missing_ok=True)


def replace_ext(filename: str | Path, new_ext: str) -> Path:
    """
    Replaces extension of a filename.

    Parameters
    ----------
    filename : :obj:`str` or :obj:`Path`
        The path to the file.

    new_ext : :obj:`str`
        The new extension.

    Returns
    -------
    Path
        Filename with new extension.

    Example
    --------
    >>> replace_ext("file.nii.gz", ".json")
        "file.json"
    """
    filename = Path(filename)
    old_ext = "".join(filename.suffixes)
    new_ext = f".{new_ext}" if not new_ext.startswith(".") else new_ext

    return Path(str(filename).replace(old_ext, new_ext))


def truncate_nifti_to_complete_volumes(
    nifti_file: str | Path, new_nifti_filename: str | Path | None = None
):
    """
    Truncate NiFTI File to Complete Volumes

    Computes the total number of bytes available in the file and truncates the
    file size based on the following equation:

    ```
        bytes_per_volume = math.prod(img.shape[:3]) * img.get_data_dtype().itemsize
        n_volumes_retained = min(img.shape[3], available_bytes // bytes_per_volume)
    ```

    Parameters
    ----------
    nifti_file: :obj:`str` or :obj:`Path`
        Path to the NIfTI file or a NIfTI image.

    new_nifti_filename: :obj:`str` or :obj:`Path`, default = None
        The new filename for to save the truncated NIfTI image. If None, then the original NIfTI file
        is overwritten with the truncated one.
    """
    nifti_img = nib.load(nifti_file)

    available_bytes = compute_uncompressed_image_size(nifti_file)
    bytes_per_volume = (
        math.prod(nifti_img.shape[:3]) * nifti_img.get_data_dtype().itemsize
    )
    n_volumes_retained = min(nifti_img.shape[3], available_bytes // bytes_per_volume)

    if n_volumes_retained < nifti_img.shape[3]:
        LGR.warning(f"Percentage of volumes kept: {n_volumes_retained}")

        truncated_image = nifti_img.slicer[..., :n_volumes_retained]

        if new_nifti_filename:
            new_nifti_filename = Path(new_nifti_filename)
            new_nifti_filename.parent.mkdir(parents=True, exist_ok=True)

        nifti_filename = new_nifti_filename if new_nifti_filename else nifti_file

        nib.save(truncated_image, nifti_filename)


__all__ = [
    "load_nifti",
    "compress_image",
    "is_nifti_truncated",
    "regex_glob",
    "get_nifti_header",
    "get_nifti_affine",
    "replace_ext",
    "compute_uncompressed_image_size",
    "truncate_nifti_to_complete_volumes",
]

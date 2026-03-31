"""Module for input/output operations."""

import shutil, re
from pathlib import Path
from typing import Iterator

import nibabel as nib
from numpy.typing import NDArray


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
    return_dst_file: bool = False,
) -> Path | None:
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

    return_dst_file : :obj:`bool`, default=False
        Return the path to the compressed file.

    Returns
    -------
    Path or None
        Path to compressed file if ``return_dst_file`` is True else None.
    """
    img = nib.load(nifti_file)

    nifti_file = Path(nifti_file)
    dst_dir = Path(dst_dir) if dst_dir else nifti_file.parent

    dst_file = dst_dir / str(nifti_file.name).replace(".nii", ".nii.gz")
    nib.save(img, dst_file)

    if remove_src_file:
        nifti_file.unlink()

    return dst_file if return_dst_file else None


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
    >>> from nifti2bids.io import regex_glob
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
    if not dst_file.parent.exists():
        dst_file.parent.mkdir(parents=True)

    shutil.copy(src_file, dst_file)

    if remove_src_file:
        Path(src_file).unlink()


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


__all__ = [
    "load_nifti",
    "compress_image",
    "regex_glob",
    "get_nifti_header",
    "get_nifti_affine",
    "replace_ext",
]

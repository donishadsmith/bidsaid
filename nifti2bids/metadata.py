"""Utility functions to extract or create metadata."""

import datetime, re, sys
from pathlib import Path
from typing import Any, Iterable, Literal

import nibabel as nib, numpy as np

from ._exceptions import SliceAxisError, DataDimensionError
from ._decorators import check_all_none, check_nifti
from .io import load_nifti, get_nifti_header, get_nifti_affine
from .logging import setup_logger

LGR = setup_logger(__name__)

_VOXEL_INDX_DICT = {"i": 0, "j": 1, "k": 2}


def needs_resampling(
    source_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    target_file_or_img: str | Path | nib.nifti1.Nifti1Image,
) -> bool:
    """
    Determines if a source image needs to be resampled.

    Parameters
    ----------
    source_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        The source image that may need to be resampled.

    target_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        The target image that it may need to be resampled to.

    Returns
    -------
    bool
        True if resampling is needed and False if no resampling is needed.

    """
    source_img = load_nifti(source_file_or_img)
    target_img = load_nifti(target_file_or_img)

    equal_shape = source_img.shape == target_img.shape
    equal_affine = np.allclose(source_img.affine, target_img.affine, atol=1e-5)

    return False if (equal_shape and equal_affine) else True


@check_all_none(parameter_names=["nifti_file_or_img", "nifti_header"])
def determine_slice_axis(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image | None = None,
    nifti_header: nib.nifti1.Nifti1Header | None = None,
) -> int:
    """
    Determine the slice axis.

    Uses "slice_end" plus one to determine the likely slice axis.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`, default=None
        Path to the NIfTI file or a NIfTI image. Must be specified
        if ``nifti_header`` is None.

    nifti_header : :obj:`Nifti1Header`, default=None
        Path to the NIfTI file or a NIfTI image. Must be specified
        if ``nifti_file_or_img`` is None.

    Returns
    -------
    int
        A number representing the slice axis.
    """
    kwargs = {"nifti_file_or_img": nifti_file_or_img, "nifti_header": nifti_header}
    slice_end, hdr = get_hdr_metadata(
        **kwargs, metadata_name="slice_end", return_header=True
    )
    if not slice_end or np.isnan(slice_end):
        raise ValueError("'slice_end' metadata field not set.")

    n_slices = int(slice_end) + 1
    dims = np.array(hdr.get_data_shape()[:3])
    slice_axis_arr = np.where(dims == n_slices)[0]
    if slice_axis_arr.size == 0:
        raise ValueError("Slice axis could not be determined.")

    return slice_axis_arr[0]


def _is_numeric(value: Any) -> bool:
    """
    Check if value is a number.
    """
    return isinstance(value, (float, int))


def _to_native_numeric(value: np.floating | np.int_) -> float | int:
    """
    Ensures numpy floats and integers are converted
    to regular Python floats and integers.
    """
    return float(value) if isinstance(value, np.floating) else int(value)


@check_all_none(parameter_names=["nifti_file_or_img", "nifti_header"])
def get_hdr_metadata(
    metadata_name: str,
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image | None = None,
    nifti_header: nib.nifti1.Nifti1Header | None = None,
    return_header: bool = False,
) -> Any | tuple[Any, nib.nifti1.Nifti1Header]:
    """
    Get metadata from a NIfTI header.

    Parameters
    ----------
    metadata_name : :obj:`str`
        Name of the metadata field to return.

    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`, default=None
        Path to the NIfTI file or a NIfTI image. Must be specified
        if ``nifti_header`` is None.

    nifti_header : :obj:`Nifti1Header`, default=None
        Path to the NIfTI file or a NIfTI image. Must be specified
        if ``nifti_file_or_img`` is None.

    return_header : :obj:`bool`
        Returns the NIfTI header

    Returns
    -------
    Any or tuple[Any, Nifti1Header]
        If ``return_header`` is False, only returns the associated
        value of the metadata. If ``return_header`` is True returns
        a tuple containing the assoicated value of the metadata
        and the NIfTI header.
    """
    hdr = nifti_header if nifti_header else get_nifti_header(nifti_file_or_img)
    metadata_value = hdr.get(metadata_name)
    metadata_value = (
        _to_native_numeric(metadata_value)
        if _is_numeric(metadata_value)
        else metadata_value
    )

    return metadata_value if not return_header else (metadata_value, hdr)


def get_n_volumes(nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image) -> int:
    """
    Get the number of volumes from a 4D NIftI image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    Returns
    -------
    int
        The number of volumes in img.
    """
    img = load_nifti(nifti_file_or_img)

    if is_3d_img(img):
        raise DataDimensionError("Image is 3D not 4D.")

    return img.get_fdata().shape[-1]


def get_image_orientation(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
) -> tuple[dict[str, str], tuple[str, str, str]]:
    """
    Get the image orientation.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`, default=None
        Path to the NIfTI file or a NIfTI image.

    Returns
    -------
    tuple[dict[str, str], tuple[str, str, str]]:
        A tuple consisting of a dictionary mapping the voxel dimension
        to its starting and ending anatomical location (from 0 to N)
        and a tuple of the image orientation.

        .. note::
           The reverse direction (N to 0) is just {voxel dimension}-
           (e.g. "i-" in RAS orientation is "R -> L").

    Examples
    --------
    >>> from nifti2bids.simulate import simulate_nifti_image
    >>> from nifti2bids.metadata import get_image_orientation
    >>> img = simulate_nifti_image((10, 10, 10, 10))
    >>> get_image_orientation(img)
        ({"i": "L -> R", "j": "P -> A", "k": "I -> S"}, ("R", "A", "S"))

    References
    ----------
    Weber, D. (n.d.). MRI orientation notes. https://eeg.sourceforge.net/mri_orientation_notes.html

    Orientation and Voxel-Order Terminology: RAS, LAS, LPI, RPI, XYZ and All That. (n.d.).
    Www.grahamwideman.com. http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    """
    orientation_list = [
        np.array(("L", "R")),
        np.array(("A", "P")),
        np.array(("S", "I")),
    ]
    affine = get_nifti_affine(nifti_file_or_img)
    orientation = nib.orientations.aff2axcodes(affine)

    orientation_dict = {}
    for voxel_dim, axis_end in zip(("i", "j", "k"), orientation):
        for axis in orientation_list:
            if axis_end in axis:
                axis_start = str(axis[axis != axis_end][0])
                orientation_dict[voxel_dim] = f"{axis_start} -> {axis_end}"

    return orientation_dict, orientation


def direction_to_voxel_axis(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    anatomical_directions: Iterable[str],
) -> tuple[str, int]:
    """
    Returns the voxel coordinate axis (and position) of a given anatomical direction.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`, default=None
        Path to the NIfTI file or a NIfTI image.

    anatomical_directions : :obj:`Iterable[str]`
        The anatomical directions.

    Returns
    -------
    tuple(str, int)
        A tuple of the axis and its position (axis, position).

    Example
    -------
    Assuming image has RAS orientation
    >>> from nifti2bids.simulate import simulate_nifti_image
    >>> nifti_img = simulate_nifti_image(img_shape=(5, 5, 5, 5))
    >>> direction_to_voxel_axis(nifti_img, ("A", "P"))
        ("j", 2)
    """
    _, orientation = get_image_orientation(nifti_file_or_img)

    arr = np.array(orientation)
    voxel_index = int(np.where(np.isin(arr, anatomical_directions))[0][0])

    voxel_pos_to_axis = {item: key for key, item in _VOXEL_INDX_DICT.items()}

    return voxel_pos_to_axis[voxel_index], voxel_index


def get_n_slices(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    slice_axis: Literal["i", "j", "k"] | None = None,
    verbose: bool = False,
) -> int:
    """
    Gets the number of slices from the header of a NIfTI image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    slice_axis : :obj:`Literal["i", "j", "k"]` or :obj:`None`, default=None
        Axis the image slices were collected in. If None,
        determines the slice axis using metadata ("slice_end")
        from the NIfTI header.

    verbose : :obj:`bool`, default=False
        Logs information.

    Returns
    -------
    int
        The number of slices.
    """
    hdr = get_nifti_header(nifti_file_or_img)
    if slice_axis:
        n_slices = hdr.get_data_shape()[_VOXEL_INDX_DICT[slice_axis]]
        if slice_end := get_hdr_metadata(nifti_header=hdr, metadata_name="slice_end"):
            if not np.isnan(slice_end) and n_slices != slice_end + 1:
                raise SliceAxisError(slice_axis, n_slices, slice_end)

        slice_dim_indx = _VOXEL_INDX_DICT[slice_axis]
    else:
        slice_dim_indx = determine_slice_axis(nifti_header=hdr)

    reversed_slice_dim_map = {v: k for v, k in _VOXEL_INDX_DICT.items()}

    n_slices = hdr.get_data_shape()[slice_dim_indx]

    if verbose:
        LGR.info(
            f"Number of slices based on "
            f"{reversed_slice_dim_map.get(slice_dim_indx)}: {n_slices}"
        )

    return _to_native_numeric(n_slices)


def get_tr(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image, verbose: bool = False
) -> float:
    """
    Get the repetition time from the header of a NIfTI image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    verbose : :obj:`bool`, default=False
        Logs "INFO" level information if True.

    Returns
    -------
    float
        The repetition time.
    """
    hdr = get_nifti_header(nifti_file_or_img)

    if not (tr := hdr.get_zooms()[3]):
        raise ValueError(f"Suspicious repetition time: {tr}.")

    if verbose:
        LGR.info(f"Repetition Time: {tr}.")

    return round(_to_native_numeric(tr), 2)


def _flip_slice_order(slice_order: list[int], ascending: bool) -> list[int]:
    """
    Flip slice index order.

    Parameters
    ----------
    slice_order : :obj:`list[int]`
        List containing integer values representing the slices.

    ascending : :obj:`bool`, default=True
        If slices were collected in ascending order (True) or descending
        order (False).

    Returns
    -------
    list[int]
        The order of the slices.
    """
    return np.flip(slice_order) if not ascending else slice_order


def _create_sequential_order(n_slices: int, ascending: bool = True) -> list[int]:
    """
    Create index ordering for sequential acquisition method.

    Parameters
    ----------
    n_slices : :obj:`int`
        The number of slices.

    ascending : :obj:`bool`, default=True
        If slices were collected in ascending order (True) or descending
        order (False).

    Returns
    -------
    list[int]
        The order of the slices.
    """
    slice_order = list(range(0, n_slices))

    return _flip_slice_order(slice_order, ascending)


def _create_interleaved_order(
    n_slices: int,
    ascending: bool = True,
    interleaved_pattern: Literal["even", "odd", "philips"] = "odd",
) -> list[int]:
    """
    Create index ordering for interleaved acquisition method.

    .. note:: Equivalent to Philips default order.

    Parameters
    ----------
    n_slices : :obj:`int`
        The number of slices.

    ascending : :obj:`bool`, default=True
        If slices were collected in ascending order (True) or descending
        order (False).

    interleaved_pattern : :obj:`Literal["even", "odd", "philips"]`, default="odd"
        If slices for interleaved acquisition were collected by acquiring the
        "even" or "odd" slices first. For "philips" (the interleaved implementation
        by Philips'), slices are acquired by a step factor equivalent to the rounded
        square root of the total slices.
        mode).

        .. important::
           Philips' "default" mode is equivalent to "interleave" with the pattern
           set to "odd", and ascending set to True.

    Returns
    -------
    list[int]
        The order of the slices.
    """
    if interleaved_pattern not in ["even", "odd", "philips"]:
        raise ValueError(
            "``interleaved_start`` must be either 'even', 'odd', or 'philips'."
        )

    if interleaved_pattern == "odd":
        slice_order = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
    elif interleaved_pattern == "even":
        slice_order = list(range(1, n_slices, 2)) + list(range(0, n_slices, 2))
    else:
        step = round(np.sqrt(n_slices))
        slice_order = [
            slice_indx
            for base in range(step)
            for slice_indx in range(base, n_slices, step)
        ]

    return _flip_slice_order(slice_order, ascending)


def _create_central_order(n_slices: int) -> list[int]:
    """
    Create central slice order.

    Defined as the first slice being the ceiling of the midpoint
    followed by a ping pong to the final slice.

    Parameters
    ----------
    n_slices : :obj:`int`
        The number of slices.

    Returns
    -------
    list[int]
        The order of the slices.
    """
    slice_range = list(range(n_slices))
    # Slices start with 0 so just get max val
    midpoint = int(np.ceil(np.max(slice_range) / 2))
    # Get the number of unique ping pong groupings
    n_ping_pongs = int(np.ceil((len(slice_range) - 1) / 2))

    slice_order = [midpoint]
    for i in range(1, (n_ping_pongs + 1)):
        slice_order.extend([midpoint - i, midpoint + i])

    slice_order = np.array(slice_order)
    slice_order = slice_order[slice_order < len(slice_range)]

    return slice_order.tolist()


def _create_reversed_central_order(n_slices: int) -> list[int]:
    """
    Create reversed central slice order.

    Defined as starting at the outer slice (starting at 0), then
    followed by a ping pong order to the middle slice (e.g.
    [0, 4, 1, 3, 2] or [0, 5, 1, 4, 2, 3])

    Parameters
    ----------
    n_slices : :obj:`int`
        The number of slices.

    Returns
    -------
    list[int]
        The order of the slices.
    """
    max_steps = n_slices - 1
    multipliers = [1, -1] * int(np.ceil(max_steps / 2))
    multipliers = multipliers[:max_steps]

    slice_order = [0]
    for mult_indx, i in enumerate(range(max_steps, 0, -1)):
        curr_val = slice_order[-1]
        slice_order.append(curr_val + i * multipliers[mult_indx])

    return slice_order


def _create_singleband_timing(tr: float | int, slice_order: list[int]) -> list[float]:
    """
    Create singleband timing based on slice order.

    Parameters
    ----------
    tr : :obj:`float` or :obj:`int`
        Repetition time in seconds.

    slice_order : :obj:`list[int]`
        Order of the slices.

    Returns
    -------
    list[float]
        Ordered slice timing information.
    """
    n_slices = len(slice_order)
    slice_duration = tr / n_slices
    slice_timing = np.linspace(0, tr - slice_duration, n_slices)
    # Pair slice with timing then sort dict
    sorted_slice_timing = dict(
        sorted({k: v for k, v in zip(slice_order, slice_timing.tolist())}.items())
    )

    return list(sorted_slice_timing.values())


def _generate_sequence(
    start: int, n_count: int, step: int, ascending: bool
) -> list[int]:
    """
    Generate a sequence of numbers.

    Parameters
    ----------
    start : :obj:`int`
        Starting number.

    n_count : :obj:`int`
        The amount of numbers to generate.

    step : :obj:`int`
        Step size between numbers.

    ascending : :obj:`int`
        If numbers are ascending or descending relative to ``start``.

    Returns:
        list[int]
            The sequence list.
    """
    if ascending:
        stop = start + n_count * step
        return np.arange(start, stop, step).tolist()
    else:
        stop = start - n_count * step
        return np.arange(start, stop, -step).tolist()


def _create_multiband_slice_groupings(
    slice_order: list[int], multiband_factor: int, n_time_steps: int, ascending: bool
) -> list[tuple[int, int]]:
    """
    Create slice groupings for multiband based on ``multiband_factor``.

    Parameters
    ----------
    slice_order : :obj:`list[int]`
        Order of the slices from single slice acquisition.

    multiband_factor : :obj:`int`
        The multiband acceleration factor, which is the number of slices
        acquired simultaneously during multislice acquistion.

    n_time_steps : :obj:`int`
        The number of time steps computed by dividing the number of slices
        by the multiband factor.

    Returns
    -------
    list[tuple[int, int]]
        A list of tuples containing the binned slice indices

    Example
    -------
    >>> from nifti2bids.metadata import _create_mutiband_slice_groupings
    >>> slice_order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9] # interleaved order
    >>> _create_mutiband_slice_groupings(slice_order, multiband_factor=2, n_time_steps=5, ascending=True)
    >>> [(0, 5), (2, 7), (4, 9), (1, 6), (3, 8)]
    """
    slice_groupings = []
    for slice_indx in slice_order:
        if not any(
            slice_indx in multiband_group for multiband_group in slice_groupings
        ):
            # Prevents invalid slice groupings
            # which produce values outside of possible range
            sequence = _generate_sequence(
                slice_indx, multiband_factor, n_time_steps, ascending
            )
            if max(sequence) >= len(slice_order) or min(sequence) < 0:
                continue

            slice_groupings.append(tuple(sequence))

    return slice_groupings


def _create_multiband_timing(
    tr: float | int, slice_order: list[int], multiband_factor: int, ascending: bool
) -> list[float]:
    """
    Create multiband timing based on slice order.

    Parameters
    ----------
    tr : :obj:`float` or :obj:`int`
        Repetition time in seconds.

    slice_order : :obj:`list[int]`
        Order of the slices from single slice acquisition.

    multiband_factor : :obj:`int`
        The multiband acceleration factor, which is the number of slices
        acquired simultaneously during multislice acquisition.

    ascending : :obj:`bool`, default=True
        If slices were collected in ascending order (True) or descending
        order (False).

    Returns
    -------
    list[float]
        Ordered slice timing information for multiband acquisition.

    Example
    -------
    >>> from nifti2bids.metadata import _create_mutiband_timing
    >>> slice_order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9] # interleaved order
    >>> _create_mutiband_timing(0.8, slice_order, multiband_factor=2, ascending=True)
    >>> [0.0, 0.48, 0.16, 0.64, 0.32, 0.0, 0.48, 0.16, 0.64, 0.32]
    >>> # slices grouping: [[0, 5], [2, 7], [4, 9], [1, 6], [3, 8]]
    """
    n_slices = len(slice_order)
    if n_slices % multiband_factor != 0:
        raise ValueError(
            f"Number of slices ({n_slices}) must be evenly divisible by "
            f"multiband factor ({multiband_factor})."
        )

    # Step corresponds to number of unique slice timings and the index step size
    n_time_steps = n_slices // multiband_factor
    slice_duration = tr / n_time_steps
    unique_slice_timings = np.linspace(0, tr - slice_duration, n_time_steps)
    slice_timing = np.zeros(n_slices)

    slice_groupings = _create_multiband_slice_groupings(
        slice_order, multiband_factor, n_time_steps, ascending
    )
    for time_indx, multiband_group in enumerate(slice_groupings):
        slice_timing[list(multiband_group)] = unique_slice_timings[time_indx]

    return slice_timing.tolist()


def create_slice_timing(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    tr: float | int | None = None,
    slice_axis: Literal["i", "j", "k"] | None = None,
    slice_acquisition_method: Literal[
        "sequential", "interleaved", "central", "reversed_central"
    ] = "interleaved",
    ascending: bool = True,
    interleaved_pattern: Literal["even", "odd", "philips"] = "odd",
    multiband_factor: int | None = None,
    verbose: bool = False,
) -> list[float]:
    """
    Create slice timing dictionary mapping the slice index to its
    acquisition time.

    .. important:: For Philips, single-package is assumed.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    tr : :obj:`float` or :obj:`int`
        Repetition time in seconds. If None, the repetition time is
        extracted from the NIfTI header.

    slice_axis : :obj:`Literal["i", "j", "k"]` or :obj:`None`, default=None
        Axis the image slices were collected in. If None,
        determines the slice axis using metadata ("slice_end")
        from the NIfTI header.

    slice_acquisition_method : :obj:`Literal["sequential", "interleaved", "central", "reversed_central"]`, default="interleaved"
        Method used for acquiring slices.

        .. note::
           - "interleaved" is the common interleaving pattern (e.g [0, 2, 4, 6, 1, 3, 5, 7]),
             which is also equivalent to Philips' "default".

           - "central" is an order for Philips scanners that collect the middle slice first,
             then the remaining slices are selected in a ping pong order (e.g. [2, 1, 3, 0, 4] or
             [3, 2, 4, 1, 5, 0]).

           - "reversed_central" is an order for Philips scanners that collect the outer slice
             (starting at 0) first, then the remaining slices follow a ping pong order
             to the middle slice (e.g. [0, 4, 1, 3, 2] or [0, 5, 1, 4, 2, 3])

    ascending : :obj:`bool`, default=True
        If slices were collected in ascending order (True) or descending
        order (False).

        .. important::
           ``ascending`` always set to True when ``slice_acquisition_method`` is
           "central" and "reversed_central" to prevent ``numpy.flip`` from.

    interleaved_pattern : :obj:`Literal["even", "odd", "philips"]`, default="odd"
        If slices for interleaved acquisition were collected by acquiring the
        "even" or "odd" slices first. For "philips" (the interleaved implementation
        by Philips'), slices are acquired by a step factor equivalent to the rounded
        square root of the total slices mode (e.g. [0, 3, 6, 9, 1, 4, 7, 2, 5, 8];
        rounded sqrt of 10 is 3).

        .. important::
           Philips' "default" mode is equivalent to "interleave" with the pattern
           set to "odd", and ascending set to True.

    multiband_factor : :obj:`int` or :obj:`None`, default=None
        The multiband acceleration factor, which is the number of slices
        acquired simultaneously during multislice acquisition. Slice
        ordering is created using a step factor equivalent to
        ``n_slices / multiband_factor``. For instance, if ``n_slices`` is
        12 and ``slice_acquisition_method`` is "interleaved" with
        ``multiband_factor`` of 3, then the traditional interleaved
        order using the "odd" first ascending pattern is [0, 2, 4, 6, 8, 10,
        1, 3, 5, 7, 9, 11]. This order is then grouped into sets of 3
        with a step of 4 (12 slices divided by multiband factor of 3),
        resulting in slice groups: (0, 4, 8), (2, 6, 10), (1, 5, 9), (3, 7, 11).
        The final slice timing order is [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11].

        .. important::
           - Multiband grouping is primarily based on based on
             Philip's ordering for multiband acquisition for different
             slice acquisition methods. For more information refer to the
             `University of Washington Diagnostic Imaging Sciences Center Technical Notes
             <https://depts.washington.edu/mrlab/technotes/fmri.shtml>`_.

            - Parameter not used for "central" and "reversed_central"  as there is
              no reference to assure ordering in multiband acquisition.

    verbose : :obj:`bool`, default=False
        Logs "INFO" level information if True.

    Returns
    -------
    list[float]
        List containing the slice timing acquisition.

    References
    ----------
    Parker, David, et al. "Optimal Slice Timing Correction and Its Interaction with
    FMRI Parameters and Artifacts." Medical Image Analysis, vol. 35, Jan. 2017, pp. 434–445,
    https://doi.org/10.1016/j.media.2016.08.006. Accessed 28 Jan. 2022.

    SPM/Slice Timing - Wikibooks, open books for an open world. (2022). Wikibooks.org.
    https://en.wikibooks.org/wiki/SPM/Slice_Timing
    """
    slice_ordering_func = {
        "sequential": _create_sequential_order,
        "interleaved": _create_interleaved_order,
        "central": _create_central_order,
        "reversed_central": _create_reversed_central_order,
    }

    if multiband_factor and slice_acquisition_method in ["central", "reversed_central"]:
        raise NotImplementedError(
            "'central' and 'reversed_central' cannot be used with ``multiband_factor``."
        )

    slice_start = get_hdr_metadata(
        nifti_file_or_img=nifti_file_or_img, metadata_name="slice_start"
    )
    if slice_start != 0 and not np.isnan(slice_start):
        LGR.warning(
            "Slice start index must start at 0. Starting slice index begins at "
            f"index {slice_start} so slice timing may not be accurate."
        )

    n_slices = get_n_slices(nifti_file_or_img, slice_axis, verbose)
    acquisition_kwargs = {"n_slices": n_slices}
    if slice_acquisition_method == "interleaved":
        acquisition_kwargs.update({"interleaved_pattern": interleaved_pattern})

    if slice_acquisition_method not in ["central", "reversed_central"]:
        acquisition_kwargs.update({"ascending": ascending})

    slice_order = slice_ordering_func[slice_acquisition_method](**acquisition_kwargs)
    tr = tr if tr else get_tr(nifti_file_or_img, verbose)
    band_kwargs = {"tr": tr, "slice_order": slice_order}

    return (
        _create_singleband_timing(**band_kwargs)
        if not multiband_factor
        else _create_multiband_timing(
            multiband_factor=multiband_factor,
            ascending=ascending,
            **band_kwargs,
        )
    )


def is_3d_img(nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image) -> bool:
    """
    Determines if ``nifti_file_or_img`` is a 3D image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    Returns
    -------
    bool
        True if ``nifti_file_or_img`` is a 3D image.
    """
    return len(get_nifti_header(nifti_file_or_img).get_zooms()) == 3


def get_scanner_info(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
) -> tuple[str, str]:
    """
    Determines the manufacturer and model name of scanner.

    .. important::
        Assumes this information is in the "descrip" of the NIfTI
        header, which can contain any information.


    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    Returns
    -------
    tuple[str, str]
        The manufacturer and model name for the scanner.
    """
    if not (
        scanner_info := get_hdr_metadata(
            nifti_file_or_img=nifti_file_or_img,
            metadata_name="descrip",
            return_header=False,
        )
    ):
        raise ValueError("No scanner information in NIfTI header.")

    scanner_info = str(scanner_info.astype(str)).rstrip(" ")
    manufacturer_name, _, model_name = scanner_info.partition(" ")

    return manufacturer_name, model_name


def is_valid_date(date_str: str, date_fmt: str) -> bool:
    """
    Determine if a string is a valid date based on format.

    Parameters
    ----------
    date_str : :obj:`str`
        The string to be validated.

    date_fmt : :obj:`str`
        The expected format of the date.

    Return
    ------
    bool
        True if ``date_str`` has the format specified by ``date_fmt``.

    Example
    -------
    >>> from nifti2bids.metadata import is_valid_date
    >>> is_valid_date("241010", "%y%m%d")
        True
    """
    try:
        datetime.datetime.strptime(date_str, date_fmt)
        return True
    except ValueError:
        return False


def parse_date_from_path(path: str | Path, date_fmt: str) -> str | None:
    """
    Get date from the stem of a path.

    Parameters
    ----------
    path : :obj:`str` or :obj:`Path`
        The absolute path, name of file, or folder.

    date_fmt : :obj:`str`
        The expected format of the date.

    Returns
    -------
    str or None
        A string if a valid date based on specified ``date_fmt`` is detected
        or None if no valid date is detected.

    Example
    -------
    >>> from nifti2bids.metadata import parse_date_from_path
    >>> date_str = parse_date_from_path("101_240820_mprage_32chan.nii", "%y%m%d")
    >>> print(date_str)
        "240820"
    >>> folder = r"Users/users/Documents/101_240820"
    >>> date_str = parse_date_from_path(folder, "%y%m%d")
    >>> print(date_str)
        "240820"
    """
    split_pattern = "|".join(map(re.escape, ["_", "-", " "]))

    basename = Path(path).name
    split_basename = re.split(split_pattern, basename)

    date_str = None
    for part in split_basename:
        if is_valid_date(part, date_fmt):
            date_str = part
            break

    return date_str


def get_file_timestamp(path: Path | str) -> float:
    """
    Get timestamp of file.

    .. important::
       Returns timestamp of file creation for Windows
       and modification timestamp for non-Windows systems (e.g.,
       Linux, MAC, etc)

       `Info about date issue for Unix-based
       <https://docs.vultr.com/python/examples/get-file-creation-and-modification-date>`_.

    Parameter
    ---------
    path : :obj:`str` or :obj:`Path`
        Path to file.

    Return
    ------
    float
        The file timestamp (creation time for Windows and
        modification time for non-Windows systems).
    """
    stat = Path(path).stat()
    if sys.platform != "win32":
        timestamp = stat.st_mtime
    else:
        if hasattr(stat, "st_birthtime"):
            timestamp = stat.st_birthtime
        else:
            timestamp = stat.st_ctime

    return timestamp


def get_file_creation_date(path: str | Path, date_fmt: str) -> str:
    """
    Get creation date of a file

    .. important::
       Returns file creation date for Windows and file modification
       date for non-Windows systems (e.g., Linux, MAC, etc)

       `Info about date issue for Unix-based systems
       <https://docs.vultr.com/python/examples/get-file-creation-and-modification-date>`_.


    Parameters
    ----------
    path : :obj:`str` or :obj:`Path`
        Path to file.

    date_fmt : :obj:`str`
        The desired output format of the date.

    Returns
    -------
    str
        File creation date for Windows and modification date for non-Windows systems.
    """
    timestamp = get_file_timestamp(path)

    converted_timestamp = datetime.datetime.fromtimestamp(timestamp)

    return converted_timestamp.strftime(date_fmt)


def infer_task_from_image(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    task_volume_map: dict[str, int] | dict[int, str],
) -> str:
    """
    Infer the task based on the number of volumes in a 4D NIfTI image.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    task_volume_map : :obj:`dict[str, int]` or :obj:`dict[int, str]`
        A mapping of the task names to the expected number of
        volumes.

    Returns
    -------
    str
        The task name.

    Example
    -------
    >>> from nifti2bids.io import simulate_nifti_image
    >>> from nifti2bids.metadata import infer_task_from_image
    >>> img = simulate_nifti_image((100, 100, 100, 260))
    >>> task_volume_map = {"flanker": 300, "nback": 260}
    >>> infer_task_from_image(img, task_volume_map)
        "nback"
    """
    n_volumes = get_n_volumes(nifti_file_or_img)

    if isinstance(next(iter(task_volume_map)), str):
        volume_lookup = dict(zip(task_volume_map.values(), task_volume_map.keys()))
    else:
        volume_lookup = task_volume_map

    return volume_lookup.get(n_volumes)


@check_nifti()
def get_recon_matrix_pe(
    nifti_file_or_img: str | Path | nib.nifti1.Nifti1Image,
    phase_encoding_axis: Literal["i", "j", "k"],
) -> int:
    """
    Get the reconstruction matrix of the phase encoding axis.

    Parameters
    ----------
    nifti_file_or_img : :obj:`str`, :obj:`Path`, or :obj:`Nifti1Image`
        Path to the NIfTI file or a NIfTI image.

    phase_encoding_axis : :obj:`Literal["i", "j", "k"]`
        The axis phase encoding was applied to.

    Returns
    -------
    int
        The value of the reconstruction matrix for ``phase_encoding_axis``,
        which is just the shape of the data in that dimension.
    """
    img = load_nifti(nifti_file_or_img)

    return img.get_fdata().shape[_VOXEL_INDX_DICT[phase_encoding_axis]]


def compute_effective_echo_spacing(
    water_fat_shift_pixels: float, epi_factor: int
) -> float:
    """
    Compute the effective echo spacing for Philips 3T MRI.

    The following formula is used:

    effective_echo_spacing (in seconds) = water_fat_shift (in pixels) / (field_strength * gyromagnetic ratio * water_fat_difference_ppm) * (epi_factor + 1)

    where:
    field_strength (assumed to be 3T) * gyromagnetic ratio * water_fat_difference_ppm = 3 * 42.58 * 3.4 is approximately 434.215

    Parameters
    ----------
    water_fat_shift_pixels : :obj:`float`
        The water and fat chemical shift in pixels.

    epi_factor : :obj:`int`
        The EPI factor or the number of echoes per excitation.

        .. note:: in plane acceleration already accounted for in this factor

    Returns
    -------
    float
        The effective echo spacing in seconds.

    References
    ----------
    sdcflows.utils.epimanip module - sdcflows 0+unknown documentation. (2022). Nipreps.org.
    https://www.nipreps.org/sdcflows/master/api/sdcflows.utils.epimanip.html#mjx-eqn%3Aeq%3Arotime-ees
    """
    return water_fat_shift_pixels / (434.215 * (epi_factor + 1))


def compute_total_readout_time(
    effective_echo_spacing: float | None = None,
    recon_matrix_pe: int | None = None,
    use_fallback_trt: bool = False,
) -> float:
    """
    Compute the total readout time.

    The following formula is used:

    total_readout_time = effective_echo_spacing * (recon_matrix_pe - 1)

    where:
    echo_train_length = epi_factor + 1
    (Number of echos acquired every radiofrequency pulse (repetition time))

    Parameters
    ----------
    effective_echo_spacing : :obj:`float`
        The effective echo spacing in seconds.

    recon_matrix_pe : :obj:`int`
        The number of pixels in the phase encoding axis of the reconstruction matrix.

    use_fallback_trt : :obj:`bool`
        If True, a fallback readout time of 0.03125 is used.

    Returns
    -------
    float
        The total readout time in seconds.

    References
    ----------
    sdcflows.utils.epimanip module - sdcflows 0+unknown documentation. (2022). Nipreps.org.
    https://www.nipreps.org/sdcflows/master/api/sdcflows.utils.epimanip.html#mjx-eqn%3Aeq%3Arotime-ees
    """
    if not (effective_echo_spacing or recon_matrix_pe) and not use_fallback_trt:
        raise ValueError(
            "`effective_echo_spacing` and `recon_matrix_pe` must be provided when `use_fallback_trt` is False."
        )

    return (
        effective_echo_spacing * (recon_matrix_pe - 1)
        if not use_fallback_trt
        else 0.03125
    )


__all__ = [
    "needs_resampling",
    "determine_slice_axis",
    "get_hdr_metadata",
    "get_n_volumes",
    "get_image_orientation",
    "direction_to_voxel_axis",
    "get_n_slices",
    "get_tr",
    "create_slice_timing",
    "is_3d_img",
    "get_scanner_info",
    "is_valid_date",
    "parse_date_from_path",
    "get_file_timestamp",
    "get_file_creation_date",
    "infer_task_from_image",
    "get_recon_matrix_pe",
    "compute_effective_echo_spacing",
    "compute_total_readout_time",
]

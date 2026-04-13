"Utilities related to parsing or sorting filenames."

import datetime, re, sys
from pathlib import Path
from re import Match

from .logging import setup_logger

LGR = setup_logger(__name__)


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
    >>> from bidsaid.metadata import is_valid_date
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
    >>> from bidsaid.path_utils import parse_date_from_path
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


def get_file_acquisition_order(
    filename: str | Path, return_as_string: bool = True
) -> str | Match[str] | None:
    r"""
    Get the file acquistion order using the following regex "(?<=_)(\d+)(?:_(\d+))?(?=\.)".

    Parameters
    ----------
    filename: :obj:`str` or :obj:`Path`
        The filename.

    return_as_string: :obj:`bool`, default=True
        Return as a string or a ``Match[str]`` object.

    Returns
    -------
    str, Match[str], or None
        A string or Match[str] object of the capture groups if found else returns None.
    """
    pattern = r"(?<=_)(\d+)(?:_(\d+))?(?=\.)"
    capture_groups = re.search(pattern, Path(filename).name)
    try:
        return capture_groups.group(0) if return_as_string else capture_groups
    except AttributeError:
        LGR.warning(
            r"The following file does not have the expected pattern ('(?<=_)(\d+)(?:_(\d+))?(?=\.)') "
        )
        return None


def sort_by_acquisition_order(
    filenames: list[str] | list[Path], return_filenames_only: bool = True
) -> list[Path] | list[tuple[int, int, Path]]:
    r"""
    Sort filenames based on acquisition sequence.

    Sorts numerically based on the acquisition order implied by the filename and lexicographically
    based on parent filename and full basename.

    For instance:

    ::

        filenames = ["101_10_2.nii", "101_10_1.nii", "101_3_1.nii"]
        sorted_filenames = ["101_3_1.nii", "101_10_1.nii", "101_10_2.nii"]

    Note, numerical sorting depends on the acquisition order in the filename since the
    regex pattern used for sorting is "(?<=_)(\d+)(?:_(\d+))?(?=\.)". This is used to
    create a tuple (capture_group_2, capture_group_1, filename), which consist of the last capture
    group, the first capture group, and the filename. Thus, the sorting can result in:

        filenames = ["101_10_2.nii", "101_10_1.nii", "101_3_1.nii", "102_2_1.nii", "102_1_1.nii"]
        sorted_filenames = ["102_1_1.nii", "102_2_1.nii", "101_3_1.nii", "101_10_1.nii", "101_10_2.nii"]

    Parameters
    ----------
    filenames : :obj:`list[str]` or :obj:`list[Path]`
        A list of paths.

    return_filenames_only : :obj:`bool`, default=True
        If True, returns a list of sorted filenames.

    Returns
    -------
    list[Path] or list[tuple[int, int, Path]]
        A list of paths sorted based on acquisition order if ``return_filenames_only`` is True
        else returns a list of sorted tuples where each tuple has the form (capture_group_2, capture_group_1, filename).
    """
    tuple_list = []
    for filename in filenames:
        capture_groups = get_file_acquisition_order(filename, return_as_string=False)
        if not capture_groups:
            continue

        try:
            last_number = int(capture_groups.group(2))
        except AttributeError:
            last_number = 1

        first_number = int(capture_groups.group(1))
        tuple_list.append((last_number, first_number, Path(filename)))

    sorted_tuple_list = sorted(tuple_list)

    return (
        [x[-1] for x in sorted_tuple_list]
        if return_filenames_only
        else sorted_tuple_list
    )


__all__ = [
    "is_valid_date",
    "parse_date_from_path",
    "get_file_timestamp",
    "get_file_creation_date",
    "get_file_acquisition_order",
    "sort_by_acquisition_order",
]

import csv, io, tempfile, subprocess, sys
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np, pandas as pd

from ._constants import EDATAAID_PATH, EDATAAID_CONTROL_FILE


def _determine_delimiter(
    textlines: list[str], initial_column_headers: Iterable[str]
) -> str:
    """
    Identify the delimiter used for the data based on the
    delimiter used for the initial column header.

    Parameters
    ----------
    textlines : :obj:`list[str]`
        The lines of text from the presentation log file.

    initial_column_headers : :obj:`Iterable[str]`
        The initial column headers for data.

    Returns
    -------
    str
        The delimiter
    """
    sniffer = csv.Sniffer()
    for indx, line in enumerate(textlines):
        if line.startswith(tuple(initial_column_headers)):
            header_string = textlines[indx]

    return sniffer.sniff(header_string, delimiters=None).delimiter


def _convert_time(
    df: pd.DataFrame, convert_to_seconds: list[str], divisor: int
) -> pd.DataFrame:
    """
    Change time resolution of specific columns.

    Parameters
    ----------
    presentation_df : :obj:`pandas.DataFrame`
        Pandas Dataframe of the Presentation log

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Columns to convert to time.

    divisor : :obj:`int` or :obj:`None`, default=None
        Value to divide columns listed in ``convert_to_columns`` by.

    Returns
    -------
    pandas.Dataframe
        Dataframe with timing of the columns listed in
        ``convert_to_seconds`` converted to the units
        of the ``divisor``floats and time resolution
    """
    convert_to_seconds = (
        [convert_to_seconds]
        if isinstance(convert_to_seconds, str)
        else convert_to_seconds
    )
    df[convert_to_seconds] = df[convert_to_seconds].apply(
        lambda x: x.astype(str).str.lower()
    )
    df[convert_to_seconds] = df[convert_to_seconds].replace("null", "nan")
    df[convert_to_seconds] = df[convert_to_seconds].replace("", "nan")
    df[convert_to_seconds] = df[convert_to_seconds].astype(float)
    df[convert_to_seconds] = df[convert_to_seconds].apply(lambda x: x / divisor)

    return df


def convert_edat3_to_text(
    edat_path: str | Path,
    dst_path: Optional[str | Path] = None,
    format: Literal["csv", "tsv"] = "csv",
    return_dst_path: bool = True,
) -> Path | None:
    """
    Converts a file with an "edat3" extension to a text file.

    .. important::
       - Only works with Windows platforms with E-Prime 3 installed.

    edat_path : :obj:`str` or :obj:`Path`
        Absolute path to the E-Prime file with an "edat3" extension.

    dst_path : :obj:`str` or :obj:`Path`, default=None
        Absolute path to the output text file that the edat3 file will be converted to.
        If None, the text file will be saved in the same folder as the edat3 file.

    format : :obj:`Literal["csv", "tsv"], default="csv"
        The file extension.

    return_dst_path : :obj:`bool`, default=True
        Returns the destination path if True.

    Returns
    -------
    Path or None
        Returns the destination path if ``return_dst_path`` is True.
    """
    assert (
        Path(edat_path).suffix == ".edat3"
    ), "`edat_path` must be a file with the '.edat3' extension."

    if sys.platform != "win32":
        raise OSError("Function only works for Windows platforms.")

    if not Path(EDATAAID_PATH).exists:
        raise FileNotFoundError(
            f"E-Prime 3 must be installed to use the following program {EDATAAID_PATH}."
        )

    assert (format := format.lower()) in [
        "csv",
        "tsv",
    ], f"`format` must be 'csv' or 'tsv'."

    delimiter_dict = {"csv": ",", "tsv": "\t"}
    dst_path = dst_path if dst_path else str(edat_path).replace(".edat3", f".{format}")
    control_file = EDATAAID_CONTROL_FILE.format(
        edat_path=edat_path, dst_path=dst_path, delimiter=delimiter_dict[format]
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmpfile:
        tmpfile.write(control_file)

    # https://stackoverflow.com/questions/7006238/how-do-i-hide-the-console-when-i-use-os-system-or-subprocess-call
    # Hide Window and prevent flashing
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE

    subprocess.run(
        [EDATAAID_PATH, "/e", "/f", tmpfile.name], startupinfo=startupinfo, shell=False
    )

    Path(tmpfile.name).unlink()

    return Path(dst_path) if return_dst_path else None


def _is_float(value: str) -> bool:
    """Checks if string value is a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def _text_to_df(
    log_filepath: str | Path, initial_column_headers: Iterable[str]
) -> pd.DataFrame:
    """
    Convert text to a DataFrame.

    Parameters
    ----------
    log_filepath : :obj:`str` or :obj:`Path`
        Path to log data.

    initial_column_headers :obj:`Iterable[str]`
        The initial column headers for data.

    Returns
    -------
    pandas.DataFrame
        A clean DataFrame of the log data.
    """
    with open(log_filepath, "r") as f:
        initial_column_headers = tuple(initial_column_headers)
        textlines = f.readlines()
        delimiter = _determine_delimiter(textlines, initial_column_headers)
        content_indices = []
        cleaned_textlines = [line for line in textlines if line != "\n"]
        for indx, line in enumerate(cleaned_textlines):
            # Get the starting index of the data columns
            if line.startswith(f"{delimiter}".join(initial_column_headers)):
                content_indices.append(indx)
            # Get one more than the final index of the data colums
            # More flexible, checks to see if line has at least one digit, which is likely a data line
            elif content_indices and not any(
                element.isdigit() or _is_float(element)
                for element in line.split(f"{delimiter}")
            ):
                content_indices.append(indx)
                break

        start_indx = content_indices[0]
        stop_indx = (
            content_indices[1] if len(content_indices) > 1 else len(cleaned_textlines)
        )

        text = "".join(cleaned_textlines[start_indx:stop_indx])
        df = pd.read_csv(
            io.StringIO(text, newline=None), sep=delimiter, low_memory=False
        )

    return df


def load_eprime_log(
    log_filepath: str | Path,
    convert_to_seconds: list[str] = None,
    initial_column_headers: Iterable[str] = ("ExperimentName", "Subject"),
) -> pd.DataFrame:
    """
    Loads E-Prime 3 log file as a Pandas Dataframe.

    .. important::
       - If the log file extension is "edat3", use :func:`nifti2bids.parsers.convert_edat3_to_tsv`
         to convert it to text form. If exporting manually, remove the checkmark from
         the "Unicode" field. The type of text file the edat file is exported as is irrelevent.

       - Data are assumed to have at least one element that is an digit or float
         during parsing.

    Parameters
    ----------
    log_filepath : :obj:`str` or :obj:`Path`
        Absolute path to the Presentation log file (i.e text, log, excel files).

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from milliseconds to seconds.
        See `EPrime Timing <http://support.pstnet.com/hc/en-us/articles/4414899313303-TIMING-E-Prime-clock-starting-point-34474>`_.

    initial_column_headers : :obj:`Iterable[str]`, default=("ExperimentName", "Subject")
        The initial column headers for data.

    drop_columns : :obj:`list[str]` or :obj:`None`, default=None
        Remove specified columns from dataframe.

    Returns
    -------
    pandas.Dataframe
        A Pandas DataFrame of the behavioral log data.

    Notes
    -----
    This function works by first identifying the line containing the column headers specified in
    ``initial_column_headers`` and using that line to extract the delimiter (assumed to be the
    delimiter for the data). After, all blank lines are removed then the remaining lines in the
    log file is iterated through to identify the boundaries of the experimental log data.
    Data is assumed to to contain at least one digit or float (which includes NaN).
    """
    assert (
        not Path(log_filepath).suffix == ".edat3"
    ), "`log_filepath` cannot be a file with the '.edat3' extension."

    df = _text_to_df(log_filepath, initial_column_headers)
    df.replace("", np.nan, inplace=True)

    return (
        df
        if not convert_to_seconds
        else _convert_time(df, convert_to_seconds, divisor=1e3)
    )


def load_presentation_log(
    log_filepath: str | Path,
    convert_to_seconds: list[str] = None,
    initial_column_headers: Iterable[str] = ("Trial", "Event Type"),
) -> pd.DataFrame:
    """
    Loads Presentation log file as a Pandas DataFrame.

    .. important::
        Data are assumed to have at least one element that is an digit or float
        during parsing.

    Parameters
    ----------
    log_filepath : :obj:`str` or :obj:`Path`
        Absolute path to the Presentation log file (i.e text, log, Excel files).

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1ms to seconds.
        See `Presentation Timing <https://www.neurobs.com/pres_docs/html/04_reference/03_pcl_reference/03_stimulus_types/38_stimulus_data.htm>`_.

    initial_column_headers : :obj:`str`, default=("Trial", "Event Type")
        The initial column headers for data.

    Returns
    -------
    pandas.Dataframe
        A Pandas DataFrame of the behavioral log data.

    Notes
    -----
    This function works by first identifying the line containing the column headers specified in
    ``initial_column_headers`` and using that line to extract the delimiter (assumed to be the
    delimiter for the data). After, all blank lines are removed then the remaining lines in the
    log file is iterated through to identify the boundaries of the experimental log data.
    Data is assumed to to contain at least one digit or float (which includes NaN).
    """
    df = _text_to_df(log_filepath, initial_column_headers)
    df.replace("", np.nan, inplace=True)

    return (
        df
        if not convert_to_seconds
        else _convert_time(df, convert_to_seconds, divisor=1e4)
    )


__all__ = [
    "convert_edat3_to_text",
    "load_eprime_log",
    "load_presentation_log",
]

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._helpers import is_path
from ..logging import setup_logger

LGR = setup_logger(__name__)


def compute_n_dummy_scans(confounds_file_or_df: str | Path | pd.DataFrame) -> int:
    """
    Compute the number of dummy scans based on the number
    of "non_steady_state_outlier_XX" columns in an fMRIPrep
    confounds TSV file.

    Parameters
    ----------
    confounds_file_or_df : :obj:`str`, :obj:`Path`, or :obj:`pandas.DataFrame`
        A confounds file or dataframe from fMRIPrep.

    Returns
    -------
    int
        Number of non-steady state scans.
    """
    df = (
        pd.read_csv(confounds_file_or_df, sep="\t")
        if is_path(confounds_file_or_df)
        else confounds_file_or_df
    )

    return sum(col.startswith("non_steady_state_outlier") for col in df.columns)


def _get_input_data(
    input_data: str | Path | pd.DataFrame | NDArray,
    column_name: str | None = None,
    has_header: bool = True,
    verbose: bool = False,
    return_df: bool = False,
) -> NDArray | pd.DataFrame:
    """
    Returns the input data.

    Parameters
    ----------
    input_data : :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`, or :obj:`NDArray`
        Input data containing values to threshold. Can be a path to a text
        file (where delimiter is whitespace, tabs, or commas), a DataFrame,
        or a 1D numpy array.

    column_name : :obj:`str` or :obj:`None`, default=None
        Name of the column to extract from DataFrame or file. If None and
        input is a DataFrame/file, uses the first column.

    has_header : :obj:`bool`, default=True
        Whether the input file has a header row. Only used when ``input_data``
        is a file path.

    verbose : :obj:`bool`, default=False
        Logs "INFO" level information if True.

    return_df :  :obj:`bool`, default=False
        Returns the full DataFrame if True.

    Returns
    -------
    NDArray or pandas.DataFrame
        Numpy array of the input data if ``return_df`` ir True else returns a pandas dataframe.
    """
    if is_path(input_data):
        input_data = pd.read_csv(
            input_data,
            sep=r"[\s,]+",
            engine="python",
            header=(0 if has_header else None),
        ).fillna(0)

    if isinstance(input_data, pd.DataFrame) and not return_df:
        if column_name:
            input_data = input_data[column_name].to_numpy()
        else:
            input_data = input_data.iloc[:, 0].to_numpy()

    if verbose:
        LGR.info(f"The data has the following dimensions: {input_data.shape}")

    return input_data


def compute_framewise_displacement(
    input_data: str | Path | pd.DataFrame | NDArray,
    has_header: bool = True,
    rotation_units: Literal["radians", "degrees"] = "radians",
    radius: float | int = 50,
    verbose: bool = False,
) -> NDArray:
    """
    Computes the framewise displacement using the Power (2012) formula:

    FDi=|Δxi|+|Δyi|+|Δzi|+|Δαi|+|Δβi|+|Δγi|

    - where i is the timepoint/volume
    - x, y and z are the three translation parameters
    - α, β and γ are the three rotational parameters
    - Δxi=xi−1−xi

    Parameters
    ----------
    input_data : :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`, or :obj:`NDArray`
        Input data containing values to threshold. Can be a path to a text
        file (where delimiter is whitespace, tabs, or commas), a DataFrame.

        .. important::
           - If ``has_header`` is False, then the first six columns will be assumed to be the
             three translations and three rotations.
           - Assumes first three columns are translation parameters in mm and last three columns
             are rotation parameters in radians or degrees.

    has_header : :obj:`bool`, default=True
        Whether the input file has a header row. Only used when ``input_data``
        is a file path.

    rotation_units : :obj:`Literal["radians", "degrees"]`, default="radians"
        The units of the rotation parameters.

    radius : :obj:`int` or :obj:`float`, default=50
        The radius of the head in mm.

    verbose : :obj:`bool`, default=False
        Logs "INFO" level information if True.

    Returns
    -------
    NDArray
        A numpy array of the computed framewise displacements. Note that the first
        value will always be 0.

    References
    ----------
    Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012).
    Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion.
    NeuroImage, 59(3), 2142–2154. https://doi.org/10.1016/j.neuroimage.2011.10.018

    Mejia, A., Muschelli, J., & Pham, D. fMRIscrub: fMRI scrubbing (R package).
    https://neuroconductor.org/help/fMRIscrub/reference/FD.html
    """
    if rotation_units not in ["radians", "degrees"]:
        raise ValueError("`rotation_units` must be either 'radians' or 'degrees'")

    if not isinstance(input_data, np.ndarray):
        data = _get_input_data(
            input_data, has_header=has_header, verbose=verbose, return_df=True
        )
    else:
        data = input_data

    if data.shape[1] < 6:
        raise ValueError(
            "The data has less than 6 columns. "
            "The formula requires three tranlations and three rotational parameters."
        )

    if isinstance(data, pd.DataFrame):
        if has_header:
            arr = data[
                ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
            ].to_numpy(copy=True)
        else:
            arr = data.to_numpy(copy=True)
    else:
        arr = data

    arr = arr[:, :6]
    if rotation_units == "radians":
        arr[:, 3:] *= radius
    else:
        arr[:, 3:] = np.deg2rad(arr[:, 3:]) * radius

    fd_arr = np.abs(np.diff(arr, axis=0)).sum(axis=1)

    return np.insert(fd_arr, 0, 0)


def create_censor_mask(
    input_data: str | Path | pd.DataFrame | NDArray,
    column_name: str | None = None,
    threshold: float | None = None,
    n_dummy_scans: int = 0,
    has_header: bool = True,
    verbose: bool = False,
) -> NDArray:
    """
    Create a censor mask where 0 indicates volumes to censor and 1 indicates
    volumes to keep.

    Parameters
    ----------
    input_data : :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`, or :obj:`NDArray`
        Input data containing values to threshold. Can be a path to a text
        file (where delimiter is whitespace, tabs, or commas), a DataFrame,
        or a 1D numpy array.

    column_name : :obj:`str` or :obj:`None`, default=None
        Name of the column to extract from DataFrame or file. If None and
        input is a DataFrame/file, uses the first column.

    threshold : :obj:`float` or :obj:`None`, default=None
        Values exceeding this threshold will be censored (set to 0 in mask).
        If None, no threshold-based censoring is applied.

    n_dummy_scans : :obj:`int`, default=0
        Number of non-steady-state scans to censor.

    has_header : :obj:`bool`, default=True
        Whether the input file has a header row. Only used when ``input_data``
        is a file path.

    verbose : :obj:`bool`, default=False
        Logs "INFO" level information if True.

    Returns
    -------
    NDArray
        Binary mask array where 1 = keep, 0 = censor.

    Examples
    --------
    Censor only dummy scans:

    >>> censor_mask = create_censor_mask(data, n_dummy_scans=4)

    Censor based on framewise displacement threshold:

    >>> censor_mask = create_censor_mask(
    ...            confounds_df,
    ...            column_name="framewise_displacement",
    ...            threshold=0.5
    ...            )

    Censor both dummy scans and high motion volumes:

    >>> censor_mask = create_censor_mask(
    ...            "confounds.tsv",
    ...            column_name="framewise_displacement",
    ...            threshold=0.5,
    ...            n_dummy_scans=4
    ...            )
    """
    input_data = _get_input_data(input_data, column_name, has_header, verbose)
    censor_mask = np.ones(input_data.shape[0], dtype=int)

    if n_dummy_scans > 0:
        censor_mask[:n_dummy_scans] = 0

    if threshold is not None:
        censor_mask[input_data > threshold] = 0

    return censor_mask


def merge_censor_masks(censor_masks: Iterable[NDArray]) -> NDArray:
    """
    Merge multiple censor masks.

    A volume is kept (1) only if it is **NOT** censored in **ALL** input masks.

    Parameters
    ----------
    censor_masks: :obj:`Iterable[NDArray]`
        An iterable of binary censor masks, all with the same length.

    Returns
    -------
    NDArray
        Merged binary mask where 1 = keep, 0 = censor.
    """
    return np.min(np.vstack(censor_masks), axis=0)


def _get_contiguous_segments(censor_mask: NDArray) -> NDArray:
    """
    Get contiguous segments of high motion and low motion data the censor mask.

    Parameters
    ----------
    censor_mask: :obj:`NDArray`
        A numpy array where 1 = keep, 0 = censor.

    Return
    ------
    NDArray
        A numpy array where each element is an array containing all 0's or all 1's.

    Example
    -------
    The logic:

    >>> import numpy as np
    >>> arr = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    >>> diff_arr = np.diff(arr) # [0, 1, 0, -1, 0, 1, 0, -1, 0]
    >>> indxs = np.where(diff_arr != 0)[0] + 1 # [1, 3 ,5 , 7] -> [2, 4, 6, 8]
    >>> segments = np.split(arr, indxs)
    >>> print(segments)
        [array([0, 0]), array([1, 1]), array([0, 0]), array([1, 1]), array([0, 0])]
    """
    split_indices = np.where(np.diff(censor_mask, n=1) != 0)[0] + 1

    return np.split(censor_mask, split_indices)


def compute_consecutive_censor_stats(
    censor_mask: NDArray, n_dummy_scans: int = 0
) -> dict[str, float]:
    """
    Compute the mean and standard deviation of the consecutive censored volumes.

    Parameters
    ----------
    censor_mask: :obj:`NDArray`
        A numpy array where 1 = keep, 0 = censor.

    n_dummy_scans : :obj:`int`, default=0
        Number of non-steady-state scans to censor.

    Return
    ------
    dict[str, float]
        A dictionary with the following keys:

        - "consecutive_censored_volumes_mean": Mean number of consecutively
          censored volumes. Returns 0.0 if no volumes were censored.
        - "consecutive_censored_volumes_std": Standard deviation of consecutively
          censored volumes. Returns NaN if no volumes were censored.
    """
    metric_dict = {
        "consecutive_censored_volumes_mean": 0.0,
        "consecutive_censored_volumes_std": np.nan,
    }

    segments = _get_contiguous_segments(censor_mask[n_dummy_scans:])
    censored_segments_counts = np.array(
        [len(segment) for segment in segments if segment[0] == 0]
    )

    if censored_segments_counts.size > 0:
        metric_dict["consecutive_censored_volumes_mean"] = (
            censored_segments_counts.mean()
        )
        metric_dict["consecutive_censored_volumes_std"] = censored_segments_counts.std(
            ddof=0
        )

    return metric_dict


def create_spike_regressors(censor_mask: NDArray) -> pd.DataFrame:
    """
    Creates spike regressors that can be used to censor certain frames during
    regression.

    Parameters
    ----------
    censor_mask: :obj:`NDArray`
        A numpy array where 1 = keep, 0 = censor.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of spike regressors
    """
    spike_regressor_names = [
        f"spike_regressor_{i}" for i in range(censor_mask[censor_mask == 0].size)
    ]

    indices = np.where(censor_mask == 0)
    spike_regressors = np.eye(censor_mask.size, dtype=int)[indices].T

    return pd.DataFrame(spike_regressors, columns=spike_regressor_names or None)


def get_n_censored_volumes(arr_or_file: str | Path | NDArray) -> int:
    """
    Computes number of censored volumes.

    Parameters
    ----------
    arr_or_file : :obj:`str`, :obj:`Path`, :obj:`NDArray`
        A numpy array or text file where 1 = keep, 0 = censor.

    Returns
    -------
    int
        Number of censored volumes
    """
    arr = np.loadtxt(arr_or_file) if is_path(arr_or_file) else arr_or_file
    arr = arr.astype(int)

    return arr[arr == 0].size

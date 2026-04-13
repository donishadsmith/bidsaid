"""Module for extracting timing information from Presentation or E-Prime."""

import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal

import numpy as np, pandas as pd

from bidsaid._helpers import iterable_to_str
from bidsaid.logging import setup_logger
from bidsaid.parsers import (
    load_eprime_log,
    load_presentation_log,
    _convert_time,
    _is_float,
)

LGR = setup_logger(__name__)


def _process_log_or_df(
    log_or_df: str | Path | pd.DataFrame,
    convert_to_seconds: list[str] | None,
    initial_column_headers: Iterable[str],
    divisor: float | int,
    software: Literal["Presentation", "E-Prime"],
):
    """
    Processes the event log from a neurobehavioral software.

    Parameters
    ----------
    log_or_df : :obj:`str`, :obj:`Path`, :obj:`pd.DataFrame`
        The log or DataFrame of event informaiton from a neurobehavioral software.

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns.

    initial_column_headers : :obj:`Iterable[str]`
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    divisor : :obj:`float` or :obj:`int`
        Value to divide columns specified in ``convert_to_seconds`` by.

    software : :obj:`Literal["Presentation", "EPrime"]
        The specific neurobehavioral software.

    Returns
    -------
    pandas.DataFrame
        A dataframe of the task information.
    """
    loader = {"Presentation": load_presentation_log, "E-Prime": load_eprime_log}

    if not isinstance(log_or_df, pd.DataFrame):
        df = loader[software](
            log_or_df,
            convert_to_seconds=convert_to_seconds,
            initial_column_headers=tuple(initial_column_headers),
        )
    elif convert_to_seconds:
        df = _convert_time(
            log_or_df.copy(deep=True),
            convert_to_seconds=convert_to_seconds,
            divisor=divisor,
        )
    else:
        df = log_or_df

    return df.replace("", np.nan, inplace=False)


def combine_non_block_cue_names(rest_block_codes: Iterable[str] | None, quit_code: str):
    """
    Combines name of rest codes and quit code.

    Parameters
    ----------
    rest_block_codes : :obj:`Iterable[str]` or :obj:`None`
        The rest block code(s).

    quit_code : :obj:`str`
        The quit code.

    Returns
    -------
    list[str] or list[None]
        The combined elements in a list
    """
    if isinstance(rest_block_codes, Iterable):
        return rest_block_codes + [quit_code]
    else:
        return [rest_block_codes] + [quit_code]


def _validate_no_cue_conflicts(
    block_cue_names: Iterable[str],
    rest_block_codes: Iterable[str] | None,
    quit_code: str | None,
) -> None:
    """
    Check for overlap between task cues and boundary codes.

    Parameters
    ----------
    block_cue_names : :obj:`Iterable[str]`
        The names of the block cues.

    rest_block_codes : :obj:`Iterable[str]` or :obj:`None`
        The rest block code(s).

    quit_code : :obj:`str`
        The quit code.
    """
    conflicting_codes = set(block_cue_names).intersection(
        filter(None, combine_non_block_cue_names(rest_block_codes, quit_code))
    )

    if conflicting_codes:
        raise ValueError(
            f"Conflicting codes found: {iterable_to_str(conflicting_codes)}. "
            "A code cannot be in 'block_cue_names' and also be assigned to "
            "``rest_block_codes`` or ``quit_code``. To include rest/quit "
            "intervals in the output, add them to ``block_cue_names`` instead."
        )


def _get_starting_block_indices(
    log_df: pd.DataFrame, trial_column_name: str, block_cue_names: Iterable[str]
) -> list[int]:
    """
    Get starting indices for blocks.

    Parameters
    ----------
    log_df : :obj:`pandas.DataFrame`
        DataFrame of neurobehavioral log data.

    trial_column_name : :obj:`str`
        Name of the column containing the trial information.

    block_cue_names : :obj:`Iterable[str]`
        The names of blocks.

    Returns
    -------
    list[int]
        The starting index of each block.
    """
    trial_series = log_df[trial_column_name]

    # Get the starting index for each block via grouping indices with the same
    trial_list = trial_series.tolist()
    starting_block_indices = []
    current_index = 0
    for _, group in itertools.groupby(trial_list):
        starting_block_indices.append(current_index)
        current_index += len(list(group))

    starting_block_indices = set(starting_block_indices)
    for trial_type in trial_series.unique():
        if trial_type not in block_cue_names:
            trial_indxs = trial_series[trial_series == trial_type].index.tolist()
            starting_block_indices = starting_block_indices.difference(trial_indxs)

    # Remove empty
    missing_indices = trial_series[trial_series.isna()].index.tolist()
    starting_block_indices = starting_block_indices.difference(missing_indices)

    return sorted(list(starting_block_indices))


def _get_next_block_index(
    trial_series: pd.Series,
    block_start_index: int,
    rest_block_codes: Iterable[str] | None,
    rest_code_frequency: Literal["fixed", "variable"],
    block_cue_names: Iterable[str],
    quit_code: str | None = None,
) -> int:
    """
    Get the starting index for each block.

    Parameters
    ----------
    trial_series : :obj:`pandas.Series`
        A Pandas Series of the column containing the trial type information.

    block_start_index : :obj:`int`
        The current row index.

    rest_block_codes : :obj:`Iterable[str]` or :obj:`None`
        The rest block code(s).

    rest_code_frequency : :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest blocks(s).

    block_cue_names : :obj:`Iterable[str]`
        The names of the blocks. When used, identifies
        the indices of all block names minus the indices
        corresponding to the current trial type. Used when
        ``rest_block_codes`` is not None and ``rest_code_frequency``
        is not "fixed".

    quit_code : :obj:`str` or :obj:`None`, default=None
        The quit code. Ideally, this should be a unique code.

    Returns
    -------
    int
        The starting index of the next block.
    """
    curr_trial = trial_series.at[block_start_index]
    filtered_trial_series = trial_series[trial_series.index > block_start_index]
    filtered_trial_series = filtered_trial_series.astype(str)

    if rest_block_codes and rest_code_frequency == "fixed":
        target_codes = filter(
            None, combine_non_block_cue_names(rest_block_codes, quit_code)
        )
        next_block_indxs = filtered_trial_series[
            filtered_trial_series.isin(tuple(target_codes))
        ].index.tolist()
    else:
        target_block_names = set(tuple(block_cue_names))
        target_block_names.discard(curr_trial)

        rest_code = rest_block_codes if rest_code_frequency == "variable" else None
        target_block_names.update(
            filter(None, combine_non_block_cue_names(rest_code, quit_code))
        )

        next_block_indxs = filtered_trial_series[
            filtered_trial_series.isin(target_block_names)
        ].index.tolist()

    return next_block_indxs[0] if next_block_indxs else trial_series.index[-1]


def _separate_block_from_cue(
    split_cue_as_instruction: bool,
    block_cue_name: str,
    block_cues_without_instruction: Iterable[str],
) -> bool:
    """
    Determine whether to seperate the block from instruction cue.

    Parameters
    ----------
    split_cue_as_instruction : :obj:`bool`
        Whether to separate cue as instruction trial.

    block_cue_name : :obj:`str`
        The name of the block cue.

    block_cues_without_instruction : :obj:`Iterable[str]`
        Names of the blocks that should not have their cue separated
        as instruction.
    """
    return (
        split_cue_as_instruction
        and block_cue_name not in block_cues_without_instruction
    )


def _extract_block_trial_types(
    df: pd.DataFrame,
    starting_block_indices: list[int],
    trial_column_name: str,
    split_cue_as_instruction: bool = False,
    block_cues_without_instruction: Iterable[str] = (),
    instruction_suffix: str = "_instruction",
    drop_instruction_cues: bool = False,
) -> list[str]:
    """
    Extract trial types for each blocks.

    Parameters
    ----------
    df : :obj:`pd.DataFrame`
        DataFrame containing the log data.

    starting_block_indices : :obj:`list[int]`
        List of row indices to extract trial types from.

    trial_column_name : :obj:`str`
        Name of the column containing trial type information.

    split_cue_as_instruction : :obj:`bool`, default=False
        Whether to separate cue as instruction trial. Will compute timing information
        from the row index immediately proceeding the row index of the block cue.

    block_cues_without_instruction : :obj:`Iterable[str]` or :obj:`tuple[()], default=()
        If ``split_cue_as_instruction`` is True and certain names from
        ``block_cue_names`` are specified in this parameter, then those
        blocks will not have their cue separated as instruction and will always start
        on the row index containing that name.

    instruction_suffix : :obj:`str`, default="_instruction"
        Suffix to append to trial type names for instruction trials.

    drop_instruction_cues : :obj:`bool`, default=False
        If True and ``split_cue_as_instruction`` is True, the instruction
        trials are not included in the output.

    Returns
    -------
    list[str]
        A list of trial types for each block.
    """
    trial_types = []
    for block_start_index in starting_block_indices:
        block_cue_name = df.loc[block_start_index, trial_column_name]
        should_separate_block = _separate_block_from_cue(
            split_cue_as_instruction,
            block_cue_name,
            block_cues_without_instruction,
        )

        if should_separate_block and not drop_instruction_cues:
            trial_types.extend(
                [f"{block_cue_name}{instruction_suffix}", block_cue_name]
            )
        else:
            trial_types.append(block_cue_name)

    return trial_types


def _filter_response_trial_names(
    response_trial_names: Iterable[str] | None,
    split_cue_as_instruction: bool,
    block_cue_names: Iterable[str],
) -> Iterable[str] | None:
    """
    Filter response trials in blocks by removing block cue names.

    When ``split_cue_as_instruction`` is True, block cue names are removed
    from ``response_trial_names`` since cue rows do not contain responses.

    Parameters
    ----------
    response_trial_names : :obj:`Iterable[str]` or :obj:`None`
        The codes identifying trials to include for response extraction.

    split_cue_as_instruction : :obj:`bool`
        Whether cue is being separated as instruction trial.

    block_cue_names : :obj:`Iterable[str]`
        The names of the block cues to remove from response trial codes.

    Returns
    -------
    Iterable[str] or None
        Filtered response trial codes with block cue names removed,
        or the original value if ``split_cue_as_instruction`` is False.
    """
    if response_trial_names and split_cue_as_instruction:
        response_trial_names = tuple(
            set(response_trial_names).difference(block_cue_names)
        )

    return response_trial_names


def _should_exclude_end_index(
    df: pd.DataFrame,
    block_end_index: int,
    trial_column_name: str,
    block_cue_names: Iterable[str],
    rest_block_codes: Iterable[str] | None,
    quit_code: str | None,
) -> bool:
    """
    Determine if ``block_end_index`` should be excluded from the block slice.
    The ``block_end_index`` is usually a boundary code (i.e. rest block, quit code,
    or next block cue) and should be excluded if it is. However, there are
    cases when log files end without any boundary codes (either due to how it was coded
    or due to a quit error), so the block_end_index should not be excluded.

    Parameters
    ----------
    df : :obj:`pd.DataFrame`
        DataFrame containing the log data.

    block_end_index : :obj:`int`
        The index returned by ``_get_next_block_index``, representing
        the boundary of the current block.

    trial_column_name : :obj:`str`
        Name of the column containing the trial types.

    block_cue_names : :obj:`Iterable[str]`
        The names of block cue codes that mark block boundaries.

    rest_block_codes : :obj:`Iterable[str]` or :obj:`None`
        The rest block code(s).

    quit_code : :obj:`str` or :obj:`None`
        The quit code.

    Returns
    -------
    bool
        If True, then ``block_end_index`` should be excluded by subtracting 1
        when extracting the block data and if False, the index should be included.
    """
    if block_end_index != df.index[-1]:
        return True

    end_code = df.loc[block_end_index, trial_column_name]
    boundary_codes = set(block_cue_names)
    boundary_codes.update(
        filter(None, combine_non_block_cue_names(rest_block_codes, quit_code))
    )

    return end_code in boundary_codes


def convert_to_literal(
    patterns: Iterable[str] | str | None, condition_series: pd.Series
):
    """
    Convert regex to literals.

    Parameters
    ----------
    patterns : :obj:`str`, obj:`Iterable[str]`, or obj:`None`
        A list or string containing regex or non-regex values.

    condition_series: :obj:`pd.Series`
        A Pandas series of the column containing the condition names.

    Return
    ------
    list[str] or None
        If ``patterns`` is not None, then returns a list of the condition or
        rest block names.
    """
    if not patterns:
        return None

    if isinstance(patterns, str):
        patterns = [patterns]

    patterns_str = "|".join(patterns)

    matches = condition_series.str.fullmatch(patterns_str, na=False)
    condition_names = condition_series[matches].unique().tolist()

    if not condition_names:
        raise ValueError(
            f"No matches found for patterns: {patterns}. "
            f"Available values: {condition_series.dropna().unique().tolist()}"
        )

    return condition_names


class LogExtractor(ABC):
    """Abstract Base Class for Extractors."""

    @abstractmethod
    def extract_onsets(self):
        """Extract onsets."""

    @abstractmethod
    def extract_durations(self):
        """Extract durations."""

    @abstractmethod
    def extract_trial_types(self):
        """Extract the trial types."""


class BlockExtractor(LogExtractor):
    """Abstract Base Class for Block Extractors."""

    @abstractmethod
    def extract_mean_reaction_times(self):
        """Extract mean reaction times for each block."""

    @abstractmethod
    def extract_mean_accuracies(self):
        """Extract mean accuracy for each block."""


class EventExtractor(LogExtractor):
    """Abstract Base Class for Event Extractors."""

    @abstractmethod
    def extract_reaction_times(self):
        """Extract reaction time for each trial."""


class PresentationExtractor:
    """
    Base class for Presentation log extractors.

    Provides shared initialization and extraction logic for both block
    and event design extractors.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        condition_codes: Iterable[str],
        scanner_event_type: str,
        scanner_trigger_code: str,
        trial_column_name: str = "Code",
        convert_to_seconds: list[str] | None = None,
        initial_column_headers: Iterable[str] = ("Trial", "Event Type"),
        n_discarded_volumes: int = 0,
        tr: float | int | None = None,
    ):

        df = _process_log_or_df(
            log_or_df,
            convert_to_seconds,
            initial_column_headers,
            divisor=1e4,
            software="Presentation",
        )
        self.trial_column_name = trial_column_name
        self.scanner_event_type = scanner_event_type
        self.scanner_trigger_code = scanner_trigger_code

        scanner_start_index_list = df.loc[
            (df["Event Type"] == self.scanner_event_type)
            & (df["Code"] == self.scanner_trigger_code)
        ].index.tolist()

        if scanner_start_index_list:
            scanner_start_index = scanner_start_index_list[0]
            self.scanner_start_time = df.loc[scanner_start_index, "Time"]
            df = df.loc[scanner_start_index:, :]
            self.df = df.reset_index(inplace=False, drop=True)
        else:
            LGR.warning(
                f"No scanner trigger under 'Event Type': {self.scanner_event_type} "
                f"and 'Code': {self.scanner_trigger_code} "
            )
            self.scanner_start_time = None
            self.df = df

        self.n_discarded_volumes = n_discarded_volumes
        self.tr = tr
        if self.n_discarded_volumes > 0:
            if not self.tr:
                raise ValueError(
                    "``tr`` must be provided when ``n_discarded_volumes`` is greater than 0."
                )

            if not self.scanner_start_time:
                raise ValueError(
                    "``scanner_start_time`` is None so time shift cannot be added."
                )

            self.scanner_start_time += self.n_discarded_volumes * self.tr

    def _process_scanner_start_time(
        self,
        scanner_start_time: float | int | None,
    ) -> list[float]:
        """Process scanner start time."""
        if scanner_start_time is not None:
            self.scanner_start_time = scanner_start_time

        if self.scanner_start_time is None:
            raise ValueError(
                "A value for `scanner_start_time` needs to be given "
                "since ``self.scanner_event_type`` and ``self.scanner_trigger_code`` "
                "did not identify a time."
            )


class PresentationBlockExtractor(PresentationExtractor, BlockExtractor):
    """
    Extract onsets, durations, and trial types from Presentation logs using a block design.

    Parameters
    ----------
    log_or_df : :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The Presentation log as a file path or the Presentation DataFrame
        returned by :code:`bidsaid.parsers.load_presentation_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    block_cue_names : :obj:`Iterable[str]`
        The names of the block cue names (i.e., "Face", "Place"). Serves to
        identify the boundaries of a trial type of interest. Regex can be used.

    trial_column_name : :obj:`str`, default="Code"
        Name of the column containing the trial types.

    scanner_event_type : :obj:`str`
        The event type in the "Event Type" column the scanner
        trigger is listed under (e.g., "Pulse", "Response", "Picture", etc).

    scanner_trigger_code : :obj:`str`
        Code listed under "Code" for the scanner start (e.g., "54", "99", "trigger).
        Used with ``scanner_event_type`` to compute the onset
        times of the trials relative to the scanner start time then
        clip the dataframe to ensure that no trials
        before the start of the scanner is initiated.

        .. note::
           Uses the first index of the rows in the dataframe with values
           provided for ``scanner_event_type`` and ``scanner_trigger_code``.

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1 ms to seconds.
        See `Presentation Timing <https://www.neurobs.com/pres_docs/html/04_reference/03_pcl_reference/03_stimulus_types/38_stimulus_data.htm>`_.

        .. important::
           Recommend time resolution of the "Time" and "Duration" column to be converted.

    initial_column_headers : :obj:`Iterable[str]`, default=("Trial", "Event Type")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes : :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
           - Only used when ``trigger_column_name`` is specified.
           - Only set this parameter if scanner trigger is sent **before** these volumes are
             acquired so that the start time of the first retained volume is shifted forward
             by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
             discarding the volumes, do not set this parameter.
             `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in ten thousand seconds if not converted.

    rest_block_codes : :obj:`str`, :obj:`Iterable[str]` or :obj:`None`, default=None
        The code name(s) for the rest block(s). Used when a resting state
        block is between the events to compute the correct block duration.
        If None, the block duration will be computed based on the starting
        index of the block cue codes given by ``block_cue_names``. If specified
        and ``rest_code_frequency`` is "variable", will be used with
        ``block_cue_names`` to compute the correct duration. Regex can be used.

    rest_code_frequency : :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest blocks. For "fixed", the rest code is assumed to
        appear between each block. For "variable", it is assumed that the rest
        code(s) inconsistently appears after each block.

    quit_code : :obj:`str` or :obj:`None`, default=None
        The quit code. Suggest to use in cases when a quit code, as opposed
        to a rest code, is preceded by a trial block. Ideally, this should
        be a unique code.

    split_cue_as_instruction : :obj:`bool`, default=False
        Whether to separate cue as instruction trial. If True, the row index
        containing the name of the block cue will have a separate onset and
        duration time. In addition, only the row indices proceeding the
        row index of the block cue will be used for computing the mean
        reaction time and accuracy for a block.

        .. important::
            Use this when your log structure has the following stucture, where
            ``block_cue_names`` are followed by their first stimulus:

            +-------+----------+
            | Row   | Code     |
            +=======+==========+
            | 0     | Face     |
            +-------+----------+
            | 1     | stim1    |
            +-------+----------+
            | 2     | response |
            +-------+----------+
            | 3     | stim2    |
            +-------+----------+
            | 4     | response |
            +-------+----------+
            | 5     | Face     |
            +-------+----------+
            | 6     | stim1    |
            +-------+----------+
            | ...   | ...      |
            +-------+----------+

            Do **not** use this when every row in the block shares the same code:

            +-------+----------+
            | 0     | Face     |
            +=======+==========+
            | 1     | response |
            +-------+----------+
            | 2     | Face     |
            +-------+----------+
            | 3     | response |
            +-------+----------+
            | ...   | ...      |
            +-------+----------+

    block_cues_without_instruction : :obj:`Iterable[str]` or :obj:`None`, default=None
        Block cue names that should not have their cue separated as instruction.
        Only used when ``split_cue_as_instruction`` is True. Blocks specified
        here will always start at the cue row.

    instruction_suffix : :obj:`str`, default="_instruction"
        Suffix for instruction trial type names.

    drop_instruction_cues : :obj:`bool`, defaut=False
        Whether to drop instruction cues from output.

    Attributes
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame containing the log data. If the scanner trigger is identified
        using ``scanner_event_type`` and ``scanner_trigger_code``, then rows
        preceding the first scanner are dropped and the index is reset.

    block_cue_names : :obj:`list[str]`
        The names of the block cue codes (i.e., Face, Place).

    scanner_event_type : :obj:`str`
        Event type of scanner trigger.

    scanner_trigger_code : :obj:`str`
        Code for the scanner trigger.

    trial_column_name : :obj:`str`
        Name of column containing the trial types.

    n_discarded_scans : :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time : :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    starting_block_indices : :obj:`list[int]`
        The indices of when each trial block of interest (specified by ``block_cue_names``)
        begins.

    rest_block_codes : obj:`list[str]` or :obj:`None`, default=None
        The rest block code(s).

    rest_code_frequency : :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest blocks.

    quit_code : :obj:`str` or :obj:`None`
        The quit code.

    split_cue_as_instruction : :obj:`bool`
        Whether to separate cue as instruction trial.

    block_cues_without_instruction : :obj:`Iterable[str]`
        Names of the blocks that should not have their cue separated
        as instruction. Regex can be used.

    instruction_suffix : :obj:`str`
        Suffix for instruction trial type names.

    drop_instruction_cues : :obj:`bool`
        Whether to drop instruction cues from output.

    Example
    -------
    >>> import pandas as pd
    >>> from bidsaid.events import PresentationBlockExtractor
    >>> extractor = PresentationBlockExtractor(
    ...     log_file,
    ...     block_cue_names=("Face", "Place"),
    ...     scanner_event_type="Pulse",
    ...     scanner_trigger_code="99",
    ...     convert_to_seconds=["Time"],
    ...     rest_block_codes="crosshair",
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations()
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> # Mean reaction time for all trials with a response
    >>> events["mean_rt"] = extractor.extract_mean_reaction_times()
    >>> # Mean reaction time for correct trials only
    >>> response_map = {"hit": 1, "miss": 0}
    >>> events["mean_rt_correct"] = extractor.extract_mean_reaction_times(
    ...     response_map=response_map,
    ...     response_type="correct",
    ... )
    >>> # Mean accuracy
    >>> events["mean_accuracy"] = extractor.extract_mean_accuracies(response_map=response_map)
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        block_cue_names: list[str],
        scanner_event_type: str,
        scanner_trigger_code: str,
        trial_column_name: str = "Code",
        convert_to_seconds: list[str] | None = None,
        initial_column_headers: Iterable[str] = ("Trial", "Event Type"),
        n_discarded_volumes: int = 0,
        tr: float | int | None = None,
        rest_block_codes: str | Iterable[str] | None = None,
        rest_code_frequency: Literal["fixed", "variable"] = "fixed",
        quit_code: str | None = None,
        split_cue_as_instruction: bool = False,
        block_cues_without_instruction: Iterable[str] | None = None,
        instruction_suffix: str = "_instruction",
        drop_instruction_cues: bool = False,
    ):
        super().__init__(
            log_or_df,
            block_cue_names,
            scanner_event_type,
            scanner_trigger_code,
            trial_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )

        self.block_cue_names = convert_to_literal(
            block_cue_names, self.df[self.trial_column_name]
        )

        assert rest_code_frequency in [
            "fixed",
            "variable",
        ], "`rest_code_frequency` must be either 'fixed' or 'variable'."

        self.rest_block_codes = convert_to_literal(
            rest_block_codes, self.df[self.trial_column_name]
        )
        self.rest_code_frequency = rest_code_frequency
        self.quit_code = quit_code

        self.starting_block_indices = _get_starting_block_indices(
            self.df, self.trial_column_name, self.block_cue_names
        )

        self.split_cue_as_instruction = split_cue_as_instruction
        self.block_cues_without_instruction = (
            convert_to_literal(
                block_cues_without_instruction, self.df[self.trial_column_name]
            )
            or ()
        )
        self.instruction_suffix = instruction_suffix
        self.drop_instruction_cues = drop_instruction_cues

        _validate_no_cue_conflicts(
            self.block_cue_names, self.rest_block_codes, self.quit_code
        )

    def extract_onsets(
        self, scanner_start_time: float | int | None = None
    ) -> list[float]:
        """
        Extract the onset times for each event.

        Onset is calculated as the difference between the event time and
        the scanner start time (e.g. first pulse).

        Parameters
        ----------
        scanner_start_time : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The start time for the scanner.

            .. important::
               Scanner start time will be detected during class initialization, unless
               the ``self.scanner_event_code`` and ``self.scanner_trigger_code`` does
               not return an index.

        Returns
        -------
        list[float]
            A list of onset times for each block.
        """
        self._process_scanner_start_time(scanner_start_time)

        onsets = []
        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.trial_column_name]
            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )

            row_shift = 1 if should_separate_block else 0
            block_start_time = (
                self.df.loc[block_start_index + row_shift, "Time"]
                - self.scanner_start_time
            )

            if should_separate_block and not self.drop_instruction_cues:
                cue_start_time = (
                    self.df.loc[block_start_index, "Time"] - self.scanner_start_time
                )

                onsets.extend([cue_start_time, block_start_time])
            else:
                onsets.append(block_start_time)

        return onsets

    def extract_durations(self) -> list[float]:
        """
        Extract the duration for each block.

        Duration is computed as the difference between the onset time of the
        current block and the onset time of the next block start of the block
        (either a rest block or some task block).

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        durations = []
        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.trial_column_name]
            block_end_index = _get_next_block_index(
                trial_series=self.df[self.trial_column_name],
                block_start_index=block_start_index,
                rest_block_codes=self.rest_block_codes,
                rest_code_frequency=self.rest_code_frequency,
                block_cue_names=self.block_cue_names,
                quit_code=self.quit_code,
            )

            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )

            row_shift = 1 if should_separate_block else 0
            block_start_time = self.df.loc[block_start_index + row_shift, "Time"]
            block_end_time = self.df.loc[block_end_index, "Time"]
            block_duration = block_end_time - block_start_time

            block_start_time = self.df.loc[block_start_index + row_shift, "Time"]
            if should_separate_block and not self.drop_instruction_cues:
                cue_start_time = self.df.loc[block_start_index, "Time"]
                cue_duration = block_start_time - cue_start_time

                durations.extend([cue_duration, block_duration])
            else:
                durations.append(block_duration)

        return durations

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each block.

        Returns
        -------
        list[str]
            A list of trial types for each block.
        """
        return _extract_block_trial_types(
            df=self.df,
            starting_block_indices=self.starting_block_indices,
            trial_column_name=self.trial_column_name,
            split_cue_as_instruction=self.split_cue_as_instruction,
            block_cues_without_instruction=self.block_cues_without_instruction,
            instruction_suffix=self.instruction_suffix,
            drop_instruction_cues=self.drop_instruction_cues,
        )

    def _get_block_trials(
        self,
        block_start_index: int,
        response_trial_names: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get trials within a block. Filtered by ``response_trial_names`` if not None.

        Parameters
        ----------
        block_start_index : :obj:`int`
            The starting index of the block.

        response_trial_names : :obj:`Iterable[str]` or :obj:`None`, default=None
            The stimulus trial names within each block to include for the
            mean accuracy and reaction time computation.

            .. important::
               If ``split_cue_as_instruction`` is True, block cue names
               are excluded from this parameter.

        Returns
        -------
        pandas.DataFrame
            DataFrame only containing the filtered block trials.
        """
        block_end_index = _get_next_block_index(
            trial_series=self.df[self.trial_column_name],
            block_start_index=block_start_index,
            rest_block_codes=self.rest_block_codes,
            rest_code_frequency=self.rest_code_frequency,
            block_cue_names=self.block_cue_names,
            quit_code=self.quit_code,
        )

        if _should_exclude_end_index(
            self.df,
            block_end_index,
            self.trial_column_name,
            self.block_cue_names,
            self.rest_block_codes,
            self.quit_code,
        ):
            block_end_index = block_end_index - 1

        block_df = self.df.loc[block_start_index:block_end_index, :]
        if self.split_cue_as_instruction:
            block_df = block_df.loc[(block_start_index + 1) :]

        if response_trial_names:
            block_df = block_df[
                block_df[self.trial_column_name].isin(response_trial_names)
            ]

        return block_df

    def _extract_rts_and_responses(
        self,
        block_df: pd.DataFrame,
    ) -> tuple[list[float], list[str]]:
        """
        Extract reaction times and responses for "Picture" events in a block.

        Parameters
        ----------
        block_df : :obj:`pandas.DataFrame`
            DataFrame containing the block trials (should be reset index).

        Returns
        -------
        tuple[list[float], list[str]]
            A tuple of (reaction_times, responses) for each Picture event.
        """
        picture_indices = block_df[block_df["Event Type"] == "Picture"].index
        reaction_times = []
        responses = []

        for row_indx in picture_indices:
            stimulus_row = block_df.loc[row_indx, :]
            trial_num = stimulus_row["Trial"]
            response_row = block_df[
                (block_df["Trial"] == trial_num)
                & (block_df["Event Type"] == "Response")
            ]

            response = stimulus_row["Stim Type"]
            if not response_row.empty:
                reaction_time = float(
                    response_row.iloc[0]["Time"] - stimulus_row["Time"]
                )
            else:
                reaction_time = np.nan

            reaction_times.append(reaction_time)
            responses.append(response)

        return reaction_times, responses

    def extract_mean_reaction_times(
        self,
        response_map: dict[str, int] | None = None,
        response_type: Literal["correct", "incorrect"] = "correct",
        response_trial_names: Iterable[str] | None = None,
    ) -> list[float]:
        """
        Extract mean reaction times for each block.

        Parameters
        ----------
        response_map : :obj:`dict[str, int]`
            A dictionary mapping response codes, from "Stim Type" column
            (ie. "hit", "miss", "other", "false alarm", "incorrect"), to accuracy
            values (0 for incorrect, 1 for correct). If provided, reaction times
            are filtered based on ``response_type``. If None, all reaction times
            with a response are included.

        response_type : :obj:`Literal["correct", "incorrect"]`, default="correct"
            Whether to compute mean RT for correct or incorrect trials.
            Only used when ``response_map`` is provided.

        response_trial_names : :obj:`Iterable[str]` or :obj:`None`, default=None
            The stimulus trial names within each block to include for the
            reaction_time computation.

            .. important::
               If ``split_cue_as_instruction`` is True, block cue names
               are excluded from this parameter.

        Returns
        -------
        list[float]
            A list of mean reaction times for each block.

        Note
        ----
        The reaction time is computed for the first response only. Also,
        if instruction cue is separated, NaN will be assigned for its reaction time.

        Example
        -------
        >>> # Mean RT for all trials with a response
        >>> mean_reaction_times = extractor.extract_mean_reaction_times()
        >>> # Mean RT for correct trials only
        >>> response_map = {"hit": 1, "miss": 0}
        >>> mean_reaction_times = extractor.extract_mean_reaction_times(
        ...     response_map=response_map,
        ...     response_type="correct",
        ... )
        """
        response_trial_names = _filter_response_trial_names(
            response_trial_names, self.split_cue_as_instruction, self.block_cue_names
        )

        mean_reaction_times = []
        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.trial_column_name]
            block_df = self._get_block_trials(block_start_index, response_trial_names)
            reaction_times, responses = self._extract_rts_and_responses(block_df)

            if response_map is not None:
                target_correctness = 1 if response_type == "correct" else 0
                filtered_rts = [
                    rt
                    for rt, resp in zip(reaction_times, responses)
                    if response_map.get(resp) == target_correctness
                ]
                mean_rt = (
                    np.nanmean(filtered_rts)
                    if len(filtered_rts) > 0 and not np.all(np.isnan(filtered_rts))
                    else np.nan
                )
            else:
                mean_rt = (
                    np.nanmean(reaction_times)
                    if len(reaction_times) > 0 and not np.all(np.isnan(reaction_times))
                    else np.nan
                )

            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )
            if should_separate_block and not self.drop_instruction_cues:
                mean_reaction_times.extend([np.nan, mean_rt])
            else:
                mean_reaction_times.append(mean_rt)

        return mean_reaction_times

    def extract_mean_accuracies(
        self,
        response_map: dict[str, int],
        response_trial_names: Iterable[str] | None = None,
    ) -> list[float]:
        """
        Extract mean accuracy for each block.

        Parameters
        ----------
        response_map : :obj:`dict[str, int]`
            A dictionary mapping response codes, from "Stim Type" column
            (ie. "hit", "miss", "other", "false alarm", "incorrect"), to accuracy
            values (0 for incorrect, 1 for correct).

        response_trial_names : :obj:`Iterable[str]` or :obj:`None`, default=None
            The stimulus trial names within each block to include for the
            accuracy computation.

            .. important::
               If ``split_cue_as_instruction`` is True, block cue names
               are excluded from this parameter.

        Returns
        -------
        list[float]
            A list of mean accuracies for each block.

        Note
        ----
        If instruction cue is separated, NaN will be assigned for its accuracy.

        Example
        -------
        >>> response_map = {"hit": 1, "miss": 0}
        >>> mean_accuracies = extractor.extract_mean_accuracies(response_map=response_map)
        """
        response_trial_names = _filter_response_trial_names(
            response_trial_names, self.split_cue_as_instruction, self.block_cue_names
        )

        mean_accuracies = []
        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.trial_column_name]
            block_df = self._get_block_trials(block_start_index, response_trial_names)
            _, responses = self._extract_rts_and_responses(block_df)

            converted_responses = [response_map[resp] for resp in responses]

            if len(converted_responses) > 0:
                mean_accuracy = sum(converted_responses) / len(converted_responses)
            else:
                mean_accuracy = np.nan

            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )
            if should_separate_block and not self.drop_instruction_cues:
                mean_accuracies.extend([np.nan, mean_accuracy])
            else:
                mean_accuracies.append(mean_accuracy)

        return mean_accuracies


class PresentationEventExtractor(PresentationExtractor, EventExtractor):
    """
    Extract onsets, durations, trial types, reaction times, and responses
    from Presentation logs using an event design.

    Parameters
    ----------
    log_or_df : :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The Presentation log as a file path or the Presentation DataFrame
        returned by :code:`bidsaid.parsers.load_presentation_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    trial_types : :obj:`Iterable[str]`
        The names of the trial types (i.e "congruentleft", "seen").
        Regex can be used.

        .. note::
           If your block design does not include a rest block or
           crosshair code, include the code immediately after the
           final block.

    scanner_event_type : :obj:`str`
        The event type in the "Event Type" column the scanner
        trigger is listed under (e.g., "Pulse", "Response", "Picture", etc).

    scanner_trigger_code : :obj:`str`
        Code listed under "Code" for the scanner start (e.g., "54", "99", "trigger).
        Used with ``scanner_event_type`` to compute the onset
        times of the trials relative to the scanner start time then
        clip the dataframe to ensure that no trials before the start
        of the scanner is initiated.

        .. note::
           Uses the first index of the rows in the dataframe with values
           provided for ``scanner_event_type`` and ``scanner_trigger_code``.

    trial_column_name : :obj:`str`, default="Code"
        Name of the column containing the trial types.

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from 0.1 ms to seconds.
        See `Presentation Timing <https://www.neurobs.com/pres_docs/html/04_reference/03_pcl_reference/03_stimulus_types/38_stimulus_data.htm>`_.

        .. important::
           Recommend time resolution of the "Time" column and "Duration" column
           to be converted.

    initial_column_headers : :obj:`Iterable[str]`, default=("Trial", "Event Type")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes : :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
           - Only used when ``trigger_column_name`` is specified.
           - Only set this parameter if scanner trigger is sent **before** these volumes are
             acquired so that the start time of the first retained volume is shifted forward
             by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
             discarding the volumes, do not set this parameter.
             `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in ten thousand seconds if not converted.

    Attributes
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame containing the log data. If the scanner trigger is identified
        using ``scanner_event_type`` and ``scanner_trigger_code``, then rows
        preceeding the first scanner are dropped and the index is reset.

    trial_types : :obj:`list[str]`
        The names of the trial types.

    scanner_event_type : :obj:`str`
        Event type of scanner trigger.

    scanner_trigger_code : :obj:`str`
        Code for the scanner trigger.

    trial_column_name : :obj:`str`
        Name of column containing the trial types.

    n_discarded_scans : :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr ``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    event_trial_indices : :obj:`list[int]`
        The indices of when each trial event of interest (specified by ``trial_types``)
        begins.

    Example
    -------
    >>> import pandas as pd
    >>> from bidsaid.events import PresentationEventExtractor
    >>> extractor = PresentationEventExtractor(
    ...     log_file,
    ...     trial_types=("congruentleft", "congruentright", "incongruentleft", "incongruentright", "nogo"),
    ...     scanner_event_type="Pulse",
    ...     scanner_trigger_code="99",
    ...     convert_to_seconds=["Time"],
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations()
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> events["reaction_time"] = extractor.extract_reaction_times()
    >>> events["response"] = extractor.extract_responses()
    >>> events["accuracy"] = extractor.extract_responses(response_map={"hit": 1, "miss": 0})
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        trial_types: Iterable[str],
        scanner_event_type: str,
        scanner_trigger_code: str,
        trial_column_name: str = "Code",
        convert_to_seconds: list[str] | None = None,
        initial_column_headers: Iterable[str] = ("Trial", "Event Type"),
        n_discarded_volumes: int = 0,
        tr: float | int | None = None,
    ):
        super().__init__(
            log_or_df,
            trial_types,
            scanner_event_type,
            scanner_trigger_code,
            trial_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )
        self.trial_types = convert_to_literal(
            trial_types, self.df[self.trial_column_name]
        )
        trial_series = self.df.loc[
            self.df[self.trial_column_name].isin(self.trial_types),
            self.trial_column_name,
        ]
        self.event_trial_indices = trial_series.index.tolist()

    def extract_onsets(
        self, scanner_start_time: float | int | None = None
    ) -> list[float]:
        """
        Extract the onset times for each event.

        Onset is calculated as the difference between the event time and
        the scanner start time (e.g. first pulse).

        Parameters
        ----------
        scanner_start_time : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The start time for the scanner.

            .. important::
               Scanner start time will be detected during class initialization, unless
               the ``self.scanner_event_code`` and ``self.scanner_trigger_code`` does
               not return an index.

        Returns
        -------
        list[float]
            A list of onset times for each event.
        """
        self._process_scanner_start_time(scanner_start_time)

        return [
            self.df.loc[index, "Time"] - self.scanner_start_time
            for index in self.event_trial_indices
        ]

    def _extract_rt_and_responses(self) -> tuple[list[float], list[str]]:
        """
        Extracts and reaction time and responses for each event.

        Reaction time is computed as the difference between the event stimulus
        and the response. When no response is given, the reaction is the
        difference between the starting time of that trial and the starting
        time of the subsequent stimuli.

        Returns
        -------
        tuple[list[float], list[str]]
            A tuple containing a list of durations and a list of responses.
        """
        reaction_times, responses = [], []
        for row_indx in self.event_trial_indices:
            row = self.df.loc[row_indx, :]
            trial_num = row["Trial"]
            response_row = self.df[
                (self.df["Trial"] == trial_num) & (self.df["Event Type"] == "Response")
            ]
            if not response_row.empty:
                reaction_time = float(response_row.iloc[0]["Time"] - row["Time"])
                response = row["Stim Type"]
            else:
                reaction_time = float("nan")
                response = row["Stim Type"]

            reaction_times.append(reaction_time)
            responses.append(response)

        return reaction_times, responses

    def extract_durations(self) -> list[float]:
        """
        Extract the duration for each event. Will extract the duration from the
        "Duration" column.

        Returns
        -------
        list[float]
            A list of durations for each event.
        """
        return self.df.loc[self.event_trial_indices, "Duration"].tolist()

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each event.

        Returns
        -------
        list[str]
            A list of trial types for each event.
        """
        return [
            self.df.loc[index, self.trial_column_name]
            for index in self.event_trial_indices
        ]

    def extract_reaction_times(self) -> list[float]:
        """
        Extract the reaction time for each event.

        .. note::
           - The reaction time is computed for the first response.
           - Will provide reaction time for all trials subject responds too.

        Reaction time is computed as the difference between the trial onset and the time
        the response if recorded. If no reponse is recorded, then the reaction time is NaN.
        """
        reaction_times, _ = self._extract_rt_and_responses()

        return reaction_times

    def extract_responses(self) -> list[str]:
        """
        Extract the response for each event.

        Returns
        -------
        list[str]
            A list of responses for each event.
        """
        _, responses = self._extract_rt_and_responses()

        return responses

    def extract_accuracies(
        self,
        response_map: dict[str, int | str],
    ) -> list[int] | list[str]:
        """
        Extract the accuracy (correct or incorrect) for each event.

        Parameters
        ----------
        response_map : :obj:`dict[str, int | str]`
            A dictionary mapping response codes, from "Stim Type" column
            (ie. "hit", "miss", "other", "false alarm", "incorrect"), to accuracy
            values (e.g., 0 for incorrect, 1 for correct).

        Returns
        -------
        list[int] or list[str]
            A list of accuracy values for each event (e.g., 0 = incorrect, 1 = correct).
        """
        _, responses = self._extract_rt_and_responses()

        return [response_map.get(response) for response in responses]


class EPrimeExtractor:
    """
    Base class for E-Prime 3 log extractors.

    Provides shared initialization and extraction logic for both block
    and event design extractors.
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        condition_codes: Iterable[str],
        onset_column_name: str,
        procedure_column_name: str,
        trigger_column_name: str | None = None,
        convert_to_seconds: list[str] | None = None,
        initial_column_headers: Iterable[str] = ("ExperimentName", "Subject"),
        n_discarded_volumes: int = 0,
        tr: float | int | None = None,
    ):

        self.df = _process_log_or_df(
            log_or_df,
            convert_to_seconds,
            initial_column_headers,
            divisor=1e3,
            software="E-Prime",
        )
        self.onset_column_name = onset_column_name
        self.procedure_column_name = procedure_column_name
        self.trigger_column_name = trigger_column_name

        if self.trigger_column_name:
            self.scanner_start_time = (
                self.df[self.trigger_column_name].dropna(inplace=False).unique()[0]
            )
        else:
            self.scanner_start_time = None

        self.n_discarded_volumes = n_discarded_volumes
        self.tr = tr
        if self.n_discarded_volumes > 0:
            if not self.tr:
                raise ValueError(
                    "``tr`` must be provided when ``n_discarded_volumes`` is greater than 0."
                )

            if not self.scanner_start_time:
                raise ValueError(
                    "``scanner_start_time`` is None so time shift cannot be added."
                )

            self.scanner_start_time += self.n_discarded_volumes * self.tr

    def _process_scanner_start_time(
        self,
        scanner_start_time: float | int | None,
    ) -> list[float]:
        """Process scanner start time."""
        if scanner_start_time is not None:
            self.scanner_start_time = scanner_start_time

        if self.scanner_start_time is None:
            raise ValueError(
                "``scanner_start_time`` must be provided either due to ``trigger_column_name`` not being provided "
                "or due to a non-NaN value could not be extracted from ``trigger_column_name``."
            )


class EPrimeBlockExtractor(EPrimeExtractor, BlockExtractor):
    """
    Extract onsets, durations, and trial types from E-Prime logs using a block design.

    Parameters
    ----------
    log_or_df : :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The E-Prime log as a file path or the E-Prime DataFrame
        returned by :code:`bidsaid.parsers.load_eprime_log`.

        .. important::
            If a text file is used, data are assumed to have at least one element
            that is an digit or float during parsing.

    block_cue_names : :obj:`Iterable[str]`
        The names of the block cue names (i.e., "Face", "Place"). Serves to
        identify the boundaries of a trial type of interest. Regex can be used.

    onset_column_name : :obj:`str`
        The name of the column containing stimulus onset time.

    procedure_column_name : :obj:`str`
        The name of the column containing the procedure names.

    trigger_column_name : :obj:`str` or :obj:`None`, default=None
        The name of the column containing the scanner start time.
        Uses the first value that is not NaN as the scanner start
        time. If None, the scanner start time will need to be
        given when using ``self.extract_onsets``.

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from milliseconds to seconds.
        See `EPrime Timing <http://support.pstnet.com/hc/en-us/articles/4414899313303-TIMING-E-Prime-clock-starting-point-34474>`_.

        .. important::
            Recommend time resolution of the columns containing the onset time, offset time (duration),
            reaction time, and scanner start time (``trigger_column_name``) be converted to seconds.

    initial_column_headers : :obj:`Iterable[str]`, default=("ExperimentName", "Subject")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes : :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
            - Only used when ``trigger_column_name`` is specified.
            - Only set this parameter if scanner trigger is sent **before** these volumes are
                acquired so that the start time of the first retained volume is shifted forward
                by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
                discarding the volumes, do not set this parameter.
                `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in milliseconds if not converted.

    rest_block_codes : :obj:`str`, :obj:`Iterable[str]`, or :obj:`None`, default=None
        The code name(s) for the rest block(s). Used when a resting state
        block is between the events to compute the correct block duration.
        If None, the block duration will be computed based on the starting
        index of the block cue codes given by ``block_cue_names``. If specified
        and ``rest_code_frequency`` is "variable", will be used with
        ``block_cue_names`` to compute the correct duration. Regex can be used.

     :obj:`Literal["fixed", "variable"]`, default="fixed"
        Frequency of the rest blocks. For "fixed", the rest code is assumed to
        appear between each block. For "variable", it is assumed that the rest
        code(s) inconsistently appears after each block.

    quit_code : :obj:`str` or :obj:`None`, default=None
        The quit code. Suggest to use in cases when a quit code, as opposed
        to a rest code, is preceded by a trial block. Ideally, this should
        be a unique code.

    split_cue_as_instruction : :obj:`bool`, default=False
        Whether to separate cue as instruction trial. If True, the row index
        containing the name of the block cue will have a separate onset and
        duration time. In addition, only the row indices proceeding the
        row index of the block cue will be used for computing the mean
        reaction time and accuracy for a block.

        .. important::
            Use this when your log structure has the following stucture, where
            ``block_cue_names`` are followed by their first stimulus:

            +-------+----------+
            | Row   | Code     |
            +=======+==========+
            | 0     | Face     |
            +-------+----------+
            | 1     | stim1    |
            +-------+----------+
            | 2     | stim2    |
            +-------+----------+
            | 3     | Face     |
            +-------+----------+
            | 4     | stim1    |
            +-------+----------+
            |...    | ...      |
            +-------+----------+


            Do **not** use this when every row in the block shares the same code:

            +-------+----------+
            | 0     | Face     |
            +=======+==========+
            | 1     | Face     |
            +-------+----------+
            | 2     | Face     |
            +-------+----------+
            | ...   | ...      |
            +-------+----------+

    block_cues_without_instruction : :obj:`Iterable[str]` or :obj:`None`, default=None
        Block cue names that should not have their cue separated as instruction.
        Only used when ``split_cue_as_instruction`` is True. Blocks specified
        here will always start at the cue row.

    instruction_suffix : :obj:`str`, default="_instruction"
        Suffix for instruction trial type names.

    drop_instruction_cues : :obj:`bool`, defaut=False
        Whether to drop instruction cues from output.

    Attributes
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame containing the log data.

    block_cue_names : :obj:`list[str]`
        The names of the blocks.

    onset_column_name : :obj:`str`
        Name of column containing the onset time.

    procedure_column_name : :obj:`str`
        Name of column containing the procedure names.

    trigger_column_name : :obj:`str` or :obj:`None`
        Name of column containing time when scanner sent pulse/scanner start time.

    n_discarded_scans : :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time : :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    starting_block_indices : :obj:`list[int]`
        The indices of when each trial block of interest (specified by ``block_cue_names``)
        begins.

    rest_block_codes : :obj:`list[str]` or :obj:`None`
        The rest block code(s).

    rest_code_frequency : :obj:`Literal["fixed", "variable"]`
        Frequency of the rest blocks.

    quit_code : :obj:`str` or :obj:`None`
        The quit code.

    split_cue_as_instruction : :obj:`bool`
        Whether to separate cue as instruction trial. Will compute timing information
        from the row index immediately proceeding the row index of the block cue.

    block_cues_without_instruction : :obj:`Iterable[str]` or :obj:`None`
        Names of the blocks that should not have their cue separated
        as instruction. Regex can be used.

    instruction_suffix : :obj:`str`
        Suffix for instruction trial type names.

    drop_instruction_cues : :obj:`bool`
        Whether to drop instruction cues from output.

    Example
    -------
    >>> import pandas as pd
    >>> from bidsaid.events import EPrimeBlockExtractor
    >>> extractor = EPrimeBlockExtractor(
    ...     log_file,
    ...     block_cue_names=("Face", "Place"),
    ...     onset_column_name="Stimulus.OnsetTime",
    ...     procedure_column_name="Procedure",
    ...     trigger_column_name="ScannerTrigger.RTTime",
    ...     convert_to_seconds=["Stimulus.OnsetTime", "ScannerTrigger.RTTime", "Stimulus.RT"],
    ...     rest_block_codes="Rest",
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations(offset_column_name="Stimulus.OffsetTime")
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> # Mean reaction time for correct Face trials only
    >>> events["mean_rt"] = extractor.extract_mean_reaction_times(
    ...     reaction_time_column_name="Stimulus.RT",
    ...     subject_response_column="Stimulus.RESP",
    ...     correct_response_column="Simulus.CRESP",
    ...     response_type="correct",
    ...     response_trial_names=("Face")
    ... )
    >>> # Mean accuracy across all trial types
    >>> events["mean_accuracy"] = extractor.extract_mean_accuracies(
    ...     subject_response_column="Stimulus.RESP",
    ...     correct_response_column="Stimulus.CRESP",
    ... )
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        block_cue_names: Iterable[str],
        onset_column_name: str,
        procedure_column_name: str,
        trigger_column_name: str | None = None,
        convert_to_seconds: list[str] | None = None,
        initial_column_headers: Iterable[str] = ("ExperimentName", "Subject"),
        n_discarded_volumes: int = 0,
        tr: float | int | None = None,
        rest_block_codes: str | Iterable[str] | None = None,
        rest_code_frequency: Literal["fixed", "variable"] = "fixed",
        quit_code: str | None = None,
        split_cue_as_instruction: bool = False,
        block_cues_without_instruction: Iterable[str] | None = None,
        instruction_suffix: str = "_instruction",
        drop_instruction_cues: bool = False,
    ):
        super().__init__(
            log_or_df,
            block_cue_names,
            onset_column_name,
            procedure_column_name,
            trigger_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )

        self.block_cue_names = convert_to_literal(
            block_cue_names, self.df[self.procedure_column_name]
        )

        assert rest_code_frequency in [
            "fixed",
            "variable",
        ], "`rest_code_frequency` must be either 'fixed' or 'variable'."

        self.rest_block_codes = convert_to_literal(
            rest_block_codes, self.df[self.procedure_column_name]
        )
        self.rest_code_frequency = rest_code_frequency
        self.quit_code = quit_code

        self.starting_block_indices = _get_starting_block_indices(
            self.df, self.procedure_column_name, self.block_cue_names
        )

        self.split_cue_as_instruction = split_cue_as_instruction
        self.block_cues_without_instruction = (
            convert_to_literal(
                block_cues_without_instruction, self.df[self.procedure_column_name]
            )
            or ()
        )
        self.instruction_suffix = instruction_suffix
        self.drop_instruction_cues = drop_instruction_cues

        _validate_no_cue_conflicts(
            self.block_cue_names, self.rest_block_codes, self.quit_code
        )

    def extract_onsets(
        self, scanner_start_time: float | int | None = None
    ) -> list[float]:
        """
        Extract the onset times for each block.

        Onset is calculated as the difference between the event time and
        the scanner start time.

        Parameters
        ----------
        scanner_start_time : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The scanner start time. Used to compute onset relative to
            the start of the scan.

            .. note:: Does not need to be given if ``trigger_column_name`` was provided.

        Returns
        -------
        list[float]
            A list of onset times for each block.
        """
        self._process_scanner_start_time(scanner_start_time)

        onsets = []
        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.procedure_column_name]
            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )

            row_shift = 1 if should_separate_block else 0
            block_start_time = (
                self.df.loc[block_start_index + row_shift, self.onset_column_name]
                - self.scanner_start_time
            )

            if row_shift and not self.drop_instruction_cues:
                cue_start_time = (
                    self.df.loc[block_start_index, self.onset_column_name]
                    - self.scanner_start_time
                )

                onsets.extend([cue_start_time, block_start_time])
            else:
                onsets.append(block_start_time)

        return onsets

    def extract_durations(self, offset_column_name: str | None = None) -> list[float]:
        """
        Extract the duration for each block.

        Duration is computed in one of two ways:

        If ``offset_column_name`` is None, then duration is computed as the difference between
        the onset time of the current block and the onset time of the next block start of the block
        (either a rest block or some task block).

        If ``offset_column_name`` is not None, duration is computed by subtracting the final
        stimulus offset time (considered to be the index prior to the start of the next block,
        which may be a rest block or a different task block) by the onset time of
        the start of the block.

        Note, if there is no final quit code or rest block for the final task block, then the
        block ends at the final index of the DataFrame.

        Parameters
        ----------
        offset_column_name : :obj:`str`, default=None
            The name of the stimulus offset column.

        Returns
        -------
        list[float]
            A list of durations for each block.
        """
        durations = []
        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.procedure_column_name]
            block_end_index = _get_next_block_index(
                trial_series=self.df[self.procedure_column_name],
                block_start_index=block_start_index,
                rest_block_codes=self.rest_block_codes,
                rest_code_frequency=self.rest_code_frequency,
                block_cue_names=self.block_cue_names,
                quit_code=self.quit_code,
            )

            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )
            row_shift = 1 if should_separate_block else 0
            block_start_time = self.df.loc[
                block_start_index + row_shift, self.onset_column_name
            ]
            if offset_column_name:
                if _should_exclude_end_index(
                    self.df,
                    block_end_index,
                    self.procedure_column_name,
                    self.block_cue_names,
                    self.rest_block_codes,
                    self.quit_code,
                ):
                    block_end_index = block_end_index - 1

                block_end_time = self.df.loc[block_end_index, offset_column_name]
            else:
                block_end_time = self.df.loc[block_end_index, self.onset_column_name]

            block_duration = block_end_time - block_start_time

            if should_separate_block and not self.drop_instruction_cues:
                cue_start_time = self.df.loc[block_start_index, self.onset_column_name]
                cue_duration = block_start_time - cue_start_time

                durations.extend([cue_duration, block_duration])
            else:
                durations.append(block_duration)

        return durations

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each block.

        Returns
        -------
        list[str]
            A list of trial types for each block.
        """
        return _extract_block_trial_types(
            df=self.df,
            starting_block_indices=self.starting_block_indices,
            trial_column_name=self.procedure_column_name,
            split_cue_as_instruction=self.split_cue_as_instruction,
            block_cues_without_instruction=self.block_cues_without_instruction,
            instruction_suffix=self.instruction_suffix,
            drop_instruction_cues=self.drop_instruction_cues,
        )

    def _get_block_trials(
        self,
        block_start_index: int,
        response_trial_names: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get trials within a block. Filtered by ``response_trial_names`` if not None.

        Parameters
        ----------
        block_start_index : :obj:`int`
            The starting index of the block.

        response_trial_names : :obj:`Iterable[str]` or :obj:`None`, default=None
            The stimulus trial names within each block to include for the
            mean accuracy and reaction time computation.

            .. important::
               If ``split_cue_as_instruction`` is True, block cue names
               are excluded from this parameter.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the filtered block trials.
        """
        block_end_index = _get_next_block_index(
            trial_series=self.df[self.procedure_column_name],
            block_start_index=block_start_index,
            rest_block_codes=self.rest_block_codes,
            rest_code_frequency=self.rest_code_frequency,
            block_cue_names=self.block_cue_names,
            quit_code=self.quit_code,
        )

        if _should_exclude_end_index(
            self.df,
            block_end_index,
            self.procedure_column_name,
            self.block_cue_names,
            self.rest_block_codes,
            self.quit_code,
        ):
            block_end_index = block_end_index - 1

        block_df = self.df.loc[block_start_index:block_end_index, :]
        if self.split_cue_as_instruction:
            block_df = block_df.loc[(block_start_index + 1) :]

        if response_trial_names:
            block_df = block_df[
                block_df[self.procedure_column_name].isin(response_trial_names)
            ]

        return block_df

    def _compute_trial_correctness(
        self,
        block_df: pd.DataFrame,
        subject_response_column: str,
        correct_response_column: str,
        response_required_only: bool = False,
    ) -> pd.Series:
        """
        Compute correctness for each trial in a block.

        A trial is correct if:
        - Subject response matches correct response, OR
        - Both are NaN (correct withhold on no-go trial)

        Parameters
        ----------
        block_df : :obj:`pd.DataFrame`
            DataFrame containing block trials.

        subject_response_column : :obj:`str`
            Column name for subject's response.

        correct_response_column : :obj:`str`
            Column name for correct response.

        response_required_only : :obj:`bool`, default=False
            Compute accuracy only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        tuple[pandas.Series, pandas.DataFrame]
            Boolean series indicating correctness for each trial and the original or
            filtered ``block_df``.
        """

        subject_resp = block_df[subject_response_column].apply(
            lambda x: float(x) if _is_float(x) else x
        )
        correct_resp = block_df[correct_response_column].apply(
            lambda x: float(x) if _is_float(x) else x
        )

        if response_required_only:
            block_df = block_df[~correct_resp.isna()]
            subject_resp = subject_resp[~correct_resp.isna()]
            correct_resp = correct_resp[~correct_resp.isna()]

        both_nan = subject_resp.isna() & correct_resp.isna()
        both_equal = subject_resp == correct_resp

        return both_nan | both_equal, block_df

    def extract_mean_reaction_times(
        self,
        reaction_time_column_name: str,
        subject_response_column: str,
        correct_response_column: str,
        response_type: Literal["correct", "incorrect"] = "correct",
        response_trial_names: Iterable[str] | None = None,
        response_required_only: bool = False,
    ) -> list[float]:
        """
        Extract mean reaction times for each block.

        Computes mean reaction time filtered for correct or incorrect trials.

        Parameters
        ----------
        reaction_time_column_name : :obj:`str`
            The name of the column containing reaction time values.
            Usually the column name ending in ".RT" not the column
            ending in ".RTTime".

        subject_response_column : :obj:`str`
            The name of the column containing the subject's response.
            Usually the column name ending in ".RESP".

        correct_response_column : :obj:`str`
            The name of the column containing the correct response.
            For no-go trials, this should be empty/NaN.

        response_type : :obj:`Literal["correct", "incorrect"]`, default="correct"
            Whether to compute mean reaction time for correct or incorrect trials.

        response_trial_names : :obj:`Iterable[str]` or :obj:`None`, default=None
            The stimulus trial names within each block to include for the
            mean reaction time computation.

            .. important::
               If ``split_cue_as_instruction`` is True, block cue names
               are excluded from this parameter.

        response_required_only : :obj:`bool`, default=False
            Compute reaction times only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        list[float]
            A list of mean reaction times for each block. Returns NaN for
            blocks with no valid reaction times.

        Note
        ----
        Correctness is determined by comparing ``subject_response_column`` to
        ``correct_response_column``:

        - A trial is **correct** if the responses match, OR if both are NaN
          (e.g. correct withhold on a no-go trial).
        - A trial is **incorrect** if they differ (wrong response, miss, or
          false alarm).

        Trials with NaN reaction times (due to filtering) are excluded via ``np.nanmean``.
        Also, if instruction cue is separated, NaN will be assigned for its reaction time.

        Example
        -------
        >>> # Get mean reaction time for correct Go trials only
        >>> mean_reaction_times = extractor.extract_mean_reaction_times(
        ...     reaction_time_column_name="Stimulus.RT",
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="CorrectResponse",
        ...     response_type="correct",
        ...     response_trial_names=("Go")
        ... )
        """
        target_correctness = response_type == "correct"
        mean_reaction_times = []

        response_trial_names = _filter_response_trial_names(
            response_trial_names, self.split_cue_as_instruction, self.block_cue_names
        )

        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.procedure_column_name]
            block_df = self._get_block_trials(block_start_index, response_trial_names)

            correctness, block_df = self._compute_trial_correctness(
                block_df,
                subject_response_column,
                correct_response_column,
                response_required_only,
            )

            if not correctness.empty:
                filtered_rts = block_df.loc[
                    correctness == target_correctness, reaction_time_column_name
                ]
                mean_rt = (
                    np.nanmean(filtered_rts)
                    if filtered_rts.size > 0 and not np.all(np.isnan(filtered_rts))
                    else np.nan
                )
            else:
                mean_rt = np.nan

            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )
            if should_separate_block and not self.drop_instruction_cues:
                mean_reaction_times.extend([np.nan, mean_rt])
            else:
                mean_reaction_times.append(mean_rt)

        return mean_reaction_times

    def extract_mean_accuracies(
        self,
        subject_response_column: str,
        correct_response_column: str,
        response_trial_names: Iterable[str] | None = None,
        response_required_only: bool = False,
    ) -> list[float]:
        """
        Extract mean accuracy for each block.

        Computes accuracy by comparing the subject's response to the correct
        response for each trial.

        Parameters
        ----------
        subject_response_column : :obj:`str`
            The name of the column containing the subject's response.
            Usually the column name ending in ".RESP".

        correct_response_column : :obj:`str`
            The name of the column containing the correct response. Trials
            where the subject should not respond should be NaN.
            Usually column name ending in ".CRESP".

        response_trial_names : :obj:`Iterable[str]` or :obj:`None`, default=None
            The stimulus trial names within each block to include for the
            accuracy computation.

            .. important::
               If ``split_cue_as_instruction`` is True, block cue names
               are excluded from this parameter.

        response_required_only : :obj:`bool`, default=False
            Compute accuracy only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        list[float]
            A list of mean accuracies for each block.

        Notes
        -----
        Correctness is determined by comparing ``subject_response_column`` to
        ``correct_response_column`` using the following logic:

        +------------------+-------------------+--------------------------+----------+
        | Subject Response | Correct Response  | Interpretation           | Accuracy |
        +==================+===================+==========================+==========+
        | "1"              | "1"               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | NaN               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | "1"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "1"              | NaN               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "2"              | "3"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+

        Note: If ``response_required_only`` is True, only accuracy for the first,
        third, and fifth row are computed since those are trials that expect a response.

        Also, if instruction cue is separated, NaN will be assigned for its accuracy.

        Example
        -------
        >>> # Get mean accuracy for all trial types
        >>> mean_accuracies = extractor.extract_mean_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="CorrectResponse"
        ... )
        >>> # Get mean accuracy for specific trial types only
        >>> mean_accuracies = extractor.extract_mean_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="CorrectResponse",
        ...     response_trial_names=("Go", "NoGo")
        ... )

        Note
        ----
        Mean accuracy for instruction cues will be NaN.
        """
        mean_accuracies = []
        for block_start_index in self.starting_block_indices:
            block_cue_name = self.df.loc[block_start_index, self.procedure_column_name]
            block_df = self._get_block_trials(block_start_index, response_trial_names)

            correctness, _ = self._compute_trial_correctness(
                block_df,
                subject_response_column,
                correct_response_column,
                response_required_only,
            )

            should_separate_block = _separate_block_from_cue(
                self.split_cue_as_instruction,
                block_cue_name,
                self.block_cues_without_instruction,
            )
            if should_separate_block and not self.drop_instruction_cues:
                mean_accuracies.extend([np.nan, correctness.mean()])
            else:
                mean_accuracies.append(correctness.mean())

        return mean_accuracies


class EPrimeEventExtractor(EPrimeExtractor, EventExtractor):
    """
    Extract onsets, durations, trial types, reaction times, and responses
    from E-Prime logs using an event design.

    Parameters
    ----------
    log_or_df : :obj:`str`, :obj:`Path`, :obj:`pandas.DataFrame`
        The E-Prime log as a file path or the E-Prime DataFrame
        returned by :code:`bidsaid.parsers.load_eprime_log`.

        .. important::
           If a text file is used, data are assumed to have at least one element
           that is an digit or float during parsing.

    trial_types : :obj:`Iterable[str]`
        The names of the trial types (i.e "congruentleft", "seen"). Regex can be used.

        .. note::
            Depending on the way your E-Prime data is structured, for block
            design the rest block may have to be included as a "trial_type"
            to compute the correct duration. These rows can then be dropped
            from the events DataFrame.

    onset_column_name : :obj:`str`
        The name of the column containing stimulus onset time.

    procedure_column_name : :obj:`str`
        The name of the column containing the procedure names.

    trigger_column_name : :obj:`str` or :obj:`None`, default=None
        The name of the column containing the scanner start time.
        Uses the first value that is not NaN as the scanner start
        time. If None, the scanner start time will need to be
        given when using ``self.extract_onsets``.

    convert_to_seconds : :obj:`list[str]` or :obj:`None`, default=None
        Convert the time resolution of the specified columns from milliseconds to seconds.
        See `EPrime Timing <http://support.pstnet.com/hc/en-us/articles/4414899313303-TIMING-E-Prime-clock-starting-point-34474>`_.

        .. important::
           Recommend time resolution of the columns containing the onset times,
           offset times (duration), reaction times, and scanner onset time (``trigger_column_name``)
           be converted to seconds.

    initial_column_headers : :obj:`Iterable[str]`, default=("ExperimentName", "Subject")
        The initial column headers for data. Only used when
        ``log_or_df`` is a file path.

    n_discarded_volumes : :obj:`int`, default=0
        Number of non-steady state scans discarded by the scanner at the start of the sequence.

        .. important::
           - Only used when ``trigger_column_name`` is specified.
           - Only set this parameter if scanner trigger is sent **before** these volumes are
             acquired so that the start time of the first retained volume is shifted forward
             by (``n_discarded_volumes * tr``). If the scanner sends trigger **after**
             discarding the volumes, do not set this parameter.
             `Explanation from Neurostars <https://neurostars.org/t/how-to-sync-time-from-e-prime-behavior-data-with-fmri/6887>`_.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        The repetition time provided in seconds if data was converted to seconds or
        in milliseconds if not converted.

    Attributes
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame containing the log data.

    trial_types : :obj:`Iterable[str]`
        The names of the trial types.

    onset_column_name : :obj:`str`
        Name of column containing the onset time.

    procedure_column_name : :obj:`str`
        Name of column containing the trial types.

    trigger_column_name : :obj:`str` or :obj:`None`
        Name of column containing time when scanner sent pulse/scanner start time.

    n_discarded_scans : :obj:`int`
        Number of non-steady state scans discarded by scanner.

    tr : :obj:`float`, :obj:`int`, or :obj:`None`
        The repetition time.

    scanner_start_time : :obj:`float` or :obj:`None`
        Time when scanner sends the pulse. If ``n_discarded_volumes``
        is not 0 and ``tr`` is specified, then this time will be
        shifted forward (``scanner_start_time = scanner_start_time + n_discarded_volumes * tr``)
        to reflect the time when the first steady state volume was retained. Otherwise, the time
        extracted from the log data is assumed to be the time when the first steady state
        volume was retained.

    event_trial_indices : :obj:`list[int]`
        The indices of when each trial event of interest (specified by ``trial_types``)
        begins.

    Example
    -------
    >>> import pandas as pd
    >>> from bidsaid.events import EPrimeEventExtractor
    >>> extractor = EPrimeEventExtractor(
    ...     log_file,
    ...     trial_types=("Go", "NoGo"),
    ...     onset_column_name="Stimulus.OnsetTime",
    ...     procedure_column_name="Procedure",
    ...     trigger_column_name="ScannerTrigger.RTTime",
    ...     convert_to_seconds=[
    ...         "Stimulus.OnsetTime",
    ...         "Stimulus.OffsetTime",
    ...         "Stimulus.RT",
    ...         "ScannerTrigger.RTTime"
    ...     ],
    ... )
    >>> events = {}
    >>> events["onset"] = extractor.extract_onsets()
    >>> events["duration"] = extractor.extract_durations(offset_column_name="Stimulus.OffsetTime")
    >>> events["trial_type"] = extractor.extract_trial_types()
    >>> events["reaction_time"] = extractor.extract_reaction_times(reaction_time_column_name="Stimulus.RT")
    >>> events["accuracy"] = extractor.extract_accuracies(
    ...     subject_response_column="Stimulus.RESP",
    ...     correct_response_column="CorrectResponse",
    ... )
    >>> df = pd.DataFrame(events)
    """

    def __init__(
        self,
        log_or_df: str | Path | pd.DataFrame,
        trial_types: Iterable[str],
        onset_column_name: str,
        procedure_column_name: str,
        trigger_column_name: str | None = None,
        convert_to_seconds: str | None = None,
        initial_column_headers: Iterable[str] = ("ExperimentName", "Subject"),
        n_discarded_volumes: int = 0,
        tr: float | int | None = None,
    ):
        super().__init__(
            log_or_df,
            trial_types,
            onset_column_name,
            procedure_column_name,
            trigger_column_name,
            convert_to_seconds,
            initial_column_headers,
            n_discarded_volumes,
            tr,
        )

        self.trial_types = convert_to_literal(
            trial_types, self.df[self.procedure_column_name]
        )
        trial_series = self.df.loc[
            self.df[self.procedure_column_name].isin(self.trial_types),
            self.procedure_column_name,
        ]
        self.event_trial_indices = trial_series.index.tolist()

    def extract_onsets(
        self,
        scanner_start_time: float | int | None = None,
    ) -> list[float]:
        """
        Extract the onset times for each event.

        Onset is calculated as the difference between the event time and
        the scanner start time.

        Parameters
        ----------
        scanner_start_time : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
            The scanner start time. Used to compute onset relative to
            the start of the scan.

            .. note:: Does not need to be given if ``trigger_column_name`` was set.

        Returns
        -------
        list[float]
            A list of onset times for each event.
        """
        self._process_scanner_start_time(scanner_start_time)

        return [
            self.df.loc[index, self.onset_column_name] - self.scanner_start_time
            for index in self.event_trial_indices
        ]

    def extract_durations(self, offset_column_name: str) -> list[float]:
        """
        Extract the duration for each event.

        Duration is computed as the difference between the trial onset
        time and the trial offset time.

        Parameters
        ----------
        offset_column_name : :obj:`str`
            The name of the column containing the offset time of trial.

        Returns
        -------
        list[float]
            A list of durations for each event.
        """
        return [
            self.df.loc[index, offset_column_name]
            - self.df.loc[index, self.onset_column_name]
            for index in self.event_trial_indices
        ]

    def extract_trial_types(self) -> list[str]:
        """
        Extract the trial type for each event.

        Returns
        -------
        list[str]
            A list of trial types for each event.
        """
        return [
            self.df.loc[index, self.procedure_column_name]
            for index in self.event_trial_indices
        ]

    def extract_reaction_times(self, reaction_time_column_name: str) -> list[float]:
        """
        Extract the reaction time for each event.

        Parameters
        ----------
        reaction_time_column_name : :obj:`str`
            The name of the column containing reaction time values.
            Usually the column name ending in ".RT".

        Returns
        -------
        list[float]
            A list of reaction times for each event. NaN for trials where
            the subject did not respond.

        Notes
        -----
        This function returns all reaction times as-is, regardless of whether
        the trial was correct or incorrect, or whether the subject was expected
        to respond or not. Filter the resulting DataFrame based on trial type and
        response if needed.

        Example
        -------
        >>> rts = extractor.extract_reaction_times("Stimulus.RT")
        >>> responses = extractor.extract_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="Stimulus.CRESP",
        ... )
        >>> # Filter to only correct Go trials:
        >>> df = pd.DataFrame({"rt": rts, "trial_type": trial_types, "response": responses})
        >>> correct_go_rts = df[(df["trial_type"] == "Go") & (df["response"] == 1)]["rt"]
        """
        return [
            self.df.loc[index, reaction_time_column_name]
            for index in self.event_trial_indices
        ]

    def extract_accuracies(
        self,
        subject_response_column: str,
        correct_response_column: str,
        response_required_only: bool = False,
    ) -> list[int]:
        """
        Extract the accuracy (correct or incorrect) for each event.

        Parameters
        ----------
        subject_response_column : :obj:`str`
            The name of the column containing the subject's response.
            Usually the column name ending in ".RESP".

        correct_response_column : :obj:`str`
            The name of the column containing the correct response. Trials
            where the subject should not respond should be NaN.
            Usually column name ending in ".CRESP".

        response_required_only : :obj:`bool`, default=False
            Compute accuracy only for trials expecting a response.
            Non-response trials are assumed to be assigned NaN
            in ``correct_response_column``.

        Returns
        -------
        list[int]
            A list of accuracy values for each event (0 = incorrect, 1 = correct,
            NaN = trial did not require response when ``response_required_only=True``).

        Notes
        -----
        Correctness is determined by comparing ``subject_response_column`` to
        ``correct_response_column``:

        +------------------+-------------------+--------------------------+----------+
        | Subject Response | Correct Response  | Interpretation           | Accuracy |
        +==================+===================+==========================+==========+
        | "1"              | "1"               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | NaN               | Correct response         | 1        |
        +------------------+-------------------+--------------------------+----------+
        | NaN              | "1"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "1"              | NaN               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+
        | "2"              | "3"               | Incorrect response       | 0        |
        +------------------+-------------------+--------------------------+----------+

        Note: If ``response_required_only`` is True, only accuracy for the first,
        third, and fifth row are assigned 0 and 1, the second and fourth trials as NaN
        since those are trials that expect a response.

        Example
        -------
        >>> responses = extractor.extract_accuracies(
        ...     subject_response_column="Stimulus.RESP",
        ...     correct_response_column="Stimulus.CRESP",
        ... )
        """
        responses = []
        for row_indx in self.event_trial_indices:
            subject_resp = self.df.loc[row_indx, subject_response_column]
            correct_resp = self.df.loc[row_indx, correct_response_column]

            subject_resp = (
                float(subject_resp) if _is_float(subject_resp) else subject_resp
            )
            correct_resp = (
                float(correct_resp) if _is_float(correct_resp) else correct_resp
            )

            both_nan = pd.isna(subject_resp) and pd.isna(correct_resp)
            both_equal = subject_resp == correct_resp

            if response_required_only and pd.isna(correct_resp):
                response = np.nan
            else:
                response = 1 if (both_nan or both_equal) else 0

            responses.append(response)

        return responses


def add_instruction_timing(
    event_df: pd.DataFrame,
    instruction_duration: int | float,
    instruction_suffix: str = "_instruction",
    exclude_trial_types: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Add instruction trials to a BIDS compliant events DataFrame.

    Splits each trial into an instruction period and a task period by creating
    a new instruction row before each trial. The instruction row has the specified
    duration, and the task row's onset is shifted forward and duration reduced
    accordingly.

    Parameters
    ----------
    event_df : :obj:`pd.DataFrame`
        A BIDS compliant events DataFrame. Must contain "onset", "duration",
        and "trial_type" columns.

    instruction_duration : :obj:`int` or :obj:`float`
        Duration of the instruction trial to add before each trial type.

    instruction_suffix : :obj:`str`, default="_instruction"
        Suffix to append to trial type names for instruction trials.

    exclude_trial_types : :obj:`Iterable[str]` or :obj:`None`, default=None
        Trial types to exclude from instruction splitting. These trials
        will be kept as-is without an instruction trial added. Regex can
        be used.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with instruction trials added, sorted by onset.

    Note
    ----
    For instruction trials, all columns other than "onset", "duration", and
    "trial_type" are set to NaN.

    Example
    -------
    >>> import pandas as pd
    >>> from bidsaid.events import add_instruction_timing
    >>> events_df = pd.DataFrame({
    ...     "onset": [0.0, 10.0, 20.0],
    ...     "duration": [10.0, 10.0, 10.0],
    ...     "trial_type": ["Face", "Place", "Rest"],
    ... })
    >>> new_events_df = add_instruction_timing(
    ...     events_df,
    ...     instruction_duration=2.0,
    ...     exclude_trial_types=["Rest"],
    ... )
    >>> print(new_events_df)
    """
    required_columns = ("onset", "duration", "trial_type")
    if not all(col in event_df.columns for col in required_columns):
        raise ValueError(
            f"`event_df` must contain the following columns: {iterable_to_str(required_columns)}."
        )

    exclude_trial_types = (
        convert_to_literal(exclude_trial_types, event_df["trial_type"]) or []
    )

    mask = ~event_df["trial_type"].isin(exclude_trial_types)
    trials_to_split = event_df[mask].copy(deep=True)
    trials_to_keep = event_df[~mask].copy(deep=True)

    instructions = trials_to_split.copy(deep=True)
    instructions["duration"] = instruction_duration
    instructions["trial_type"] = instructions["trial_type"] + instruction_suffix
    instructions[trials_to_split.columns.difference(required_columns)] = float("nan")

    tasks = trials_to_split.copy(deep=True)
    tasks["onset"] = tasks["onset"] + instruction_duration
    tasks["duration"] = tasks["duration"] - instruction_duration

    new_df = pd.concat([instructions, tasks, trials_to_keep], ignore_index=True)

    return new_df.sort_values(by="onset").reset_index(inplace=False, drop=True)


__all__ = [
    "add_instruction_timing",
    "LogExtractor",
    "BlockExtractor",
    "EventExtractor",
    "PresentationExtractor",
    "PresentationBlockExtractor",
    "PresentationEventExtractor",
    "EPrimeExtractor",
    "EPrimeBlockExtractor",
    "EPrimeEventExtractor",
]

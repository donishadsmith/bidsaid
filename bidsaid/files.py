"""Module for creating BIDS compliant files."""

import json, re
from pathlib import Path

import pandas as pd

from bidsaid.io import _copy_file


def generate_bids_filename(
    sub_id: str | int,
    desc: str,
    ext: str,
    ses_id: str | int | None = None,
    task_id: str | None = None,
    run_id: str | int | None = None,
) -> str:
    """
    Generate a BIDs compliant filename.

    Parameters
    ----------

    sub_id : :obj:`str` or :obj:`int`
        Subject ID (i.e. 01, 101, etc).

    desc : :obj:`str`
        Description of the file (i.e., T1w, bold, etc).

    ext : :obj:`str`
        The extension of the file.

    ses_id : :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Session ID (i.e. 001, 1, etc). Optional entity.

    ses_id : :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Session ID (i.e. 001, 1, etc). Optional entity.

    task_id : :obj:`str` or :obj:`None`, default=None
        Task ID (i.e. flanker, n_back, etc). Optional entity.

    run_id : :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Run ID (i.e. 001, 1, etc). Optional entity.

    Returns
    -------
    str
        A BIDS compliant filename.
    """
    bids_filename = (
        f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_run-{run_id}_{desc}.{ext}"
    )

    return _strip_none_entities(bids_filename)


def create_bids_file(
    src_file: str | Path,
    sub_id: str | int,
    desc: str,
    ses_id: str | int | None = None,
    task_id: str | None = None,
    run_id: str | int | None = None,
    dst_dir: str | Path = None,
    remove_src_file: bool = False,
    return_bids_filename: bool = False,
) -> Path | None:
    """
    Create a BIDS compliant filename with required and optional entities.

    Parameters
    ----------
    src_file : :obj:`str` or :obj:`Path`
        Path to the source file.

    sub_id : :obj:`str` or :obj:`int`
        Subject ID (i.e. 01, 101, etc).

    desc : :obj:`str`
        Description of the file (i.e., T1w, bold, etc).

    ses_id : :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Session ID (i.e. 001, 1, etc). Optional entity.

    task_id : :obj:`str` or :obj:`None`, default=None
        Task ID (i.e. flanker, n_back, etc). Optional entity.

    run_id : :obj:`str` or :obj:`int` or :obj:`None`, default=None
        Run ID (i.e. 001, 1, etc). Optional entity.

    dst_dir : :obj:`str`, :obj:`Path`, or :obj:`None`, default=None
        Directory name to copy the BIDS file to. If None, then the
        BIDS file is copied to the same directory as the ``src_file``.
        Directory will also be created if it does not exist.

    remove_src_file : :obj:`str`, default=False
        Delete the source file if True.

    return_bids_filename : :obj:`str`, default=False
        Returns the full BIDS filename if True.

    Returns
    -------
    Path or None
        If ``return_bids_filename`` is True, then the BIDS filename is
        returned.

    Note
    ----
    There are additional entities that can be used that are
    not included in this function.
    """
    ext = f"{str(src_file).partition('.')[-1]}"
    bids_filename = generate_bids_filename(sub_id, desc, ext, ses_id, task_id, run_id)

    bids_filename = (
        Path(src_file).parent / bids_filename
        if dst_dir is None
        else Path(dst_dir) / bids_filename
    )

    _copy_file(src_file, bids_filename, remove_src_file)

    return bids_filename if return_bids_filename else None


def _strip_none_entities(bids_filename: str | Path) -> str:
    """
    Removes entities with None in a BIDS compliant filename.

    Parameters
    ----------
    bids_filename : :obj:`str` or :obj:`Path`
        The BIDS filename.

    Returns
    -------
    str
        BIDS filename with entities ending in None removed.

    Example
    -------
    >>> from bidsaid.files import _strip_none_entities
    >>> bids_filename = "sub-101_ses-None_task-flanker_bold.nii.gz"
    >>> _strip_none_entities(bids_filename)
        "sub-101_task-flanker_bold.nii.gz"
    """
    basename, _, ext = str(bids_filename).partition(".")
    retained_entities = [
        entity for entity in basename.split("_") if not entity.endswith("-None")
    ]

    filename = f"{'_'.join(retained_entities)}"
    if ext:
        filename += f".{ext}"

    return filename


def get_entity_value(
    filename: str | Path, entity: str, return_entity_prefix: bool = False
) -> str | None:
    """
    Gets entity value of a BIDS compliant filename.

    Parameters
    ----------
    filename : :obj:`str` or :obj:`Path`
        Filename to extract entity from.

    entity : :obj:`str`
        The entity key (e.g. "sub", "task", etc).

    return_entity_prefix : :obj:`bool`, default=False
        Return value with the entity ("sub-101" instead of
        "101") if True.

    Returns
    -------
    str or None
        The entity value with the entity prefix if ``return_entity_prefix``
        is True.

    Example
    -------
    >>> from bidsaid.files import get_entity_value
    >>> get_entity_value("sub-01_task-flanker_bold.nii.gz", "task")
        "flanker"
    """
    basename = Path(filename).name
    match = re.search(rf"{entity.removesuffix('-')}-([^_\.]+)", basename)
    if match:
        entity_value = (
            f"{entity.removesuffix('-')}-{match.group(1)}"
            if return_entity_prefix
            else match.group(1)
        )
    else:
        entity_value = None

    return entity_value


def create_dataset_description(
    dataset_name: str, bids_version: str = "1.0.0", derivative=False
) -> dict[str, str]:
    """
    Generate a dataset description dictionary.

    Creates a dictionary containing the name and BIDs version of a dataset.

    Parameters
    ----------
    dataset_name : :obj:`str`
        Name of the dataset.

    bids_version : :obj:`str`,
        Version of the BIDS dataset.

    derivative : :obj:`bool`, default=False
        Determines if "GeneratedBy" key is added to dictionary.

    Returns
    -------
    dict[str, str]
        The dataset description dictionary
    """
    dataset_description = {"Name": dataset_name, "BIDSVersion": bids_version}

    if derivative:
        dataset_description.update({"GeneratedBy": [{"Name": dataset_name}]})

    return dataset_description


def save_dataset_description(
    dataset_description: dict[str, str], dst_dir: str | Path
) -> None:
    """
    Save a dataset description dictionary.

    Saves the dataset description dictionary as a file named "dataset_description.json" to the
    directory specified by ``output_dir``.

    Parameters
    ----------
    dataset_description : :obj:`dict`
        The dataset description dictionary.

    dst_dir : :obj:`str` or :obj:`Path`
        Path to save the JSON file to.
    """
    with open(Path(dst_dir) / "dataset_description.json", "w", encoding="utf-8") as f:
        json.dump(dataset_description, f)


def create_participant_tsv(
    bids_dir: str | Path, save_df: bool = False, return_df: bool = True
) -> pd.DataFrame | None:
    """
    Creates a basic dataframe for the "participants.tsv" file.

    Parameters
    ----------
    bids_dir : :obj:`str` or :obj:`Path`
        The root of BIDS compliant directory.

    save_df : :obj:`bool`, bool=False
        Save the dataframe to the root of the BIDS compliant directory.

    return_df : :obj:`str`, default=True
        Returns dataframe if True else return None.

    Returns
    -------
    pandas.DataFrame or None
        The dataframe if ``return_df`` is True.
    """
    participants = sorted([folder.name for folder in Path(bids_dir).glob("sub-*")])
    df = pd.DataFrame({"participant_id": participants})

    if save_df:
        df.to_csv(Path(bids_dir) / "participants.tsv", sep="\t", index=None)

    return df if return_df else None


def create_sessions_tsv(
    bids_dir: str | Path,
    sub_id: str | int,
    save_df: bool = False,
    return_df: bool = True,
) -> pd.DataFrame | None:
    """
    Creates a basic dataframe for the "sub-{subject_ID}_sessions.tsv" file
    for a specific subject.

    Parameters
    ----------
    bids_dir : :obj:`str` or :obj:`Path`
        The root of BIDS compliant directory.

    sub_id : :obj:`str` or obj:`int`
        The subject ID.

    save_df : :obj:`bool`, bool=False
        Save the dataframe in the subject folder.

    return_df : :obj:`str`, default=True
        Returns dataframe if True else return None.

    Returns
    -------
    pandas.DataFrame or None
        The dataframe if ``return_df`` is True.
    """
    target_sub = f"sub-{str(sub_id).removeprefix('sub-')}"
    subject_folder = Path(bids_dir) / target_sub
    session_folders = sorted(list(subject_folder.glob("ses-*")))
    session_ids = [session_folder.name for session_folder in session_folders]
    df = pd.DataFrame({"session_id": session_ids})

    if save_df:
        df.to_csv(Path(bids_dir) / f"{target_sub}_sessions.tsv", sep="\t", index=None)

    return df if return_df else None


__all__ = [
    "create_bids_file",
    "get_entity_value",
    "create_dataset_description",
    "save_dataset_description",
    "create_participant_tsv",
]

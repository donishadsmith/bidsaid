"""Module for creating simulated data."""

"""Module for creating simulated data."""

from pathlib import Path

import nibabel as nib, numpy as np
from joblib import Parallel, delayed
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_img
from tqdm import tqdm

from .bids import (
    _strip_none_entities,
    create_dataset_description,
    save_dataset_description,
)
from .logging import setup_logger

LGR = setup_logger(__name__)


def simulate_nifti_image(
    img_shape: tuple[int, int, int] | tuple[int, int, int, int],
) -> nib.Nifti1Image:
    """
    Simulates a NIfTI image with random data within the MNI152 brain template.

    Parameters
    ----------
    img_shape : :obj:`tuple[int, int, int]` or :obj:`tuple[int, int, int, int]`
        Shape of the NIfTI image.

    Returns
    -------
    Nifti1Image
        The NIfTI image.
    """
    if len(img_shape) not in [3, 4]:
        raise ValueError("The image shape must be 3D or 4D.")

    whole_brain_mask = load_mni152_brain_mask(resolution=1, threshold=0.20)

    new_affine = whole_brain_mask.affine.copy()
    # Compute new voxel size to retain same brain size in mm
    new_affine[:3, :3] = (
        whole_brain_mask.affine[:3, :3]
        * np.array(whole_brain_mask.shape[:3])
        / np.array(img_shape[:3])
    )

    resampled_mask = resample_img(
        whole_brain_mask,
        target_shape=img_shape[:3],
        target_affine=new_affine,
        interpolation="nearest",
    )

    mask_data = resampled_mask.get_fdata()
    if len(img_shape) == 4:
        mask_data = mask_data[..., np.newaxis]

    data = np.random.rand(*img_shape) * mask_data

    return nib.Nifti1Image(data, resampled_mask.affine)


def simulate_bids_dataset(
    n_subs: int = 1,
    n_sessions: int | None = 1,
    n_runs: int = 1,
    task_name: str = "rest",
    output_dir: str | Path | None = None,
    n_cores: int | None = None,
    progress_bar: bool = False,
) -> Path:
    """
    Generate a Simulated BIDS Dataset with fMRIPrep Derivatives.

    Creates a minimal BIDS dataset structure with fMRIPrep derivatives, including:
        - BIDS root directory with:
            - Dataset description JSON file
            - One simulated NIfTI image per subject/run
        - Derivatives folder with fMRIPrep outputs:
            - Dataset description JSON file
            - One simulated NIfTI image per subject/run

    .. note::
       Returns ``output_dir`` if the path exists.

    Parameters
    ----------
    n_subs : :obj:`int`, default=1
        Number of subjects.

    n_sessions : :obj:`int` or :obj:`None`, default=1
        Number of sessions for each subject.

    n_runs : :obj:`int`, default=1
        Number of runs for each subject.

    task_name : :obj:`str`, default="rest"
        Name of task.

    output_dir : :obj:`str`,  :obj:`Path`, or :obj:`None`, default=None
        Path to save the simulated BIDS directory to.

        .. important::
           If None, a directory named "simulated_bids_dir" will be created in the current working
           directory.

    n_cores : :obj:`int` or :obj:`None`, default=None
        The number of cores to use for multiprocessing with Joblib (over subjects). The "loky"
        backend is used.

    progress_bar : :obj:`bool`, default=False
        If True, displays a progress bar.

    Returns
    -------
    Path
        Root of the simulated BIDS directory.
    """
    if output_dir:
        bids_root = Path(output_dir)
    else:
        bids_root = Path().getcwd() / "simulated_bids_dir"

    if bids_root.exists():
        LGR.warning("`output_dir` already exists. Returning the `output_dir` string.")
        return bids_root

    # Create root directory with derivatives folder
    fmriprep_dir = bids_root / "derivatives" / "fmriprep"
    fmriprep_dir.mkdir(parents=True)

    # Create dataset description for root and fmriprep
    save_dataset_description(create_dataset_description("Mock"), bids_root)
    save_dataset_description(
        create_dataset_description("fMRIPrep", derivative=True), fmriprep_dir
    )

    # Generate list of tuples for each subject
    args_list = [
        (fmriprep_dir, sub_id, n_sessions, n_runs, task_name)
        for sub_id in range(n_subs)
    ]

    parallel = Parallel(return_as="generator", n_jobs=n_cores, backend="loky")
    # generator needed for tqdm, iteration triggers side effects (file creation)
    list(
        tqdm(
            parallel(delayed(_create_session_files)(*args) for args in args_list),
            desc="Creating Simulated Subjects",
            total=len(args_list),
            disable=not progress_bar,
        )
    )

    return bids_root


def _create_session_files(
    fmriprep_dir: Path,
    sub_id: int,
    n_sessions: int,
    n_runs: int,
    task_name: str,
) -> None:
    """Iterates through each session ID simulate dataset."""
    if n_sessions:
        n_sessions += 1
        for session_id in range(1, n_sessions):
            _create_sub_files(fmriprep_dir, sub_id, session_id, n_runs, task_name)
    else:
        _create_sub_files(fmriprep_dir, sub_id, n_sessions, n_runs, task_name)

    return None


def _create_sub_files(
    fmriprep_dir: Path,
    sub_id: int,
    session_id: int | None,
    n_runs: int,
    task_name: str,
) -> None:
    """Create directory and simulated dataset."""
    session_id = session_id or None
    sub_root_dir = (
        fmriprep_dir.parent.parent
        / f"sub-{sub_id + 1}"
        / (f"ses-{session_id}" if session_id else "")
        / "func"
    )
    sub_root_dir.mkdir(parents=True)
    sub_derivatives_dir = (
        fmriprep_dir
        / f"sub-{sub_id + 1}"
        / (f"ses-{session_id}" if session_id else "")
        / "func"
    )
    sub_derivatives_dir.mkdir(parents=True)

    for run_id in range(n_runs):
        base_filename = _strip_none_entities(
            f"sub-{sub_id + 1}_ses-{session_id}_task-{task_name}_run-{run_id + 1}"
        )
        nifti_img = simulate_nifti_image((97, 115, 98, 50))
        root_filename = base_filename + "_bold.nii.gz"
        nib.save(nifti_img, sub_root_dir / root_filename)
        derivatives_filename = (
            base_filename + "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
        nib.save(nifti_img, sub_derivatives_dir / derivatives_filename)

    return None


__all__ = ["simulate_nifti_image", "create_affine", "simulate_bids_dataset"]

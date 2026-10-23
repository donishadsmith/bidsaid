# BIDS-Aid

[![Latest Version](https://img.shields.io/pypi/v/bidsaid.svg)](https://pypi.python.org/pypi/bidsaid/)
[![Python Versions](https://img.shields.io/pypi/pyversions/bidsaid.svg)](https://pypi.python.org/pypi/bidsaid/)
[![Source Code](https://img.shields.io/badge/Source%20Code-bidsaid-purple)](https://github.com/donishadsmith/bidsaid)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Test Status](https://github.com/donishadsmith/bidsaid/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/bidsaid/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/gh/donishadsmith/bidsaid/graph/badge.svg?token=PCJ17NA627)](https://codecov.io/gh/donishadsmith/bidsaid)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/bidsaid/badge/?version=stable)](http://bidsaid.readthedocs.io/en/stable/?badge=stable)


A toolkit for creating and managing BIDS-compliant fMRI datasets without original DICOMs. Intended for cases that require custom code and flexibility, such as when NIfTI source files lack consistent naming conventions, organized folder hierarchies, or sidecar metadata. Includes utilities for metadata reconstruction from NIfTI headers, file renaming, neurobehavioral log parsing (for E-Prime and Presentation), JSON sidecar generation, surface projection, and making surface clips/movies.

## Installation

### Standard Installation
```bash
pip install bidsaid[all]
```

Surface projection and movie writing are optional extras: `bidsaid[surface]` (`neuromaps` for the fsaverage surfaces), `bidsaid[visual]` (`pyvista` for rendering) and `bidsaid[movie]` (`imageio` with ffmpeg for writing movies). `bidsaid[all]` includes everything.

### Development Version
```bash
git clone --depth 1 https://github.com/donishadsmith/bidsaid/
cd bidsaid
pip install -e .[all]
```

## Features

- **File renaming**: Convert arbitrary filenames to BIDS-compliant naming
- **File creation**: Generate `dataset_description.json` and `participants.tsv`
- **Metadata utilities**: Extract header metadata (e.g., TR, orientation, scanner info) and generate slice timing for singleband and multiband acquisitions
- **Log parsing**: Load Presentation (e.g., `.log`) and E-Prime 3 (e.g, `.edat3`, `.txt`) files as DataFrames, or use extractor classes to generate BIDS events for block and event designs:

    | Class | Software | Design | Description |
    |-------|----------|--------|-------------|
    | `PresentationBlockExtractor` | Presentation | Block | Extracts block-level timing with mean RT and accuracy |
    | `PresentationEventExtractor` | Presentation | Event | Extracts trial-level timing with individual responses |
    | `EPrimeBlockExtractor` | E-Prime 3 | Block | Extracts block-level timing with mean RT and accuracy |
    | `EPrimeEventExtractor` | E-Prime 3 | Event | Extracts trial-level timing with individual responses |

- **Auditing**: Generate a table showing the presence or absence of certain files for each subject and session
- **QC**: Creation and computation of certain quality control metrics (e.g., framewise displacement), censoring and temporal interpolation
- **Surface movies**: Project denoised runs onto the fsaverage surfaces and write per-participant, per-task or group movies with `SurfaceRenderer` and `MovieWriter`

## Quick Start

### Creating BIDS-Compliant Filenames
```python
from bidsaid.files import create_bids_file

create_bids_file(
    src_file="101_mprage.nii.gz",
    sub_id="101",
    ses_id="01",
    desc="T1w",
    dst_dir="/data/bids/sub-101/ses-01/anat",
)
```

### Extracting Metadata from NIfTI Headers
```python
from bidsaid.metadata import get_tr, create_slice_timing, get_image_orientation

tr = get_tr("sub-01_bold.nii.gz")
slice_timing = create_slice_timing(
    "sub-01_bold.nii.gz",
    slice_acquisition_method="interleaved",
    multiband_factor=4,
)
orientation_map, orientation = get_image_orientation("sub-01_bold.nii.gz")
```

### Loading Raw Log Files
```python
from bidsaid.parsers import (
    load_presentation_log,
    load_eprime_log,
    convert_edat3_to_text,
)

presentation_df = load_presentation_log("sub-01_task.log", convert_to_seconds=["Time"])

# E-Prime 3: convert .edat3 to text first, or load .txt directly
eprime_txt_path = convert_edat3_to_text("sub-01_task.edat3")
eprime_df = load_eprime_log(eprime_txt_path, convert_to_seconds=["Stimulus.OnsetTime"])
```

### Creating BIDS Events from Presentation Logs
```python
from bidsaid.events import PresentationBlockExtractor
import pandas as pd

extractor = PresentationBlockExtractor(
    "sub-01_task-faces.log",
    trial_types=("Face", "Place"),  # Can use regex ("Fa.*", "Pla.*")
    scanner_event_type="Pulse",
    scanner_trigger_code="99",
    convert_to_seconds=["Time"],
    rest_block_codes="crosshair",
    rest_code_pattern="fixed",
    split_cue_from_block=True,
)

events_df = pd.DataFrame(
    {
        "onset": extractor.extract_onsets(),
        "duration": extractor.extract_durations(),
        "trial_type": extractor.extract_trial_types(),
        "mean_rt": extractor.extract_mean_reaction_times(),
    }
)
```

### Creating BIDS Events from E-Prime Logs
```python
from bidsaid.events import EPrimeEventExtractor
import pandas as pd

extractor = EPrimeEventExtractor(
    "sub-01_task-gonogo.txt",
    trial_types="Go|NoGo",  # Can also use ("Go", "NoGo")
    onset_column_name="Stimulus.OnsetTime",
    procedure_column_name="Procedure",
    trigger_column_name="ScannerTrigger.RTTime",
    convert_to_seconds=[
        "Stimulus.OnsetTime",
        "Stimulus.OffsetTime",
        "ScannerTrigger.RTTime",
    ],
)

events_df = pd.DataFrame(
    {
        "onset": extractor.extract_onsets(),
        "duration": extractor.extract_durations(
            offset_column_name="Stimulus.OffsetTime"
        ),
        "trial_type": extractor.extract_trial_types(),
        "reaction_time": extractor.extract_reaction_times(
            reaction_time_column_name="Stimulus.RT"
        ),
    }
)
```

### Audit BIDS Dataset
```python
from bidsaid.audit import BIDSAuditor
from bidsaid.simulate import simulate_bids_dataset

bids_root = simulate_bids_dataset()

auditor = BIDSAuditor(bids_root)
auditor.check_raw_nifti_availability()
auditor.check_raw_sidecar_availability()
auditor.check_events_availability()
auditor.check_preprocessed_nifti_availability()

analysis_dir = bids_root / "first_level"
analysis_sub_dir = analysis_dir / "sub-1" / "ses-1"
analysis_sub_dir.mkdir(parents=True, exist_ok=True)

with open(analysis_sub_dir / "sub-1_task-rest_desc-betas.nii.gz", "w") as f:
    pass

auditor.check_first_level_availability(analysis_dir=analysis_dir, desc="betas")
```

### Compute QC
```python
from bidsaid.qc import create_censor_mask, compute_consecutive_censor_stats

censor_mask = create_censor_mask(
    "confounds.tsv",
    column_name="framewise_displacement",
    threshold=0.5,
    n_dummy_scans=4,
)

consecutive_censor_mean, consecutive_censor_std = compute_consecutive_censor_stats(
    censor_mask, n_dummy_scans=4
)
```

### Surface Movies
```python
from bidsaid.movie import MovieWriter, compute_vmax, create_surface_run, virtual_display
from bidsaid.surface import fetch_fsaverage_surfaces, get_cortex_masks
from bidsaid.visual import SurfaceRenderer

surfaces = fetch_fsaverage_surfaces("10k")
cortex_masks = get_cortex_masks(surfaces["medial"])

# One run per participant and task; the "sub" and "task" entities are read from the filename
surface_runs = [
    create_surface_run(
        denoised_file,
        surfaces["pial"],
        surfaces["white"],
        cortex_masks,
        n_interpolated_frames=1,  # smoother playback (display only)
        fwhm=4,
    )
    for denoised_file in [
        "sub-01_task-rest_desc-denoised_bold.nii.gz",
        "sub-01_task-music_desc-denoised_bold.nii.gz",
        "sub-02_task-rest_desc-denoised_bold.nii.gz",
        "sub-02_task-music_desc-denoised_bold.nii.gz",
    ]
]

threshold = 1.0
vmax = compute_vmax(surface_runs, threshold=threshold)

# virtual_display starts Xvfb for headless environments
with (
    virtual_display(),
    SurfaceRenderer(
        *surfaces["inflated"],
        sulcal_depths=surfaces["sulc"],
        color_range=(-vmax, vmax),
        threshold=threshold,
        top_spacing=80,  # space aboce brains for label
    ) as renderer,
):
    movie_files = MovieWriter(
        surface_runs,
        renderer,
        task_order=["rest", "music"],
        playback_speed=1,
        label_format="{subject_name} {task_name} t = {frame_time:.1f} s",
    ).write("movies", per_subject=True, per_task=True, shared=True)
```

Each participant is a row and each task a column. `per_subject` writes one movie per participant with their tasks side by side, `per_task` one movie per task with the participants stacked, and `shared` the full grid in a single movie.

See the [API documentation](https://bidsaid.readthedocs.io/en/latest/api.html) for full parameter details and additional utilities.

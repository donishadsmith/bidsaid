"""Quality control utilities for fMRI data."""

from .censoring import (
    compute_n_dummy_scans,
    compute_framewise_displacement,
    create_censor_mask,
    merge_censor_masks,
    create_spike_regressors,
    compute_consecutive_censor_stats,
    get_n_censored_volumes,
)
from .nuisance import compute_global_signal

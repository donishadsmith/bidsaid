"""Quality control utilities for fMRI data."""

from .censoring import (
    compute_n_dummy_scans,
    compute_framewise_displacement,
    create_censor_mask,
    merge_censor_masks,
    create_spike_regressors,
    compute_consecutive_censor_stats,
    get_n_censored_volumes,
    find_outlier_volumes,
)
from .denoise import regress_confounds, regress_confounds_array
from .normalize import (
    zscore_img,
    zscore_array,
    zscore_to_reference,
    zscore_to_reference_array,
)
from .nuisance import compute_global_signal
from .interpolate import (
    downsample_img,
    upsample_img,
    upsample_array,
    downsample_array,
    interpolate_censored_array,
    interpolate_censored_img,
)

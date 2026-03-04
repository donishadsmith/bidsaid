API
===
:mod:`nifti2bids.audit`
----------------------
Module for auditing directories.

.. currentmodule:: nifti2bids.audit

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   BIDSAuditor

:mod:`nifti2bids.bids`
----------------------
Module for initializing and creating BIDS compliant files.

.. currentmodule:: nifti2bids.bids

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   generate_bids_filename
   create_bids_file
   get_entity_value
   create_dataset_description
   save_dataset_description
   create_participant_tsv
   create_sessions_tsv
   add_instruction_timing

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   PresentationBlockExtractor
   PresentationEventExtractor
   EPrimeBlockExtractor
   EPrimeEventExtractor

:mod:`nifti2bids.logging`
-------------------------
Module setting up a logger.

.. currentmodule:: nifti2bids.logging

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   setup_logger


:mod:`nifti2bids.io`
--------------------
Module for input/output operations on NIfTI files and images.

.. currentmodule:: nifti2bids.io

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   load_nifti
   compress_image
   regex_glob
   get_nifti_header
   get_nifti_affine
   replace_ext

:mod:`nifti2bids.metadata`
--------------------------
Module containing functions to extract metadata information
from NIfTIs.

.. currentmodule:: nifti2bids.metadata

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   needs_resampling
   determine_slice_axis
   get_hdr_metadata
   get_n_volumes
   get_image_orientation
   direction_to_voxel_axis
   get_n_slices
   get_tr
   create_slice_timing
   is_3d_img
   get_scanner_info
   is_valid_date
   parse_date_from_path
   get_file_timestamp
   get_file_creation_date
   infer_task_from_image
   get_recon_matrix_pe
   compute_effective_echo_spacing
   compute_total_readout_time

:mod:`nifti2bids.parsers`
--------------------------
Module containing functions to parse raw logs from stimulus
Presentation and E-Prime 3 software.

.. currentmodule:: nifti2bids.parsers

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   convert_edat3_to_text
   load_eprime_log
   load_presentation_log
   get_presentation_log_date

:mod:`nifti2bids.qc`
--------------------
Module for quality control metrics and assessment of fMRI data.

.. currentmodule:: nifti2bids.qc

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   compute_n_dummy_scans
   compute_framewise_displacement
   create_censor_mask
   merge_censor_masks
   compute_consecutive_censor_stats
   create_spike_regressors
   get_n_censored_volumes
   compute_global_signal

:mod:`nifti2bids.simulate`
--------------------------
Module for simulating NIfTI images and BIDS datasets.

.. currentmodule:: nifti2bids.simulate

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   simulate_nifti_image
   create_affine
   simulate_bids_dataset

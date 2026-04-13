API
===
:mod:`bidsaid.audit`
----------------------
Module for auditing directories.

.. currentmodule:: bidsaid.audit

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   BIDSAuditor

:mod:`bidsaid.files`
----------------------
Module for initializing and creating BIDS compliant files.

.. currentmodule:: bidsaid.files

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

:mod:`bidsaid.events`
----------------------
Module containing classes for extracting timing information from Presentation and E-Prime logs.

.. currentmodule:: bidsaid.events

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   PresentationBlockExtractor
   PresentationEventExtractor
   EPrimeBlockExtractor
   EPrimeEventExtractor

:mod:`bidsaid.logging`
-------------------------
Module setting up a logger.

.. currentmodule:: bidsaid.logging

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   setup_logger


:mod:`bidsaid.io`
--------------------
Module for input/output operations on NIfTI files and images.

.. currentmodule:: bidsaid.io

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

:mod:`bidsaid.path_utils`
--------------------------
Utilities related to parsing or sorting filenames.

.. currentmodule:: bidsaid.path_utils

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   is_valid_date
   parse_date_from_path
   get_file_timestamp
   get_file_creation_date
   sort_by_acquisition_order

:mod:`bidsaid.metadata`
--------------------------
Module containing functions to extract metadata information
from NIfTIs.

.. currentmodule:: bidsaid.metadata

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
   infer_task_from_image
   get_recon_matrix_pe
   compute_effective_echo_spacing
   compute_total_readout_time

:mod:`bidsaid.parsers`
--------------------------
Module containing functions to parse raw logs from stimulus
Presentation and E-Prime 3 software.

.. currentmodule:: bidsaid.parsers

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   convert_edat3_to_text
   load_eprime_log
   load_presentation_log
   get_presentation_log_date

:mod:`bidsaid.qc`
--------------------
Module for quality control metrics and assessment of fMRI data.

.. currentmodule:: bidsaid.qc

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

:mod:`bidsaid.simulate`
--------------------------
Module for simulating NIfTI images and BIDS datasets.

.. currentmodule:: bidsaid.simulate

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   simulate_nifti_image
   simulate_bids_dataset

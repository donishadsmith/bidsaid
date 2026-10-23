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

:mod:`bidsaid.cli`
--------------------
Module containing helpers for building command-line interfaces with ``argparse``.

.. currentmodule:: bidsaid.cli

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   none_or
   str_to_bool
   n_dummy_type

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
   add_cue_timing

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
   is_nifti_truncated
   regex_glob
   get_nifti_header
   get_nifti_affine
   replace_ext
   compute_uncompressed_image_size
   truncate_nifti_to_complete_volumes

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

:mod:`bidsaid.movie`
-----------------------
Module for writing movies of surface data. Requires ``pip install bidsaid[movie]``.

.. currentmodule:: bidsaid.movie

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   MovieWriter
   SurfaceRun

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   create_surface_run
   compute_frame_timing
   compute_vmax
   virtual_display
   mp4_to_gif

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
   downsample_img
   upsample_img
   downsample_array
   upsample_array
   interpolate_censored_array
   interpolate_censored_img

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

:mod:`bidsaid.surface`
-------------------------
Module for projecting volumetric data onto cortical surfaces. ``fetch_fsaverage_surfaces`` requires ``pip install bidsaid[surface]``

.. currentmodule:: bidsaid.surface

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   fetch_fsaverage_surfaces
   load_surface_mesh
   load_surface_data
   get_cortex_masks
   project_to_surface

:mod:`bidsaid.visual`
------------------------
Module for rendering surface data as images. Requires ``pip install bidsaid[visual]``.

.. currentmodule:: bidsaid.visual

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   SurfaceRenderer

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   get_colormap
   create_polydata

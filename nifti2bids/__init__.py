"""
Post-hoc BIDS toolkit for NIfTI datasets without original DICOMs.
-----------------------------------------------------------------
Documentation can be found at https://nifti2bids.readthedocs.io.

Submodules
----------
audit -- Contains the ``BIDSAuditor`` class to check for certain file availability

bids -- Operations related to initializing and creating BIDS compliant files

io -- Generic operations related to loading NIfTI data

logging -- Set up a logger using ``RichHandler`` as the default handler if a root or
module specific handler is not available

metadata -- Operations related to extracting or creating metadata information from NIfTI images

parsers -- Operations related to standardizing and parsing information logs created by stimulus
neurobehavioral software such as Presentation and EPrime

qc -- Quality control utilities for fMRI data (motion censoring, framewise displacement, etc.)

simulate -- Simulate a basic NIfTI image or BIDS dataset for testing purposes
"""

__version__ = "0.17.5"

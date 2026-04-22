"""
A toolkit for creating and managing BIDS-compliant fMRI datasets without original DICOMs.
-----------------------------------------------------------------------------------------
Documentation can be found at https://bidsaid.readthedocs.io.

Submodules
----------
audit -- Contains the ``BIDSAuditor`` class to check for certain file availability

files -- Operations related to initializing and creating BIDS compliant files

events -- Contains classes to extract timing information from Presentation or E-Prime logs

io -- Generic operations related to loading NIfTI data

logging -- Set up a logger using ``RichHandler`` as the default handler if a root or
module specific handler is not available

metadata -- Operations related to extracting or creating metadata information from NIfTI images

parsers -- Operations related to standardizing and parsing information logs created by stimulus
neurobehavioral software such as Presentation and E-Prime

path_utils -- Utilities related to parsing or sorting filenames

qc -- Quality control utilities for fMRI data (motion censoring, framewise displacement, etc.)

simulate -- Simulate a basic NIfTI image or BIDS dataset for testing purposes
"""

__version__ = "0.24.0.post1"

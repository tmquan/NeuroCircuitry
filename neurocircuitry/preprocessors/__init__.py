"""
Preprocessor classes for handling multiple data formats.

Preprocessors normalize data to a standard internal format (numpy arrays or
torch tensors) and provide consistent methods for loading, validation, and
format conversion.
"""

from neurocircuitry.preprocessors.base import BasePreprocessor
from neurocircuitry.preprocessors.tiff import TIFFPreprocessor
from neurocircuitry.preprocessors.hdf5 import HDF5Preprocessor
from neurocircuitry.preprocessors.nrrd import NRRDPreprocessor

__all__ = [
    "BasePreprocessor",
    "TIFFPreprocessor",
    "HDF5Preprocessor",
    "NRRDPreprocessor",
]

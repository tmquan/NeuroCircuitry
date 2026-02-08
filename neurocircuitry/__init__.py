"""
NeuroCircuitry: A modular, extensible PyTorch Lightning-based infrastructure for connectomics research.

This library provides:
- MONAI-compatible dataset classes with standardized interfaces
- Preprocessor classes for handling multiple data formats (TIFF, HDF5, NRRD)
- Distributed training scaffolding with PyTorch Lightning
- Pre-built models including Vista3D for 3D segmentation
"""

__version__ = "0.1.0"

from neurocircuitry.datasets import (
    BaseConnectomicsDataset,
    SNEMI3DDataset,
    CREMI3DDataset,
    MICRONSDataset,
)
from neurocircuitry.preprocessors import (
    BasePreprocessor,
    TIFFPreprocessor,
    HDF5Preprocessor,
    NRRDPreprocessor,
)
from neurocircuitry.datamodules import (
    BaseConnectomicsDataModule,
    SNEMI3DDataModule,
    CREMI3DDataModule,
    MICRONSDataModule,
)

__all__ = [
    # Datasets
    "BaseConnectomicsDataset",
    "SNEMI3DDataset",
    "CREMI3DDataset",
    "MICRONSDataset",
    # Preprocessors
    "BasePreprocessor",
    "TIFFPreprocessor",
    "HDF5Preprocessor",
    "NRRDPreprocessor",
    # DataModules
    "BaseConnectomicsDataModule",
    "SNEMI3DDataModule",
    "CREMI3DDataModule",
    "MICRONSDataModule",
]

"""
Dataset classes for connectomics research.

All datasets inherit from BaseConnectomicsDataset and implement:
- paper: Reference/citation metadata
- resolution: Voxel/spatial resolution specification
- labels: List of segmentation class labels
- data_files: Dictionary with volume and segmentation paths/arrays
"""

from neurocircuitry.datasets.base import BaseConnectomicsDataset
from neurocircuitry.datasets.snemi3d import SNEMI3DDataset
from neurocircuitry.datasets.cremi3d import CREMI3DDataset
from neurocircuitry.datasets.microns import MICRONSDataset

__all__ = [
    "BaseConnectomicsDataset",
    "SNEMI3DDataset",
    "CREMI3DDataset",
    "MICRONSDataset",
]

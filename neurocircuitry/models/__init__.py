"""
Model architectures for connectomics segmentation.

Includes:
- BaseModel: Abstract base class for all models
- Vista3D: NVIDIA's 3D foundation model wrapper for connectomics
- SegResNetWrapper: MONAI SegResNet with customizable heads
"""

from neurocircuitry.models.base import BaseModel
from neurocircuitry.models.vista3d import Vista3DWrapper
from neurocircuitry.models.segresnet import SegResNetWrapper

__all__ = [
    "BaseModel",
    "Vista3DWrapper",
    "SegResNetWrapper",
]

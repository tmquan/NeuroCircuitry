"""
Loss functions for connectomics segmentation.

Includes:
- DiscriminativeLoss: Instance embedding loss from De Brabandere et al. (2017)
- BoundaryLoss: Boundary-aware loss functions
"""

from neurocircuitry.losses.discriminative import DiscriminativeLoss, DiscriminativeLossVectorized
from neurocircuitry.losses.boundary import BoundaryLoss, BoundaryAwareCrossEntropy

__all__ = [
    "DiscriminativeLoss",
    "DiscriminativeLossVectorized",
    "BoundaryLoss",
    "BoundaryAwareCrossEntropy",
]

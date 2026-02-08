"""
PyTorch Lightning modules for connectomics training tasks.

Includes:
- SemanticSegmentationModule: Cross-entropy based semantic segmentation
- InstanceSegmentationModule: Discriminative loss + semantic head
- AffinitySegmentationModule: Affinity-based boundary prediction
"""

from neurocircuitry.modules.semantic_seg import SemanticSegmentationModule
from neurocircuitry.modules.instance_seg import InstanceSegmentationModule
from neurocircuitry.modules.affinity_seg import AffinitySegmentationModule

__all__ = [
    "SemanticSegmentationModule",
    "InstanceSegmentationModule",
    "AffinitySegmentationModule",
]

"""
PyTorch Lightning modules for connectomics training tasks.

Includes:
- SemanticSegmentationModule: Cross-entropy based semantic segmentation
- InstanceSegmentationModule: Discriminative loss + semantic head
- AffinitySegmentationModule: Affinity-based boundary prediction
- Vista3DModule: Vista3D foundation model with auto/interactive modes
"""

from neurocircuitry.modules.semantic_seg import SemanticSegmentationModule
from neurocircuitry.modules.instance_seg import InstanceSegmentationModule
from neurocircuitry.modules.affinity_seg import AffinitySegmentationModule
from neurocircuitry.modules.vista3d_module import Vista3DModule

__all__ = [
    "SemanticSegmentationModule",
    "InstanceSegmentationModule",
    "AffinitySegmentationModule",
    "Vista3DModule",
]

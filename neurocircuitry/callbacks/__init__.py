"""
Callbacks for NeuroCircuitry training.

This module provides custom PyTorch Lightning callbacks for:
- TensorBoard volume visualization
- Custom logging and monitoring
"""

from neurocircuitry.callbacks.tensorboard_volume import TensorBoardVolumeCallback

__all__ = [
    "TensorBoardVolumeCallback",
]

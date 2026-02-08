"""
CREMI3D DataModule for PyTorch Lightning.
"""

from typing import List, Optional

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    ScaleIntensityd,
    ToTensord,
    Resized,
)

from neurocircuitry.datamodules.base import BaseConnectomicsDataModule
from neurocircuitry.datasets import CREMI3DDataset


class CREMI3DDataModule(BaseConnectomicsDataModule):
    """
    PyTorch Lightning DataModule for CREMI3D dataset.
    
    Wraps CREMI3DDataset with appropriate transforms and data loading
    configuration for training neuron and synapse segmentation models.
    
    Args:
        data_root: Path to CREMI data directory.
        batch_size: Batch size (default: 4).
        num_workers: Data loading workers (default: 4).
        train_val_split: Validation fraction (default: 0.2).
        cache_rate: Cache fraction (default: 0.5).
        pin_memory: Pin memory for GPU transfer (default: True).
        image_size: Optional resize dimensions (H, W).
        samples: List of samples to use ('A', 'B', 'C').
        include_synapses: Include synaptic cleft labels (default: True).
        slice_mode: Return 2D slices if True (default: True).
    
    Example:
        >>> dm = CREMI3DDataModule(
        ...     data_root="/path/to/cremi",
        ...     batch_size=8,
        ...     samples=["A", "B"],
        ...     include_synapses=True
        ... )
        >>> dm.setup("fit")
        >>> 
        >>> for batch in dm.train_dataloader():
        ...     images = batch["image"]  # [B, 1, H, W]
        ...     labels = batch["label"]  # [B, 1, H, W]
        ...     clefts = batch.get("clefts")  # [B, 1, H, W] if available
    """
    
    dataset_class = CREMI3DDataset
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        samples: Optional[List[str]] = None,
        include_synapses: bool = True,
        slice_mode: bool = True,
        persistent_workers: bool = True,
    ):
        self.samples = samples
        self.include_synapses = include_synapses
        self.slice_mode = slice_mode
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_val_split=train_val_split,
            cache_rate=cache_rate,
            pin_memory=pin_memory,
            image_size=image_size,
            persistent_workers=persistent_workers,
        )
    
    def _get_dataset_kwargs(self) -> dict:
        kwargs = {
            "slice_mode": self.slice_mode,
            "include_synapses": self.include_synapses,
        }
        if self.samples is not None:
            kwargs["samples"] = self.samples
        return kwargs
    
    def get_train_transforms(self) -> Compose:
        """
        Get training transforms for CREMI grayscale EM images.
        
        Includes transforms for both neuron labels and optional synapse clefts.
        
        Returns:
            MONAI Compose transform pipeline.
        """
        # Determine keys based on whether synapses are included
        label_keys = ["image", "label"]
        if self.include_synapses:
            label_keys.append("clefts")
        
        transforms = [
            EnsureChannelFirstd(keys=label_keys, channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            # Use nearest interpolation for all label types
            modes = ["bilinear"] + ["nearest"] * (len(label_keys) - 1)
            transforms.append(
                Resized(
                    keys=label_keys,
                    spatial_size=self.image_size,
                    mode=modes,
                )
            )
        
        # Augmentations (apply same spatial transforms to all)
        transforms.extend([
            RandFlipd(keys=label_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=label_keys, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=label_keys, prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            ToTensord(keys=label_keys),
        ])
        
        return Compose(transforms)
    
    def get_val_transforms(self) -> Compose:
        """Get validation transforms for CREMI dataset."""
        label_keys = ["image", "label"]
        if self.include_synapses:
            label_keys.append("clefts")
        
        transforms = [
            EnsureChannelFirstd(keys=label_keys, channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            modes = ["bilinear"] + ["nearest"] * (len(label_keys) - 1)
            transforms.append(
                Resized(
                    keys=label_keys,
                    spatial_size=self.image_size,
                    mode=modes,
                )
            )
        
        transforms.append(ToTensord(keys=label_keys))
        
        return Compose(transforms)

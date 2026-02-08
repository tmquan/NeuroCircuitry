"""
SNEMI3D DataModule for PyTorch Lightning.
"""

from typing import Optional

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
from neurocircuitry.datasets import SNEMI3DDataset


class SNEMI3DDataModule(BaseConnectomicsDataModule):
    """
    PyTorch Lightning DataModule for SNEMI3D dataset.
    
    Wraps SNEMI3DDataset with appropriate transforms and data loading
    configuration for training neuron segmentation models.
    
    Args:
        data_root: Path to SNEMI3D data directory.
        batch_size: Batch size (default: 4).
        num_workers: Data loading workers (default: 4).
        train_val_split: Validation fraction (default: 0.2).
        cache_rate: Cache fraction (default: 0.5).
        pin_memory: Pin memory for GPU transfer (default: True).
        image_size: Optional resize dimensions (H, W).
        slice_mode: Return 2D slices if True (default: True).
    
    Example:
        >>> dm = SNEMI3DDataModule(
        ...     data_root="/path/to/snemi3d",
        ...     batch_size=8,
        ...     num_workers=4
        ... )
        >>> dm.setup("fit")
        >>> 
        >>> for batch in dm.train_dataloader():
        ...     images = batch["image"]  # [B, 1, H, W]
        ...     labels = batch["label"]  # [B, 1, H, W]
    """
    
    dataset_class = SNEMI3DDataset
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        slice_mode: bool = True,
        persistent_workers: bool = True,
    ):
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
        return {"slice_mode": self.slice_mode}
    
    def get_train_transforms(self) -> Compose:
        """
        Get training transforms optimized for SNEMI3D grayscale EM images.
        
        Returns:
            MONAI Compose transform pipeline with EM-specific augmentations.
        """
        transforms = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.image_size,
                    mode=["bilinear", "nearest"],
                )
            )
        
        # EM-specific augmentations
        transforms.extend([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            # Stronger noise for EM robustness
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            # Contrast variation common in EM
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            ToTensord(keys=["image", "label"]),
        ])
        
        return Compose(transforms)

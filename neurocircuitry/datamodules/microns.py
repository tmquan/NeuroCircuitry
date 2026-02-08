"""
MICRONS DataModule for PyTorch Lightning.
"""

from typing import Optional, Tuple

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
    RandSpatialCropd,
)

from neurocircuitry.datamodules.base import BaseConnectomicsDataModule
from neurocircuitry.datasets import MICRONSDataset


class MICRONSDataModule(BaseConnectomicsDataModule):
    """
    PyTorch Lightning DataModule for MICRONS dataset.
    
    Wraps MICRONSDataset with appropriate transforms and data loading
    configuration for training on large-scale cortical connectomics data.
    
    Args:
        data_root: Path to MICRONS data directory.
        batch_size: Batch size (default: 4).
        num_workers: Data loading workers (default: 4).
        train_val_split: Validation fraction (default: 0.2).
        cache_rate: Cache fraction (default: 0.5).
        pin_memory: Pin memory for GPU transfer (default: True).
        image_size: Optional resize dimensions (H, W) or (D, H, W).
        volume_file: Name of volume file (default: 'volume').
        segmentation_file: Name of segmentation file (default: 'segmentation').
        include_synapses: Load synapse annotations (default: False).
        include_mitochondria: Load mitochondria labels (default: False).
        slice_mode: Return 2D slices if True (default: True).
        patch_size: If set, return 3D patches of this size.
        crop_size: Random crop size for training augmentation.
    
    Example:
        >>> dm = MICRONSDataModule(
        ...     data_root="/path/to/microns",
        ...     batch_size=4,
        ...     crop_size=(256, 256)
        ... )
        >>> dm.setup("fit")
    """
    
    dataset_class = MICRONSDataset
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        volume_file: str = "volume",
        segmentation_file: str = "segmentation",
        include_synapses: bool = False,
        include_mitochondria: bool = False,
        slice_mode: bool = True,
        patch_size: Optional[Tuple[int, int, int]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        persistent_workers: bool = True,
    ):
        self.volume_file = volume_file
        self.segmentation_file = segmentation_file
        self.include_synapses = include_synapses
        self.include_mitochondria = include_mitochondria
        self.slice_mode = slice_mode
        self.patch_size = patch_size
        self.crop_size = crop_size
        
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
        return {
            "volume_file": self.volume_file,
            "segmentation_file": self.segmentation_file,
            "include_synapses": self.include_synapses,
            "include_mitochondria": self.include_mitochondria,
            "slice_mode": self.slice_mode,
            "patch_size": self.patch_size,
        }
    
    def _get_label_keys(self) -> list:
        """Get list of label keys based on configuration."""
        keys = ["image", "label"]
        if self.include_synapses:
            keys.append("synapses")
        if self.include_mitochondria:
            keys.append("mitochondria")
        return keys
    
    def get_train_transforms(self) -> Compose:
        """
        Get training transforms for MICRONS dataset.
        
        Includes optional random cropping for handling large images.
        
        Returns:
            MONAI Compose transform pipeline.
        """
        label_keys = self._get_label_keys()
        
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
        
        # Random crop for large images
        if self.crop_size is not None:
            transforms.append(
                RandSpatialCropd(
                    keys=label_keys,
                    roi_size=self.crop_size,
                    random_size=False,
                )
            )
        
        # Augmentations
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
        """Get validation transforms for MICRONS dataset."""
        label_keys = self._get_label_keys()
        
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
        
        # Center crop for validation if crop_size is set
        if self.crop_size is not None:
            from monai.transforms import CenterSpatialCropd
            transforms.append(
                CenterSpatialCropd(
                    keys=label_keys,
                    roi_size=self.crop_size,
                )
            )
        
        transforms.append(ToTensord(keys=label_keys))
        
        return Compose(transforms)

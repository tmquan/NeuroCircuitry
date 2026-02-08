#!/usr/bin/env python
"""
Vista3D Training Script for Connectomics Segmentation.

Supports:
- Automatic segmentation mode (class label prompts)
- Interactive segmentation mode (point click prompts)
- Mixed training with both modes
- Random 3D cropping to configurable patch size (default 128x128x128)

Usage:
    # Train with auto mode
    python scripts/train_vista3d.py --config configs/snemi3d_vista3d.yaml
    
    # Train with interactive mode
    python scripts/train_vista3d.py --config configs/snemi3d_vista3d.yaml \
        training.mode=interactive
    
    # Train with mixed mode
    python scripts/train_vista3d.py --config configs/snemi3d_vista3d.yaml \
        training.mode=mixed
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from einops import rearrange

from monai.transforms import (
    Compose,
    MapTransform,
    ScaleIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    ToTensord,
    CenterSpatialCropd,
    SpatialPadd,
)
from monai.data import CacheDataset, DataLoader


class AddChannelDimd(MapTransform):
    """Add channel dimension to data if not present and ensure correct dtype."""
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                arr = d[key]
                # Convert to numpy if needed
                if isinstance(arr, torch.Tensor):
                    arr = arr.numpy()
                
                # Ensure float32 for image
                if key == "image":
                    arr = arr.astype(np.float32)
                
                # Add channel dim if 3D array
                if arr.ndim == 3:
                    arr = arr[np.newaxis, ...]
                
                d[key] = arr
        return d


class CreateSemanticLabelsd(MapTransform):
    """Create semantic class IDs from instance segmentation labels.
    
    Naming convention (instance segmentation challenge format):
    - 'image': input EM volume
    - 'label': indexed instance segmentation (unique IDs per object)
    - 'class_ids': semantic class for each pixel (background=0, neuron=1, mito=2, ...)
    
    Strategy for multi-class datasets:
    1. Before cropping: Map each instance ID to its semantic class (per-pixel class_ids)
    2. Crop both label and class_ids together (spatial transforms)
    3. After cropping: Relabel instances (connected components) - class_ids stays correct
    
    Example configuration:
        semantic_classes:
            names: ["background", "neuron", "mito"]
            # Map class name -> list of instance IDs in the full volume
            instance_ids:
                neuron: [1, 2, 3, 10, 23, 45, ...]
                mito: [4, 5, 8, 9, 12, ...]
    
    After random crop, instances may be split into multiple components.
    The class_ids (per-pixel) remains correct since it was computed before cropping.
    The label (instance IDs) gets relabeled by connected components in the module.
    
    Args:
        keys: Keys to process (typically ["label"]).
        class_mapping: Dict mapping class_name -> list of instance IDs.
            Example: {"neuron": [1, 2, 3], "mito": [4, 5, 6]}
        class_names: List of class names in order (index = class_id).
            Example: ["background", "neuron", "mito"]
        default_class: Default class for unmapped foreground instances.
    """
    
    def __init__(
        self,
        keys,
        class_mapping: Optional[Dict[str, List[int]]] = None,
        class_names: Optional[List[str]] = None,
        default_class: int = 1,
    ):
        super().__init__(keys)
        self.class_mapping = class_mapping
        self.class_names = class_names or ["background", "foreground"]
        self.default_class = default_class
        
        # Build instance_id -> class_id lookup table
        self.instance_to_class_lut = self._build_lut()
    
    def _build_lut(self) -> Optional[Dict[int, int]]:
        """Build lookup table: instance_id -> class_id."""
        if self.class_mapping is None:
            return None
        
        lut = {}
        for class_name, instance_ids in self.class_mapping.items():
            if class_name in self.class_names:
                class_id = self.class_names.index(class_name)
            else:
                # Skip unknown class names
                continue
            
            for inst_id in instance_ids:
                lut[inst_id] = class_id
        
        return lut
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            if key not in d or key != "label":
                continue
                
            label = d[key]
            
            # Option 1: Use pre-computed semantic labels if available
            if "class_ids" in d:
                continue  # Already has class_ids
            
            # Option 2: Use instance-to-class lookup table
            if self.instance_to_class_lut is not None:
                class_ids = self._map_with_lut(label)
                d["class_ids"] = class_ids
                continue
            
            # Option 3: Check for per-sample mapping in data dict
            if "instance_to_class" in d:
                class_ids = self._map_with_dict(label, d["instance_to_class"])
                d["class_ids"] = class_ids
                continue
            
            # Option 4: Default binary mapping (SNEMI3D style)
            # background = 0, all foreground = default_class
            if isinstance(label, np.ndarray):
                d["class_ids"] = (label > 0).astype(np.int64) * self.default_class
            elif isinstance(label, torch.Tensor):
                d["class_ids"] = (label > 0).long() * self.default_class
        
        return d
    
    def _map_with_lut(
        self, label: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Map instance IDs to class IDs using lookup table."""
        is_numpy = isinstance(label, np.ndarray)
        
        if is_numpy:
            class_ids = np.zeros_like(label, dtype=np.int64)
            # Default: all foreground gets default_class
            class_ids[label > 0] = self.default_class
            # Apply specific mappings from LUT
            for inst_id, class_id in self.instance_to_class_lut.items():
                class_ids[label == inst_id] = class_id
        else:
            class_ids = torch.zeros_like(label, dtype=torch.long)
            class_ids[label > 0] = self.default_class
            for inst_id, class_id in self.instance_to_class_lut.items():
                class_ids[label == inst_id] = class_id
        
        return class_ids
    
    def _map_with_dict(
        self,
        label: Union[np.ndarray, torch.Tensor],
        instance_to_class: Dict[str, List[int]],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Map instance IDs to class IDs using per-sample dict."""
        is_numpy = isinstance(label, np.ndarray)
        
        if is_numpy:
            class_ids = np.zeros_like(label, dtype=np.int64)
            class_ids[label > 0] = self.default_class
        else:
            class_ids = torch.zeros_like(label, dtype=torch.long)
            class_ids[label > 0] = self.default_class
        
        for class_name, instance_ids in instance_to_class.items():
            if class_name in self.class_names:
                class_id = self.class_names.index(class_name)
            else:
                continue
            
            for inst_id in instance_ids:
                if is_numpy:
                    class_ids[label == inst_id] = class_id
                else:
                    class_ids[label == inst_id] = class_id
        
        return class_ids

# Enable Tensor Core optimization
torch.set_float32_matmul_precision("high")


# =============================================================================
# Data Module for 3D Volume Training
# =============================================================================

class Vista3DDataModule(pl.LightningDataModule):
    """
    DataModule for Vista3D training with 3D random cropping.
    
    Loads full volumes and applies random 3D crops during training.
    
    Semantic class configuration:
    - class_names: List of class names ["background", "neuron", "mito", ...]
    - class_mapping: Dict mapping class_name -> list of instance IDs
    - default_class: Default class for unmapped foreground
    """
    
    def __init__(
        self,
        data_root: str,
        dataset: str = "snemi3d",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        batch_size: int = 2,
        num_workers: int = 4,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        augmentation: bool = True,
        num_train_samples: int = 100,
        num_val_samples: int = 20,
        # Semantic class configuration
        class_names: Optional[List[str]] = None,
        class_mapping: Optional[Dict[str, List[int]]] = None,
        default_class: int = 1,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.train_val_split = train_val_split
        self.augmentation = augmentation
        
        # Semantic class configuration
        self.class_names = class_names or ["background", "neuron"]
        self.class_mapping = class_mapping  # None = binary (all foreground = class 1)
        self.default_class = default_class
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _load_snemi3d_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load SNEMI3D data files."""
        from neurocircuitry.preprocessors import HDF5Preprocessor, TIFFPreprocessor
        
        hdf5_proc = HDF5Preprocessor()
        tiff_proc = TIFFPreprocessor()
        
        def find_and_load(base_name: str) -> Optional[np.ndarray]:
            for ext in [".h5", ".hdf5", ".tiff", ".tif"]:
                path = self.data_root / f"{base_name}{ext}"
                if path.exists():
                    if ext in [".h5", ".hdf5"]:
                        return hdf5_proc.load(str(path))
                    else:
                        return tiff_proc.load(str(path))
            return None
        
        # Load AC4 for train/val
        ac4_inputs = find_and_load("AC4_inputs")
        ac4_labels = find_and_load("AC4_labels")
        
        # Load AC3 for test
        ac3_inputs = find_and_load("AC3_inputs")
        ac3_labels = find_and_load("AC3_labels")
        
        if ac4_inputs is None or ac4_labels is None:
            raise FileNotFoundError(
                f"Could not find AC4 data in {self.data_root}. "
                "Expected AC4_inputs.h5 and AC4_labels.h5"
            )
        
        # Split AC4 into train/val
        n_total = ac4_inputs.shape[0]
        n_val = int(n_total * self.train_val_split)
        n_train = n_total - n_val
        
        train_data = [{
            "image": ac4_inputs[:n_train],
            "label": ac4_labels[:n_train],
            "name": "AC4_train",
        }]
        
        val_data = [{
            "image": ac4_inputs[n_train:],
            "label": ac4_labels[n_train:],
            "name": "AC4_val",
        }]
        
        test_data = []
        if ac3_inputs is not None:
            test_data.append({
                "image": ac3_inputs,
                "label": ac3_labels if ac3_labels is not None else np.zeros_like(ac3_inputs),
                "name": "AC3_test",
            })
        
        return train_data, val_data, test_data
    
    def _get_train_transforms(self) -> Compose:
        """Get training transforms with random 3D cropping.
        
        Keys (instance segmentation challenge format):
        - 'image': input EM volume
        - 'label': indexed instance segmentation (IDs: 1, 2, 3, ...)
        - 'class_ids': semantic class for each pixel (background=0, neuron=1, mito=2, ...)
        
        The class_ids is computed BEFORE cropping so it remains correct after spatial transforms.
        The label (instance IDs) is relabeled by connected components in the module AFTER cropping.
        """
        # All spatial keys that need same transforms
        spatial_keys = ["image", "label", "class_ids"]
        
        transforms = [
            AddChannelDimd(keys=["image", "label"]),
            # Create class_ids from label using semantic class mapping
            # class_ids is per-pixel, so cropping won't affect class assignment
            CreateSemanticLabelsd(
                keys=["label"],
                class_mapping=self.class_mapping,
                class_names=self.class_names,
                default_class=self.default_class,
            ),
            AddChannelDimd(keys=["class_ids"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(
                keys=spatial_keys,
                spatial_size=self.patch_size,
                mode="constant",
            ),
            RandSpatialCropd(
                keys=spatial_keys,
                roi_size=self.patch_size,
                random_size=False,
            ),
        ]
        
        if self.augmentation:
            transforms.extend([
                RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=2),
                # Only rotate on axes with same size to preserve shape
                # For patch_size [Z, Y, X] = [32, 128, 128], rotate on Y-X plane (1, 2)
                RandRotate90d(keys=spatial_keys, prob=0.5, spatial_axes=(1, 2)),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            ])
        
        transforms.append(ToTensord(keys=spatial_keys))
        
        return Compose(transforms)
    
    def _get_val_transforms(self) -> Compose:
        """Get validation transforms (center crop, no augmentation).
        
        Keys (instance segmentation challenge format):
        - 'image': input EM volume
        - 'label': indexed instance segmentation (IDs: 1, 2, 3, ...)
        - 'class_ids': semantic class for each pixel (background=0, neuron=1, mito=2, ...)
        """
        # All spatial keys that need same transforms
        spatial_keys = ["image", "label", "class_ids"]
        
        return Compose([
            AddChannelDimd(keys=["image", "label"]),
            CreateSemanticLabelsd(
                keys=["label"],
                class_mapping=self.class_mapping,
                class_names=self.class_names,
                default_class=self.default_class,
            ),
            AddChannelDimd(keys=["class_ids"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(
                keys=spatial_keys,
                spatial_size=self.patch_size,
                mode="constant",
            ),
            CenterSpatialCropd(
                keys=spatial_keys,
                roi_size=self.patch_size,
            ),
            ToTensord(keys=spatial_keys),
        ])
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets."""
        if self.dataset.lower() == "snemi3d":
            train_data, val_data, test_data = self._load_snemi3d_data()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        # Create datasets with random sampling for multiple crops per volume
        if stage == "fit" or stage is None:
            # Replicate data entries for multiple random crops
            train_data_expanded = train_data * self.num_train_samples
            val_data_expanded = val_data * self.num_val_samples
            
            # Print sample counts
            print(f"\n{'='*60}")
            print(f"Dataset Sample Counts:")
            print(f"  Base train volumes: {len(train_data)}")
            print(f"  Base val volumes: {len(val_data)}")
            print(f"  Crops per train volume: {self.num_train_samples}")
            print(f"  Crops per val volume: {self.num_val_samples}")
            print(f"  Total train samples: {len(train_data_expanded)}")
            print(f"  Total val samples: {len(val_data_expanded)}")
            print(f"  Train batches per epoch: {len(train_data_expanded) // self.batch_size}")
            print(f"  Val batches per epoch: {len(val_data_expanded) // self.batch_size}")
            print(f"{'='*60}\n")
            
            self.train_dataset = CacheDataset(
                data=train_data_expanded,
                transform=self._get_train_transforms(),
                cache_rate=min(self.cache_rate, 1.0 / self.num_train_samples),
                num_workers=self.num_workers,
            )
            
            self.val_dataset = CacheDataset(
                data=val_data_expanded,
                transform=self._get_val_transforms(),
                cache_rate=min(self.cache_rate, 1.0 / self.num_val_samples),
                num_workers=self.num_workers,
            )
        
        if stage == "test" or stage is None:
            if test_data:
                print(f"  Test samples: {len(test_data)}")
                self.test_dataset = CacheDataset(
                    data=test_data,
                    transform=self._get_val_transforms(),
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )


# =============================================================================
# Configuration and Setup
# =============================================================================

def load_config(config_path: Optional[str] = None) -> DictConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "snemi3d_vista3d.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    # Merge CLI overrides
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    return cfg


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Create DataModule based on config.
    
    Supports:
    - 'snemi3d': Single SNEMI3D dataset
    - 'cremi3d': Single CREMI3D dataset  
    - 'multi': Combined multi-dataset training
    """
    data_cfg = cfg.data
    dataset_type = data_cfg.get("dataset", "snemi3d")
    
    # Multi-dataset mode
    if dataset_type == "multi":
        from neurocircuitry.datamodules.multi_dataset import get_multi_datamodule
        return get_multi_datamodule(cfg)
    
    # Single dataset mode (snemi3d, cremi3d)
    patch_size = data_cfg.get("patch_size", [128, 128, 128])
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    
    # Get semantic class configuration
    semantic_cfg = data_cfg.get("semantic_classes", {})
    class_names = semantic_cfg.get("names", ["background", "neuron"])
    if isinstance(class_names, (list, tuple)):
        class_names = list(class_names)
    
    # Get instance-to-class mapping (None = binary, all foreground = class 1)
    class_mapping = semantic_cfg.get("instance_ids", None)
    if class_mapping is not None:
        # Convert OmegaConf to regular dict/lists
        class_mapping = {k: list(v) for k, v in class_mapping.items()}
    
    default_class = semantic_cfg.get("default_class", 1)
    
    return Vista3DDataModule(
        data_root=data_cfg.get("data_root", "data/SNEMI3D"),
        dataset=data_cfg.get("dataset", "snemi3d"),
        patch_size=patch_size,
        batch_size=data_cfg.get("batch_size", 2),
        num_workers=data_cfg.get("num_workers", 4),
        cache_rate=data_cfg.get("cache_rate", 1.0),
        train_val_split=data_cfg.get("train_val_split", 0.2),
        augmentation=data_cfg.get("augmentation", True),
        num_train_samples=data_cfg.get("num_train_samples", 100),
        num_val_samples=data_cfg.get("num_val_samples", 20),
        # Semantic class configuration
        class_names=class_names,
        class_mapping=class_mapping,
        default_class=default_class,
    )


def get_module(cfg: DictConfig) -> pl.LightningModule:
    """Create Vista3D Lightning module."""
    from neurocircuitry.modules.vista3d_module import Vista3DModule
    
    model_cfg = dict(cfg.get("model", {}))
    optimizer_cfg = dict(cfg.get("optimizer", {}))
    loss_cfg = dict(cfg.get("loss", {}))
    training_cfg = cfg.get("training", {})
    
    patch_size = cfg.data.get("patch_size", [128, 128, 128])
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    
    # Derive num_classes from semantic_classes.names (master labels list)
    # This ensures model output matches the number of semantic classes
    semantic_cfg = cfg.data.get("semantic_classes", {})
    class_names = semantic_cfg.get("names", ["background", "neuron"])
    if isinstance(class_names, (list, tuple)):
        class_names = list(class_names)
    num_classes = len(class_names)
    
    # Override model config with derived num_classes
    model_cfg["num_classes"] = num_classes
    print(f"Using {num_classes} semantic classes: {class_names}")
    
    return Vista3DModule(
        model_config=model_cfg,
        optimizer_config=optimizer_cfg,
        loss_config=loss_cfg,
        training_mode=training_cfg.get("mode", "auto"),
        num_point_prompts=training_cfg.get("num_point_prompts", 5),
        patch_size=patch_size,
    )


def setup_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Setup training callbacks."""
    from neurocircuitry.callbacks import TensorBoardVolumeCallback
    
    callbacks = []
    callback_cfg = cfg.get("callbacks", {})
    
    # Get log directory for checkpoint saving
    log_dir = cfg.get("log_dir", "logs")
    experiment_name = cfg.get("experiment_name", "vista3d")
    checkpoint_base_dir = os.path.join(log_dir, experiment_name, "checkpoints")
    
    # Model Checkpoint
    ckpt_cfg = callback_cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True):
        # Use dirpath from config if set, otherwise use log_dir/experiment_name/checkpoints
        ckpt_dirpath = ckpt_cfg.get("dirpath")
        if ckpt_dirpath is None:
            ckpt_dirpath = checkpoint_base_dir
        
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dirpath,
                filename=ckpt_cfg.get("filename", "vista3d-{epoch:02d}-{val/dice:.4f}"),
                save_top_k=ckpt_cfg.get("save_top_k", 3),
                monitor=ckpt_cfg.get("monitor", "val/dice"),
                mode=ckpt_cfg.get("mode", "max"),
                save_last=ckpt_cfg.get("save_last", True),
                verbose=True,
                auto_insert_metric_name=False,
            )
        )
        print(f"  Checkpoint dir: {ckpt_dirpath}")
    
    # Early Stopping
    es_cfg = callback_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val/dice"),
                patience=es_cfg.get("patience", 30),
                mode=es_cfg.get("mode", "max"),
                verbose=True,
            )
        )
    
    # TensorBoard Volume Visualization
    tb_vol_cfg = callback_cfg.get("tensorboard_volume", {})
    if tb_vol_cfg.get("enabled", True):
        slice_axes = tb_vol_cfg.get("slice_axes", ["axial"])
        if isinstance(slice_axes, list):
            slice_axes = tuple(slice_axes)
        
        callbacks.append(
            TensorBoardVolumeCallback(
                log_every_n_epochs=tb_vol_cfg.get("log_every_n_epochs", 5),
                num_slices=tb_vol_cfg.get("num_slices", 3),
                slice_axes=slice_axes,
                max_samples=tb_vol_cfg.get("max_samples", 2),
                log_train=tb_vol_cfg.get("log_train", True),
                log_val=tb_vol_cfg.get("log_val", True),
                log_embeddings=tb_vol_cfg.get("log_embeddings", True),
                log_instances=tb_vol_cfg.get("log_instances", True),
                clustering_bandwidth=tb_vol_cfg.get("clustering_bandwidth", 0.5),
            )
        )
        print(f"  TensorBoard volume callback: every {tb_vol_cfg.get('log_every_n_epochs', 5)} epochs")
        print(f"    Log embeddings: {tb_vol_cfg.get('log_embeddings', True)}")
        print(f"    Log instances: {tb_vol_cfg.get('log_instances', True)}")
    
    # Learning Rate Monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Progress Bar
    callbacks.append(RichProgressBar())
    
    # Model Summary
    callbacks.append(ModelSummary(max_depth=3))
    
    return callbacks


def setup_logger(cfg: DictConfig):
    """Setup experiment logger."""
    logger_type = cfg.get("logger", "tensorboard")
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=cfg.get("log_dir", "logs"),
            name=cfg.get("experiment_name", "vista3d"),
        )
    elif logger_type == "wandb":
        return WandbLogger(
            project=cfg.get("project_name", "neurocircuitry-vista3d"),
            name=cfg.get("experiment_name", "vista3d"),
            save_dir=cfg.get("log_dir", "logs"),
        )
    return True


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vista3D Training")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--gpu", type=str, default=None, 
                        help="GPU device ID(s) to use, e.g., '0' or '1,2,3'. Sets CUDA_VISIBLE_DEVICES.")
    args, _ = parser.parse_known_args()
    
    # Set CUDA_VISIBLE_DEVICES if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Print configuration
    print("=" * 70)
    print("Vista3D Training for Connectomics Segmentation")
    print("=" * 70)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    print(f"\nRandom seed: {seed}")
    
    # Initialize DataModule
    datamodule = get_datamodule(cfg)
    print(f"\nDataModule:")
    print(f"  Dataset: {cfg.data.get('dataset', 'snemi3d')}")
    print(f"  Data root: {cfg.data.get('data_root')}")
    print(f"  Patch size: {cfg.data.get('patch_size', [128, 128, 128])}")
    print(f"  Batch size: {cfg.data.get('batch_size', 2)}")
    
    # Initialize Module
    module = get_module(cfg)
    training_mode = cfg.training.get("mode", "auto")
    print(f"\nModule: Vista3DModule")
    print(f"  Training mode: {training_mode}")
    print(f"  Encoder: {cfg.model.get('encoder_name', 'segresnet')}")
    print(f"  Pretrained: {cfg.model.get('pretrained', False)}")
    print(f"  Semantic head dim: {cfg.model.get('sem_head', 16)}")
    print(f"  Instance head dim: {cfg.model.get('ins_head', 16)}")
    print(f"  Semantic head enabled: {cfg.model.get('use_sem_head', True)}")
    print(f"  Instance head enabled: {cfg.model.get('use_ins_head', True)}")
    
    # Print loss configuration
    loss_cfg = cfg.get("loss", {})
    ins_cfg = loss_cfg.get("discriminative", {})
    print(f"\nLoss Configuration:")
    print(f"  CE weight: {loss_cfg.get('ce_weight', 0.5)}")
    print(f"  Dice weight: {loss_cfg.get('dice_weight', 0.5)}")
    print(f"  Instance weight: {loss_cfg.get('ins_weight', 1.0)}")
    if ins_cfg:
        print(f"  Instance loss:")
        print(f"    delta_var: {ins_cfg.get('delta_var', 0.5)}")
        print(f"    delta_dist: {ins_cfg.get('delta_dist', 1.5)}")
    
    # Setup Callbacks
    callbacks = setup_callbacks(cfg)
    print(f"\nCallbacks: {len(callbacks)} registered")
    
    # Setup Logger
    logger = setup_logger(cfg)
    print(f"Logger: {cfg.get('logger', 'tensorboard')}")
    
    # Initialize Trainer
    training_cfg = cfg.training
    
    # Handle devices parameter
    devices = training_cfg.get("devices", 1)
    if isinstance(devices, list):
        devices = list(devices)  # Convert OmegaConf list to Python list
    
    trainer = pl.Trainer(
        max_epochs=training_cfg.get("max_epochs", 200),
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=devices,
        strategy=training_cfg.get("strategy", "auto"),
        precision=training_cfg.get("precision", "16-mixed"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=training_cfg.get("log_every_n_steps", 10),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        val_check_interval=training_cfg.get("val_check_interval", 1.0),
        check_val_every_n_epoch=training_cfg.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=training_cfg.get("num_sanity_val_steps", 2),
        deterministic=training_cfg.get("deterministic", False),
        benchmark=training_cfg.get("benchmark", True),
        fast_dev_run=training_cfg.get("fast_dev_run", False),
    )
    
    print(f"\nTrainer:")
    print(f"  Max epochs: {training_cfg.get('max_epochs', 200)}")
    print(f"  Accelerator: {training_cfg.get('accelerator', 'auto')}")
    print(f"  Devices: {devices}")
    print(f"  Precision: {training_cfg.get('precision', '16-mixed')}")
    
    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")
    
    try:
        trainer.fit(module, datamodule)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        raise
    
    # Save final checkpoint
    if trainer.global_rank == 0:
        final_path = Path("checkpoints/vista3d/final_model.ckpt")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_path))
        print(f"\nFinal model saved: {final_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    
    return trainer.callback_metrics


if __name__ == "__main__":
    main()

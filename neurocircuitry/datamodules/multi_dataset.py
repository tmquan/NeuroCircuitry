"""
Multi-Dataset DataModule for training on combined connectomics datasets.

Simple approach: concatenate existing SNEMI3D and CREMI3D datamodules
and add a transform to consolidate labels with class_ids.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from neurocircuitry.datamodules.snemi3d import SNEMI3DDataModule
from neurocircuitry.datamodules.cremi3d import CREMI3DDataModule


class ExpandedDataset(Dataset):
    """
    Wrapper that virtually expands a dataset by repeating samples.
    
    Useful when you have few samples (e.g., one volume per dataset)
    but want many iterations per epoch. Random transforms ensure
    each access returns different patches/augmentations.
    """
    
    def __init__(self, dataset: Dataset, expansion_factor: int = 100):
        """
        Args:
            dataset: Base dataset to wrap.
            expansion_factor: How many times to virtually repeat the dataset.
        """
        self.dataset = dataset
        self.expansion_factor = expansion_factor
        self._base_len = len(dataset)
    
    def __len__(self):
        return self._base_len * self.expansion_factor
    
    def __getitem__(self, idx):
        # Map expanded index back to base dataset index
        base_idx = idx % self._base_len
        return self.dataset[base_idx]


class CreateClassIds:
    """
    Transform to create class_ids from instance labels.
    
    Wraps a dataset and adds class_ids key based on dataset type.
    - SNEMI3D: all foreground (label > 0) -> class 1 (neuron)
    - CREMI3D: instance ID ranges -> class 1/2/3 (neuron/cleft/mito)
    """
    
    # CREMI ID offsets (must match CREMI3DDataset)
    # CREMI neuron IDs can be up to ~750000, so offsets must be larger
    ID_OFFSET_CLEFT = 1000000   # 1M offset
    ID_OFFSET_MITO = 2000000    # 2M offset
    
    def __init__(
        self,
        dataset: Dataset,
        dataset_dtype: str = "snemi3d",
        default_class: int = 1,
    ):
        """
        Args:
            dataset: Base dataset to wrap.
            dataset_dtype: 'snemi3d' or 'cremi3d'.
            default_class: Default class for foreground.
        """
        self.dataset = dataset
        self.dataset_dtype = dataset_dtype
        self.default_class = default_class
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get label (instance segmentation)
        label = sample["label"]
        
        # Convert to numpy if tensor
        if isinstance(label, torch.Tensor):
            label_np = label.numpy()
        else:
            label_np = np.asarray(label)
        
        # Create class_ids based on dataset type
        if self.dataset_dtype == "snemi3d":
            # SNEMI3D: all foreground is neuron (class 1)
            class_ids = np.zeros_like(label_np, dtype=np.int64)
            class_ids[label_np > 0] = 1
            
        elif self.dataset_dtype == "cremi3d":
            # CREMI3D: map based on ID ranges
            class_ids = np.zeros_like(label_np, dtype=np.int64)
            
            # Neurons: IDs 1 to ID_OFFSET_CLEFT-1
            neuron_mask = (label_np > 0) & (label_np < self.ID_OFFSET_CLEFT)
            class_ids[neuron_mask] = 1
            
            # Clefts: IDs ID_OFFSET_CLEFT to ID_OFFSET_-1
            cleft_mask = (label_np >= self.ID_OFFSET_CLEFT) & (label_np < self.ID_OFFSET_)
            class_ids[cleft_mask] = 2
            
            # Mito: IDs >= ID_OFFSET_
            mito_mask = label_np >= self.ID_OFFSET_
            class_ids[mito_mask] = 3
            
        else:
            # Generic: all foreground to default class
            class_ids = np.zeros_like(label_np, dtype=np.int64)
            class_ids[label_np > 0] = self.default_class
        
        # Add class_ids to sample
        sample = dict(sample)  # Make a copy
        sample["class_ids"] = torch.from_numpy(class_ids) if isinstance(label, torch.Tensor) else class_ids
        sample["dataset_dtype"] = self.dataset_dtype
        
        return sample


class MultiDatasetDataModule(pl.LightningDataModule):
    """
    Lightning DataModule that combines SNEMI3D and CREMI3D datasets.
    
    Simply concatenates datasets from existing datamodules and adds
    a wrapper to create class_ids from instance labels.
    
    Args:
        snemi3d_datamodule: Configured SNEMI3DDataModule (or None to skip).
        cremi3d_datamodule: Configured CREMI3DDataModule (or None to skip).
        batch_size: Override batch size (uses snemi3d's if None).
        num_workers: Override num_workers.
        use_weighted_sampling: Balance sampling across datasets.
        snemi3d_weight: Sampling weight for SNEMI3D.
        cremi3d_weight: Sampling weight for CREMI3D.
    
    Example:
        >>> snemi_dm = SNEMI3DDataModule(data_root="data/SNEMI3D", batch_size=4)
        >>> cremi_dm = CREMI3DDataModule(data_root="data/CREMI3D", samples=["A", "B"])
        >>> 
        >>> multi_dm = MultiDatasetDataModule(
        ...     snemi3d_datamodule=snemi_dm,
        ...     cremi3d_datamodule=cremi_dm,
        ... )
        >>> multi_dm.setup("fit")
        >>> 
        >>> for batch in multi_dm.train_dataloader():
        ...     images = batch["image"]
        ...     labels = batch["label"]      # Instance IDs
        ...     class_ids = batch["class_ids"]  # Semantic classes
    """
    
    def __init__(
        self,
        snemi3d_datamodule: Optional[SNEMI3DDataModule] = None,
        cremi3d_datamodule: Optional[CREMI3DDataModule] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        use_weighted_sampling: bool = True,
        snemi3d_weight: float = 1.0,
        cremi3d_weight: float = 1.0,
        train_expansion_factor: int = 100,
        val_expansion_factor: int = 10,
    ):
        super().__init__()
        
        self.snemi3d_dm = snemi3d_datamodule
        self.cremi3d_dm = cremi3d_datamodule
        
        # Get batch_size from first available datamodule if not specified
        if batch_size is not None:
            self.batch_size = batch_size
        elif snemi3d_datamodule is not None:
            self.batch_size = snemi3d_datamodule.batch_size
        elif cremi3d_datamodule is not None:
            self.batch_size = cremi3d_datamodule.batch_size
        else:
            self.batch_size = 4
        
        if num_workers is not None:
            self.num_workers = num_workers
        elif snemi3d_datamodule is not None:
            self.num_workers = snemi3d_datamodule.num_workers
        elif cremi3d_datamodule is not None:
            self.num_workers = cremi3d_datamodule.num_workers
        else:
            self.num_workers = 4
        
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling
        self.snemi3d_weight = snemi3d_weight
        self.cremi3d_weight = cremi3d_weight
        self.train_expansion_factor = train_expansion_factor
        self.val_expansion_factor = val_expansion_factor
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup combined datasets with expansion for sufficient batch sampling."""
        train_datasets = []
        train_weights = []
        val_datasets = []
        
        # Setup SNEMI3D
        if self.snemi3d_dm is not None:
            self.snemi3d_dm.setup(stage)
            
            if self.snemi3d_dm.train_dataset is not None:
                # Wrap with class_ids creator
                wrapped_train = CreateClassIds(
                    self.snemi3d_dm.train_dataset,
                    dataset_dtype="snemi3d",
                )
                # Expand dataset to create more samples per epoch
                expanded_train = ExpandedDataset(wrapped_train, self.train_expansion_factor)
                train_datasets.append(expanded_train)
                train_weights.extend([self.snemi3d_weight] * len(expanded_train))
            
            if self.snemi3d_dm.val_dataset is not None:
                wrapped_val = CreateClassIds(
                    self.snemi3d_dm.val_dataset,
                    dataset_dtype="snemi3d",
                )
                expanded_val = ExpandedDataset(wrapped_val, self.val_expansion_factor)
                val_datasets.append(expanded_val)
        
        # Setup CREMI3D
        if self.cremi3d_dm is not None:
            self.cremi3d_dm.setup(stage)
            
            if self.cremi3d_dm.train_dataset is not None:
                wrapped_train = CreateClassIds(
                    self.cremi3d_dm.train_dataset,
                    dataset_dtype="cremi3d",
                )
                expanded_train = ExpandedDataset(wrapped_train, self.train_expansion_factor)
                train_datasets.append(expanded_train)
                train_weights.extend([self.cremi3d_weight] * len(expanded_train))
            
            if self.cremi3d_dm.val_dataset is not None:
                wrapped_val = CreateClassIds(
                    self.cremi3d_dm.val_dataset,
                    dataset_dtype="cremi3d",
                )
                expanded_val = ExpandedDataset(wrapped_val, self.val_expansion_factor)
                val_datasets.append(expanded_val)
        
        # Concatenate datasets
        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
            
            if self.use_weighted_sampling:
                self.train_sampler = WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_weights),
                    replacement=True,
                )
        
        if val_datasets:
            self.val_dataset = ConcatDataset(val_datasets)
    
    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler if self.use_weighted_sampling else None,
            shuffle=not self.use_weighted_sampling,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def get_multi_datamodule(cfg) -> MultiDatasetDataModule:
    """
    Create MultiDatasetDataModule from Hydra config.
    
    Config structure:
        data:
          dataset: multi
          datasets:
            snemi3d:
              enabled: true
              data_root: data/SNEMI3D
              weight: 1.0
            cremi3d:
              enabled: true
              data_root: data/CREMI3D
              volumes: ["A", "B"]
              weight: 1.0
    """
    data_cfg = cfg.data
    datasets_cfg = data_cfg.get("datasets", {})
    
    snemi_cfg = datasets_cfg.get("snemi3d", {})
    cremi_cfg = datasets_cfg.get("cremi3d", {})
    
    snemi3d_dm = None
    cremi3d_dm = None
    
    # Get patch_size from config (required for multi-dataset training)
    patch_size = data_cfg.get("patch_size", [32, 128, 128])
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    
    # Create SNEMI3D datamodule if enabled
    if snemi_cfg.get("enabled", True):
        snemi3d_root = snemi_cfg.get("data_root", "data/SNEMI3D")
        if Path(snemi3d_root).exists():
            snemi3d_dm = SNEMI3DDataModule(
                data_root=snemi3d_root,
                batch_size=data_cfg.get("batch_size", 4),
                num_workers=data_cfg.get("num_workers", 4),
                train_val_split=data_cfg.get("train_val_split", 0.2),
                cache_rate=data_cfg.get("cache_rate", 0.5),
                patch_size=patch_size,  # Random crop to patch_size
                slice_mode=False,  # 3D mode for Vista3D
            )
    
    # Create CREMI3D datamodule if enabled
    if cremi_cfg.get("enabled", True):
        cremi3d_root = cremi_cfg.get("data_root", "data/CREMI3D")
        if Path(cremi3d_root).exists():
            cremi3d_dm = CREMI3DDataModule(
                data_root=cremi3d_root,
                batch_size=data_cfg.get("batch_size", 4),
                num_workers=data_cfg.get("num_workers", 4),
                train_val_split=data_cfg.get("train_val_split", 0.2),
                cache_rate=data_cfg.get("cache_rate", 0.5),
                patch_size=patch_size,  # Random crop to patch_size
                volumes=list(cremi_cfg.get("volumes", ["A", "B"])),
                include_clefts=cremi_cfg.get("include_clefts", True),
                include_mito=cremi_cfg.get("include_mito", False),
            )
    
    # Get expansion factors for virtual dataset size
    # This allows sampling many patches from few volumes per epoch
    train_expansion = data_cfg.get("train_expansion_factor", 100)
    val_expansion = data_cfg.get("val_expansion_factor", 10)
    
    return MultiDatasetDataModule(
        snemi3d_datamodule=snemi3d_dm,
        cremi3d_datamodule=cremi3d_dm,
        batch_size=data_cfg.get("batch_size", 4),
        num_workers=data_cfg.get("num_workers", 4),
        use_weighted_sampling=True,
        snemi3d_weight=snemi_cfg.get("weight", 1.0),
        cremi3d_weight=cremi_cfg.get("weight", 1.0),
        train_expansion_factor=train_expansion,
        val_expansion_factor=val_expansion,
    )

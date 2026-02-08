"""
Affinity Segmentation Lightning Module.

Affinity-based boundary prediction for connectomics segmentation.
Used in methods like Flood-Filling Networks and waterz watershed.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat

from neurocircuitry.models import SegResNetWrapper


class AffinitySegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for affinity-based segmentation.
    
    Predicts affinity maps between neighboring voxels, which can then be
    used with watershed or agglomeration algorithms to produce instance
    segmentation.
    
    Supports multiple affinity configurations:
    - Direct neighbor affinities (6-connected in 3D, 4-connected in 2D)
    - Long-range affinities for better boundary detection
    - Anisotropic affinities for EM data with different z-resolution
    
    Args:
        model_config: Model configuration dictionary.
        optimizer_config: Optimizer configuration dictionary.
        loss_config: Loss configuration dictionary.
        affinity_offsets: List of offset tuples defining affinity neighbors.
            Default for 2D: [(0, 1), (1, 0)]
            Default for 3D: [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        spatial_dims: Spatial dimensions (2 or 3).
    
    Example:
        >>> # 2D affinity prediction
        >>> model_config = {
        ...     "in_channels": 1,
        ...     "spatial_dims": 2
        ... }
        >>> module = AffinitySegmentationModule(
        ...     model_config=model_config,
        ...     affinity_offsets=[(0, 1), (1, 0)],  # right, down
        ...     spatial_dims=2
        ... )
        >>> 
        >>> # 3D with long-range affinities
        >>> offsets_3d = [
        ...     (0, 0, 1), (0, 1, 0), (1, 0, 0),  # direct neighbors
        ...     (0, 0, 9), (0, 9, 0), (3, 0, 0),  # long-range
        ... ]
        >>> module = AffinitySegmentationModule(
        ...     model_config={"in_channels": 1, "spatial_dims": 3},
        ...     affinity_offsets=offsets_3d,
        ...     spatial_dims=3
        ... )
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        affinity_offsets: Optional[List[Tuple[int, ...]]] = None,
        spatial_dims: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default configurations
        model_config = model_config or {}
        optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}
        
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        self.spatial_dims = spatial_dims
        
        # Default affinity offsets
        if affinity_offsets is None:
            if spatial_dims == 2:
                self.affinity_offsets = [(0, 1), (1, 0)]  # right, down
            else:
                self.affinity_offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # z, y, x
        else:
            self.affinity_offsets = affinity_offsets
        
        self.num_affinities = len(self.affinity_offsets)
        
        # Model outputs affinity channels
        model_config.setdefault("out_channels", self.num_affinities)
        model_config.setdefault("spatial_dims", spatial_dims)
        
        # Initialize model
        self.model = SegResNetWrapper(**model_config)
        
        # Loss function - weighted BCE for imbalanced boundaries
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([loss_config.get("pos_weight", 1.0)])
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning affinity predictions."""
        outputs = self.model(x)
        outputs["affinity_logits"] = outputs["logits"]
        outputs["affinity"] = torch.sigmoid(outputs["logits"])
        return outputs
    
    def _compute_affinity_targets(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute affinity targets from instance labels.
        
        Affinity is 1 if neighboring voxels belong to the same instance,
        0 if they belong to different instances.
        
        Args:
            labels: Instance segmentation labels [B, H, W] or [B, D, H, W].
        
        Returns:
            Affinity targets [B, num_affinities, ...spatial...].
        """
        batch_size = labels.shape[0]
        spatial_shape = labels.shape[1:]
        device = labels.device
        
        # Initialize affinity tensor
        affinity_shape = (batch_size, self.num_affinities) + spatial_shape
        affinities = torch.zeros(affinity_shape, device=device, dtype=torch.float32)
        
        for i, offset in enumerate(self.affinity_offsets):
            # Shift labels according to offset
            if self.spatial_dims == 2:
                dy, dx = offset
                if dy >= 0 and dx >= 0:
                    labels_shifted = F.pad(labels, (dx, 0, dy, 0))[:, :spatial_shape[0], :spatial_shape[1]]
                    labels_orig = labels
                elif dy >= 0 and dx < 0:
                    labels_shifted = F.pad(labels, (0, -dx, dy, 0))[:, :spatial_shape[0], -dx:]
                    labels_orig = labels[:, :, :-dx] if dx != 0 else labels
                elif dy < 0 and dx >= 0:
                    labels_shifted = F.pad(labels, (dx, 0, 0, -dy))[:, -dy:, :spatial_shape[1]]
                    labels_orig = labels[:, :-dy, :] if dy != 0 else labels
                else:
                    labels_shifted = F.pad(labels, (0, -dx, 0, -dy))[:, -dy:, -dx:]
                    labels_orig = labels[:, :-dy, :-dx]
                
                # Affinity = 1 if same instance (and not background)
                same_instance = (labels_orig == labels_shifted) & (labels_orig > 0)
                
                # Handle padding to match original size
                if dy > 0:
                    same_instance = F.pad(same_instance, (0, 0, 0, dy))
                elif dy < 0:
                    same_instance = F.pad(same_instance, (0, 0, -dy, 0))
                if dx > 0:
                    same_instance = F.pad(same_instance, (0, dx, 0, 0))
                elif dx < 0:
                    same_instance = F.pad(same_instance, (-dx, 0, 0, 0))
                
                affinities[:, i] = same_instance.float()[:, :spatial_shape[0], :spatial_shape[1]]
                
            else:  # 3D
                dz, dy, dx = offset
                # Create shifted version using roll and mask edges
                labels_shifted = torch.roll(labels, shifts=(-dz, -dy, -dx), dims=(1, 2, 3))
                
                # Create mask for valid comparisons (not comparing across boundaries)
                valid_mask = torch.ones_like(labels, dtype=torch.bool)
                if dz > 0:
                    valid_mask[:, -dz:, :, :] = False
                elif dz < 0:
                    valid_mask[:, :-dz, :, :] = False
                if dy > 0:
                    valid_mask[:, :, -dy:, :] = False
                elif dy < 0:
                    valid_mask[:, :, :-dy, :] = False
                if dx > 0:
                    valid_mask[:, :, :, -dx:] = False
                elif dx < 0:
                    valid_mask[:, :, :, :-dx] = False
                
                same_instance = (labels == labels_shifted) & (labels > 0) & valid_mask
                affinities[:, i] = same_instance.float()
        
        return affinities
    
    def _compute_metrics(
        self,
        affinity_pred: torch.Tensor,
        affinity_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute affinity prediction metrics."""
        # Binarize predictions
        pred_binary = (affinity_pred > 0.5).float()
        
        # Overall accuracy
        accuracy = (pred_binary == affinity_target).float().mean()
        
        # Precision and recall for positive class (same instance)
        true_pos = ((pred_binary == 1) & (affinity_target == 1)).float().sum()
        pred_pos = (pred_binary == 1).float().sum()
        actual_pos = (affinity_target == 1).float().sum()
        
        precision = true_pos / (pred_pos + 1e-8)
        recall = true_pos / (actual_pos + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        labels = batch["label"]
        
        # Handle label shape: [B, 1, ...] -> [B, ...]
        if labels.dim() == images.dim():
            labels = rearrange(labels, "b 1 ... -> b ...")
        
        # Forward pass
        outputs = self(images)
        affinity_logits = outputs["affinity_logits"]
        
        # Compute affinity targets
        affinity_targets = self._compute_affinity_targets(labels)
        
        # Compute loss
        loss = self.loss_fn(affinity_logits, affinity_targets)
        
        # Log metrics
        batch_size = images.shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""
        images = batch["image"]
        labels = batch["label"]
        
        # Handle label shape
        if labels.dim() == images.dim():
            labels = rearrange(labels, "b 1 ... -> b ...")
        
        # Forward pass
        outputs = self(images)
        affinity_logits = outputs["affinity_logits"]
        affinity_pred = outputs["affinity"]
        
        # Compute targets and loss
        affinity_targets = self._compute_affinity_targets(labels)
        loss = self.loss_fn(affinity_logits, affinity_targets)
        
        # Compute metrics
        metrics = self._compute_metrics(affinity_pred, affinity_targets)
        
        # Log metrics
        batch_size = images.shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val/accuracy", metrics["accuracy"], batch_size=batch_size)
        self.log("val/precision", metrics["precision"], batch_size=batch_size)
        self.log("val/recall", metrics["recall"], batch_size=batch_size)
        self.log("val/f1", metrics["f1"], prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-3),
            weight_decay=self.optimizer_config.get("weight_decay", 1e-4),
        )
        
        scheduler_config = self.optimizer_config.get("scheduler", {})
        if scheduler_config.get("type") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("T_max", 100),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val/loss",
                },
            }
        
        return optimizer

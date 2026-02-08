"""
Instance Segmentation Lightning Module.

Combines discriminative loss for instance embeddings with semantic segmentation.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange

from neurocircuitry.models import SegResNetWrapper
from neurocircuitry.losses import DiscriminativeLoss


class InstanceSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for instance segmentation.
    
    Uses a multi-task approach combining:
    - Semantic segmentation (foreground/background classification)
    - Instance embeddings (discriminative loss for clustering)
    
    Based on "Semantic Instance Segmentation with a Discriminative Loss Function"
    by De Brabandere et al. (2017).
    
    Args:
        model_config: Model configuration dictionary.
        optimizer_config: Optimizer configuration dictionary.
        loss_config: Loss configuration dictionary with discriminative params.
    
    Example:
        >>> model_config = {
        ...     "in_channels": 1,
        ...     "out_channels": 2,
        ...     "spatial_dims": 2,
        ...     "use_ins_head": True,
        ...     "emb_dim": 16
        ... }
        >>> loss_config = {
        ...     "discriminative": {
        ...         "delta_var": 0.5,
        ...         "delta_dist": 1.5
        ...     },
        ...     "semantic_weight": 1.0,
        ...     "instance_weight": 1.0
        ... }
        >>> 
        >>> module = InstanceSegmentationModule(
        ...     model_config=model_config,
        ...     loss_config=loss_config
        ... )
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default configurations
        model_config = model_config or {}
        optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}
        
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        
        # Ensure instance head is enabled
        model_config.setdefault("use_ins_head", True)
        model_config.setdefault("out_channels", 2)  # fg/bg
        
        # Initialize model
        self.model = SegResNetWrapper(**model_config)
        
        # Loss functions
        disc_config = loss_config.get("discriminative", {})
        self.discriminative_loss = DiscriminativeLoss(
            delta_var=disc_config.get("delta_var", 0.5),
            delta_dist=disc_config.get("delta_dist", 1.5),
            norm=disc_config.get("norm", 2),
            alpha=disc_config.get("alpha", 1.0),
            beta=disc_config.get("beta", 1.0),
            gamma=disc_config.get("gamma", 0.001),
        )
        
        self.semantic_loss = nn.CrossEntropyLoss()
        
        # Loss weights
        self.semantic_weight = loss_config.get("semantic_weight", 1.0)
        self.instance_weight = loss_config.get("instance_weight", 1.0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning model outputs."""
        return self.model(x)
    
    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs with 'logits' and 'embedding'.
            labels: Instance segmentation labels [B, 1, H, W] or [B, H, W].
        
        Returns:
            Dictionary with individual and total losses.
        """
        # Ensure labels shape: [B, 1, H, W] -> [B, H, W]
        if labels.dim() == 4:
            labels_squeezed = rearrange(labels, "b 1 h w -> b h w")
        else:
            labels_squeezed = labels
        
        # Semantic loss: foreground (label > 0) vs background (label == 0)
        semantic_logits = outputs["logits"]
        semantic_target = (labels_squeezed > 0).long()
        loss_semantic = self.semantic_loss(semantic_logits, semantic_target)
        
        # Discriminative loss for instance embeddings
        embedding = outputs["embedding"]
        loss_disc, loss_var, loss_dist, loss_reg = self.discriminative_loss(
            embedding, labels_squeezed
        )
        
        # Total loss
        total_loss = (
            self.semantic_weight * loss_semantic +
            self.instance_weight * loss_disc
        )
        
        return {
            "loss": total_loss,
            "loss_semantic": loss_semantic,
            "loss_disc": loss_disc,
            "loss_var": loss_var,
            "loss_dist": loss_dist,
            "loss_reg": loss_reg,
        }
    
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute segmentation metrics."""
        # Ensure labels shape
        if labels.dim() == 4:
            labels_squeezed = rearrange(labels, "b 1 h w -> b h w")
        else:
            labels_squeezed = labels
        
        # Semantic predictions
        semantic_pred = outputs["logits"].argmax(dim=1)
        semantic_target = (labels_squeezed > 0).long()
        
        # Foreground IoU
        intersection = ((semantic_pred == 1) & (semantic_target == 1)).float().sum()
        union = ((semantic_pred == 1) | (semantic_target == 1)).float().sum()
        iou = intersection / (union + 1e-8)
        
        # Accuracy
        accuracy = (semantic_pred == semantic_target).float().mean()
        
        return {
            "iou": iou,
            "accuracy": accuracy,
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        labels = batch["label"]
        
        # Forward pass
        outputs = self(images)
        
        # Compute losses
        loss_dict = self._compute_losses(outputs, labels)
        loss = loss_dict["loss"]
        
        # Log losses
        batch_size = images.shape[0]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_semantic", loss_dict["loss_semantic"], on_epoch=True, batch_size=batch_size)
        self.log("train/loss_disc", loss_dict["loss_disc"], on_epoch=True, batch_size=batch_size)
        self.log("train/loss_var", loss_dict["loss_var"], on_epoch=True, batch_size=batch_size)
        self.log("train/loss_dist", loss_dict["loss_dist"], on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""
        images = batch["image"]
        labels = batch["label"]
        
        # Forward pass
        outputs = self(images)
        
        # Compute losses and metrics
        loss_dict = self._compute_losses(outputs, labels)
        metrics = self._compute_metrics(outputs, labels)
        
        # Log metrics
        batch_size = images.shape[0]
        self.log("val/loss", loss_dict["loss"], prog_bar=True, batch_size=batch_size)
        self.log("val/loss_semantic", loss_dict["loss_semantic"], batch_size=batch_size)
        self.log("val/loss_disc", loss_dict["loss_disc"], batch_size=batch_size)
        self.log("val/iou", metrics["iou"], prog_bar=True, batch_size=batch_size)
        self.log("val/accuracy", metrics["accuracy"], batch_size=batch_size)
        
        return loss_dict["loss"]
    
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

"""
Vista3D Lightning Module for training and inference.

Supports both automatic segmentation and interactive point-based segmentation.
Features:
- Semantic head (sem_head) for foreground/background classification
- Instance head (ins_head) for pixel embeddings with discriminative loss
- ARI/AMI metrics for instance segmentation evaluation
- Label relabeling after cropping
- MONAI MeanDice and MeanIoU metrics
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
import numpy as np
from monai.metrics import DiceMetric, MeanIoU


class Vista3DModule(pl.LightningModule):
    """
    PyTorch Lightning module for Vista3D-based segmentation.
    
    Supports:
    - Automatic segmentation mode (class label prompts)
    - Interactive segmentation mode (point click prompts)
    - Mixed training with both modes
    - Instance segmentation with discriminative loss
    
    Args:
        model_config: Model configuration dict.
        optimizer_config: Optimizer configuration dict.
        loss_config: Loss function configuration dict.
        training_mode: 'auto', 'interactive', or 'mixed'.
        num_point_prompts: Number of point prompts to sample during training.
        patch_size: Size of training patches (D, H, W).
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_mode: str = "auto",
        num_point_prompts: int = 5,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.loss_config = loss_config or {}
        self.training_mode = training_mode
        self.num_point_prompts = num_point_prompts
        self.patch_size = patch_size
        
        # Semantic and instance head dimensions
        self.sem_head_dim = self.model_config.get("sem_head", 16)
        self.ins_head_dim = self.model_config.get("ins_head", 16)
        self.use_sem_head = self.model_config.get("use_sem_head", True)
        self.use_ins_head = self.model_config.get("use_ins_head", True)
        
        # Build model
        self._build_model()
        
        # Build loss functions
        self._build_loss()
    
    def _build_model(self) -> None:
        """Build Vista3D model with optional semantic and instance heads."""
        from neurocircuitry.models.vista3d import Vista3DWrapper
        
        # Get feature size from model config
        feature_size = self.model_config.get("feature_size", 48)
        num_classes = self.model_config.get("num_classes", 2)
        
        self.model = Vista3DWrapper(
            in_channels=self.model_config.get("in_channels", 1),
            num_classes=num_classes,
            pretrained=self.model_config.get("pretrained", False),
            freeze_encoder=self.model_config.get("freeze_encoder", False),
            encoder_name=self.model_config.get("encoder_name", "segresnet"),
            feature_size=feature_size,
            use_point_prompts=self.training_mode in ["interactive", "mixed"],
            use_automatic_mode=self.training_mode in ["auto", "mixed"],
        )
        
        # Semantic embedding head: projects features to sem_head_dim then to num_classes
        if self.use_sem_head:
            self.sem_head = nn.Sequential(
                nn.Conv3d(num_classes, self.sem_head_dim, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.sem_head_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.sem_head_dim, num_classes, kernel_size=1),
            )
        else:
            # Identity: use base model output directly
            self.sem_head = nn.Identity()
        
        # Instance embedding head: projects features to ins_head_dim dimensional embeddings
        if self.use_ins_head:
            self.ins_head = nn.Sequential(
                nn.Conv3d(num_classes, self.ins_head_dim * 2, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.ins_head_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.ins_head_dim * 2, self.ins_head_dim, kernel_size=1),
            )
    
    def _build_loss(self) -> None:
        """Build loss functions including discriminative loss."""
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(self.loss_config.get("class_weights", [1.0, 1.0])),
            ignore_index=self.loss_config.get("ignore_index", -100),
        )
        
        self.dice_loss_weight = self.loss_config.get("dice_weight", 0.5)
        self.ce_loss_weight = self.loss_config.get("ce_weight", 0.5)
        self.boundary_loss_weight = self.loss_config.get("boundary_weight", 0.0)
        
        # Discriminative loss for instance embeddings
        if self.use_ins_head:
            from neurocircuitry.losses.discriminative import DiscriminativeLossVectorized
            
            disc_config = self.loss_config.get("discriminative", {})
            self.discriminative_loss = DiscriminativeLossVectorized(
                delta_var=disc_config.get("delta_var", 0.5),
                delta_dist=disc_config.get("delta_dist", 1.5),
                norm=disc_config.get("norm", 2),
                alpha=disc_config.get("alpha", 1.0),
                beta=disc_config.get("beta", 1.0),
                gamma=disc_config.get("gamma", 0.001),
            )
            self.disc_loss_weight = self.loss_config.get("disc_weight", 1.0)
        
        # MONAI metrics for evaluation
        num_classes = self.model_config.get("num_classes", 2)
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )
        self.iou_metric = MeanIoU(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional semantic and instance heads.
        
        Returns:
            Dict with:
                - 'logits': Semantic logits [B, num_classes, D, H, W]
                - 'embeds': Instance embeddings [B, ins_head_dim, D, H, W] (if use_ins_head)
                - 'labels': Semantic labels after sem_head [B, num_classes, D, H, W]
                - 'membrs': Membrane/boundary predictions (if available)
        """
        # Get base model outputs
        base_outputs = self.model(
            x,
            point_coords=point_coords,
            point_labels=point_labels,
            class_ids=class_ids,
        )
        
        logits = base_outputs["logits"]
        
        # Apply semantic head (or identity if use_sem_head=False)
        semantic = self.sem_head(logits)
        
        outputs = {
            "logits": semantic,  # Use semantic head output as final logits
            "labels": semantic,
        }
        
        # Apply instance embedding head
        if self.use_ins_head:
            embeds = self.ins_head(logits)
            outputs["embeds"] = embeds
        
        return outputs
    
    def _sample_point_prompts(
        self,
        labels: torch.Tensor,
        num_points: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample point prompts from ground truth labels.
        
        Args:
            labels: Ground truth labels [B, D, H, W].
            num_points: Number of points to sample per image.
        
        Returns:
            point_coords: [B, num_points, 3] coordinates (z, y, x).
            point_labels: [B, num_points] labels (0=bg, 1=fg).
        """
        B = labels.shape[0]
        device = labels.device
        
        point_coords_list = []
        point_labels_list = []
        
        for b in range(B):
            label = labels[b]
            
            # Get foreground and background locations
            fg_mask = label > 0
            bg_mask = label == 0
            
            fg_indices = torch.nonzero(fg_mask, as_tuple=False)
            bg_indices = torch.nonzero(bg_mask, as_tuple=False)
            
            # Sample points
            coords = []
            point_labs = []
            
            # Sample foreground points
            n_fg = min(num_points // 2 + 1, len(fg_indices))
            if n_fg > 0 and len(fg_indices) > 0:
                perm = torch.randperm(len(fg_indices), device=device)[:n_fg]
                fg_points = fg_indices[perm]
                coords.append(fg_points)
                point_labs.extend([1] * n_fg)
            
            # Sample background points
            n_bg = num_points - n_fg
            if n_bg > 0 and len(bg_indices) > 0:
                perm = torch.randperm(len(bg_indices), device=device)[:n_bg]
                bg_points = bg_indices[perm]
                coords.append(bg_points)
                point_labs.extend([0] * n_bg)
            
            if len(coords) > 0:
                coords = torch.cat(coords, dim=0)
            else:
                # Fallback: random points
                coords = torch.zeros((num_points, 3), dtype=torch.long, device=device)
                point_labs = [0] * num_points
            
            # Pad if needed
            if coords.shape[0] < num_points:
                pad_size = num_points - coords.shape[0]
                coords = F.pad(coords, (0, 0, 0, pad_size), value=0)
                point_labs.extend([-1] * pad_size)  # -1 = ignore
            
            point_coords_list.append(coords[:num_points].float())
            point_labels_list.append(torch.tensor(point_labs[:num_points], device=device))
        
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)
        
        return point_coords, point_labels
    
    def _compute_dice_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        smooth: float = 1e-5,
    ) -> torch.Tensor:
        """Compute Dice loss."""
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode labels (ensure valid range)
        num_classes = logits.shape[1]
        labels_clamped = torch.clamp(labels.long(), 0, num_classes - 1)
        labels_one_hot = F.one_hot(labels_clamped, num_classes)
        labels_one_hot = rearrange(labels_one_hot, "b d h w c -> b c d h w").float()
        
        # Compute Dice per class
        intersection = (probs * labels_one_hot).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + labels_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        
        # Average over classes (excluding background optionally)
        if num_classes > 1:
            return 1 - dice[:, 1:].mean()  # Exclude background
        return 1 - dice.mean()
    
    def _compute_boundary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary-aware loss using gradient magnitude."""
        # Compute predicted boundaries
        probs = F.softmax(logits, dim=1)[:, 1:]  # Foreground prob
        
        # Sobel-like gradient
        grad_x = probs[:, :, :, :, 1:] - probs[:, :, :, :, :-1]
        grad_y = probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :]
        grad_z = probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :]
        
        # Ground truth boundaries: [B, D, H, W] -> [B, 1, D, H, W]
        labels_float = rearrange(labels.float(), "b d h w -> b 1 d h w")
        gt_grad_x = labels_float[:, :, :, :, 1:] - labels_float[:, :, :, :, :-1]
        gt_grad_y = labels_float[:, :, :, 1:, :] - labels_float[:, :, :, :-1, :]
        gt_grad_z = labels_float[:, :, 1:, :, :] - labels_float[:, :, :-1, :, :]
        
        # Boundary loss
        loss_x = F.mse_loss(grad_x.abs(), gt_grad_x.abs())
        loss_y = F.mse_loss(grad_y.abs(), gt_grad_y.abs())
        loss_z = F.mse_loss(grad_z.abs(), gt_grad_z.abs())
        
        return (loss_x + loss_y + loss_z) / 3
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss including discriminative loss for instance segmentation.
        
        Args:
            logits: Semantic logits [B, num_classes, D, H, W].
            labels: Binary labels [B, D, H, W] (0=background, 1=foreground).
            embedding: Instance embeddings [B, E, D, H, W] (optional).
            instance_labels: Instance labels [B, D, H, W] with unique IDs per instance (optional).
        """
        losses = {}
        
        # Ensure labels are in valid range [0, num_classes-1]
        num_classes = logits.shape[1]
        labels = labels.long()
        labels = torch.clamp(labels, 0, num_classes - 1)
        
        # Cross entropy loss
        ce_loss = self.ce_loss(logits, labels)
        losses["ce_loss"] = ce_loss
        
        # Dice loss
        dice_loss = self._compute_dice_loss(logits, labels)
        losses["dice_loss"] = dice_loss
        
        # Boundary loss (optional)
        if self.boundary_loss_weight > 0:
            boundary_loss = self._compute_boundary_loss(logits, labels)
            losses["boundary_loss"] = boundary_loss
        
        # Discriminative loss for instance embeddings
        if self.use_ins_head and embedding is not None and instance_labels is not None:
            disc_loss, loss_var, loss_dist, loss_reg = self.discriminative_loss(
                embedding, instance_labels
            )
            losses["disc_loss"] = disc_loss
            losses["loss_var"] = loss_var
            losses["loss_dist"] = loss_dist
            losses["loss_reg"] = loss_reg
        
        # Combined loss
        total_loss = (
            self.ce_loss_weight * ce_loss +
            self.dice_loss_weight * dice_loss
        )
        
        if self.boundary_loss_weight > 0:
            total_loss += self.boundary_loss_weight * losses["boundary_loss"]
        
        if self.use_ins_head and "disc_loss" in losses:
            total_loss += self.disc_loss_weight * losses["disc_loss"]
        
        losses["loss"] = total_loss
        
        return losses
    
    def _compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        instance_labels: Optional[torch.Tensor] = None,
        compute_instance_metrics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation metrics using MONAI's MeanDice and MeanIoU.
        
        Args:
            logits: Semantic logits [B, num_classes, D, H, W].
            labels: Binary labels [B, D, H, W].
            embedding: Instance embeddings [B, E, D, H, W] (optional).
            instance_labels: Ground truth instance labels [B, D, H, W] (optional).
            compute_instance_metrics: Whether to compute ARI/AMI (expensive).
        """
        num_classes = logits.shape[1]
        
        # Get predictions as one-hot: [B, D, H, W] -> [B, C, D, H, W]
        preds = logits.argmax(dim=1)
        preds_one_hot = F.one_hot(preds, num_classes)
        preds_one_hot = rearrange(preds_one_hot, "b d h w c -> b c d h w")
        
        # Ensure labels are binary (0 or 1) and convert to one-hot
        labels_binary = (labels > 0).long()
        labels_one_hot = F.one_hot(labels_binary, num_classes)
        labels_one_hot = rearrange(labels_one_hot, "b d h w c -> b c d h w")
        
        # MONAI MeanDice metric
        self.dice_metric.reset()
        self.dice_metric(y_pred=preds_one_hot, y=labels_one_hot)
        mean_dice = self.dice_metric.aggregate()
        
        # MONAI MeanIoU metric
        self.iou_metric.reset()
        self.iou_metric(y_pred=preds_one_hot, y=labels_one_hot)
        mean_iou = self.iou_metric.aggregate()
        
        # Accuracy (simple pixel-wise)
        accuracy = (preds == labels_binary).float().mean()
        
        metrics = {
            "accuracy": accuracy,
            "dice": mean_dice,
            "iou": mean_iou,
        }
        
        # Compute ARI/AMI for instance segmentation (expensive, only first sample)
        if compute_instance_metrics and self.use_ins_head and embedding is not None and instance_labels is not None:
            from neurocircuitry.utils.labels import (
                compute_ari_ami,
                cluster_embeddings_meanshift,
            )
            
            # Get bandwidth from discriminative loss config
            disc_config = self.loss_config.get("discriminative", {})
            bandwidth = disc_config.get("delta_var", 0.5)
            
            # Only compute for first sample (expensive)
            emb_first = embedding[0]  # [E, D, H, W]
            fg_mask = preds[0]  # [D, H, W]
            gt_labels = instance_labels[0]  # [D, H, W]
            
            # Cluster embeddings to get predicted instances
            pred_instances = cluster_embeddings_meanshift(
                emb_first,
                foreground_mask=fg_mask,
                bandwidth=bandwidth,
                min_cluster_size=50,
            )
            
            # Compute ARI and AMI
            ari, ami = compute_ari_ami(pred_instances, gt_labels)
            
            metrics["ari"] = torch.tensor(ari, device=logits.device)
            metrics["ami"] = torch.tensor(ami, device=logits.device)
        
        return metrics
    
    def _relabel_after_crop(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Relabel instance labels after cropping.
        
        After cropping, some instances may be split or entirely removed.
        This function:
        1. Finds connected components (to separate split instances)
        2. Relabels sequentially (to have consecutive IDs)
        
        Args:
            labels: Instance labels [B, D, H, W] with unique IDs per instance.
        
        Returns:
            Relabeled tensor with sequential unique labels per connected component.
        """
        from neurocircuitry.utils.labels import relabel_after_crop
        return relabel_after_crop(labels, spatial_dims=3)
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step with instance segmentation support.
        
        Batch keys (instance segmentation challenge format):
        - 'image': input EM volume
        - 'label': indexed instance segmentation (IDs: 1, 2, 3, ...)
        - 'class_ids': semantic class mapping (background=0, foreground=1)
        """
        images = batch["image"]
        instance_label = batch["label"]  # Instance IDs for discriminative loss
        class_ids = batch["class_ids"]  # Semantic classes for CE/Dice loss
        
        # Ensure images are 5D: [B, C, D, H, W]
        if images.dim() == 4:
            images = rearrange(images, "b d h w -> b 1 d h w")
        
        # Ensure instance labels are 4D: [B, D, H, W]
        if instance_label.dim() == 5:
            instance_label = rearrange(instance_label, "b 1 d h w -> b d h w")
        elif instance_label.dim() == 3:
            instance_label = rearrange(instance_label, "d h w -> 1 d h w")
        
        # Ensure class_ids are 4D: [B, D, H, W]
        if class_ids.dim() == 5:
            class_ids = rearrange(class_ids, "b 1 d h w -> b d h w")
        elif class_ids.dim() == 3:
            class_ids = rearrange(class_ids, "d h w -> 1 d h w")
        
        # Relabel instance labels after cropping (split instances get unique labels)
        instance_labels = self._relabel_after_crop(instance_label)
        
        # Use class_ids directly (already 0=background, 1=foreground)
        binary_labels = class_ids.long()
        
        # Forward pass based on training mode
        if self.training_mode == "auto":
            # Automatic mode with class labels
            outputs = self.forward(images)
        
        elif self.training_mode == "interactive":
            # Interactive mode with point prompts
            point_coords, point_labels_prompt = self._sample_point_prompts(
                binary_labels, self.num_point_prompts
            )
            outputs = self.forward(
                images,
                point_coords=point_coords,
                point_labels=point_labels_prompt,
            )
        
        else:  # mixed
            # Alternate between modes
            if batch_idx % 2 == 0:
                outputs = self.forward(images)
            else:
                point_coords, point_labels_prompt = self._sample_point_prompts(
                    binary_labels, self.num_point_prompts
                )
                outputs = self.forward(
                    images,
                    point_coords=point_coords,
                    point_labels=point_labels_prompt,
                )
        
        logits = outputs["logits"]
        embedding = outputs.get("embeds")
        
        # Compute losses (semantic + discriminative)
        losses = self._compute_loss(
            logits, binary_labels,
            embedding=embedding,
            instance_labels=instance_labels,
        )
        
        # Log losses
        batch_size = images.shape[0]
        for name, value in losses.items():
            self.log(f"train/{name}", value, prog_bar=(name == "loss"), batch_size=batch_size)
        
        return losses["loss"]
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Validation step with instance segmentation metrics.
        
        Batch keys (instance segmentation challenge format):
        - 'image': input EM volume
        - 'label': indexed instance segmentation (IDs: 1, 2, 3, ...)
        - 'class_ids': semantic class mapping (background=0, foreground=1)
        """
        images = batch["image"]
        instance_label = batch["label"]  # Instance IDs for metrics
        class_ids = batch["class_ids"]  # Semantic classes for loss
        
        # Ensure images are 5D: [B, C, D, H, W]
        if images.dim() == 4:
            images = rearrange(images, "b d h w -> b 1 d h w")
        
        # Ensure instance labels are 4D: [B, D, H, W]
        if instance_label.dim() == 5:
            instance_label = rearrange(instance_label, "b 1 d h w -> b d h w")
        
        # Ensure class_ids are 4D: [B, D, H, W]
        if class_ids.dim() == 5:
            class_ids = rearrange(class_ids, "b 1 d h w -> b d h w")
        
        # Relabel instance labels after cropping
        instance_labels = self._relabel_after_crop(instance_label)
        
        # Use class_ids directly
        binary_labels = class_ids.long()
        
        # Use automatic mode for validation
        outputs = self.forward(images)
        logits = outputs["logits"]
        embedding = outputs.get("embeds")
        
        # Compute losses (semantic + discriminative)
        losses = self._compute_loss(
            logits, binary_labels,
            embedding=embedding,
            instance_labels=instance_labels,
        )
        
        # Compute metrics (including ARI/AMI on first batch only - expensive)
        compute_instance = (batch_idx == 0) and self.use_ins_head
        metrics = self._compute_metrics(
            logits, binary_labels,
            embedding=embedding,
            instance_labels=instance_labels,
            compute_instance_metrics=compute_instance,
        )
        
        # Log losses and metrics with explicit batch_size
        batch_size = images.shape[0]
        for name, value in losses.items():
            self.log(f"val/{name}", value, prog_bar=(name == "loss"), sync_dist=True, batch_size=batch_size)
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return losses
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        opt_type = self.optimizer_config.get("type", "adamw").lower()
        lr = self.optimizer_config.get("lr", 1e-4)
        weight_decay = self.optimizer_config.get("weight_decay", 1e-5)
        
        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get("betas", (0.9, 0.999)),
            )
        elif opt_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=self.optimizer_config.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        # Scheduler
        scheduler_cfg = self.optimizer_config.get("scheduler", {})
        scheduler_type = scheduler_cfg.get("type", "cosine").lower()
        
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get("T_max", 100),
                eta_min=scheduler_cfg.get("eta_min", 1e-7),
            )
        elif scheduler_type == "cosine_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_cfg.get("T_0", 10),
                T_mult=scheduler_cfg.get("T_mult", 2),
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_cfg.get("step_size", 30),
                gamma=scheduler_cfg.get("gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 10),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }
        else:
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    
    # =========================================================================
    # Inference Methods
    # =========================================================================
    
    @torch.no_grad()
    def predict_sliding_window(
        self,
        volume: torch.Tensor,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        stride: Optional[Tuple[int, int, int]] = None,
        mode: str = "gaussian",
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Sliding window inference with overlap and aggregation.
        
        Args:
            volume: Input volume [C, D, H, W] or [D, H, W].
            patch_size: Size of patches (D, H, W).
            stride: Stride between patches. Default: patch_size // 2.
            mode: Aggregation mode ('gaussian', 'average', 'max').
            progress: Show progress bar.
        
        Returns:
            Segmentation logits [num_classes, D, H, W].
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Add channel dim if needed: [D, H, W] -> [1, D, H, W]
        if volume.dim() == 3:
            volume = rearrange(volume, "d h w -> 1 d h w")
        
        C, D, H, W = volume.shape
        pd, ph, pw = patch_size
        
        # Default stride: 50% overlap
        if stride is None:
            stride = (pd // 2, ph // 2, pw // 2)
        sd, sh, sw = stride
        
        # Compute number of patches
        nd = max(1, (D - pd) // sd + 1)
        nh = max(1, (H - ph) // sh + 1)
        nw = max(1, (W - pw) // sw + 1)
        
        # Pad volume if needed
        pad_d = max(0, (nd - 1) * sd + pd - D)
        pad_h = max(0, (nh - 1) * sh + ph - H)
        pad_w = max(0, (nw - 1) * sw + pw - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect")
        
        D_pad, H_pad, W_pad = volume.shape[1:]
        
        # Initialize output and weight tensors
        num_classes = self.model_config.get("num_classes", 2)
        output = torch.zeros((num_classes, D_pad, H_pad, W_pad), device=device)
        weight = torch.zeros((1, D_pad, H_pad, W_pad), device=device)
        
        # Create weight map for blending
        if mode == "gaussian":
            # Gaussian weight map
            sigma = min(patch_size) / 4
            z = torch.arange(pd, device=device).float() - pd / 2
            y = torch.arange(ph, device=device).float() - ph / 2
            x = torch.arange(pw, device=device).float() - pw / 2
            zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
            gaussian = torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))
            # Add channel dim: [D, H, W] -> [1, D, H, W]
            patch_weight = rearrange(gaussian, "d h w -> 1 d h w")
        else:
            patch_weight = torch.ones((1, pd, ph, pw), device=device)
        
        # Iterate over patches
        total_patches = nd * nh * nw
        patch_idx = 0
        
        iterator = range(nd)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Sliding window inference")
            except ImportError:
                pass
        
        for i in iterator:
            for j in range(nh):
                for k in range(nw):
                    # Extract patch
                    d_start = i * sd
                    h_start = j * sh
                    w_start = k * sw
                    
                    patch = volume[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ]
                    
                    # Forward pass: [C, D, H, W] -> [1, C, D, H, W]
                    patch_input = rearrange(patch, "c d h w -> 1 c d h w").to(device)
                    outputs = self.forward(patch_input)
                    # Remove batch dim: [1, num_classes, pd, ph, pw] -> [num_classes, pd, ph, pw]
                    logits = rearrange(outputs["logits"], "1 c d h w -> c d h w")
                    
                    # Aggregate
                    output[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += logits * patch_weight
                    
                    weight[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += patch_weight
                    
                    patch_idx += 1
        
        # Normalize by weight
        output = output / (weight + 1e-8)
        
        # Remove padding
        output = output[:, :D, :H, :W]
        
        return output
    
    @torch.no_grad()
    def predict_interactive(
        self,
        volume: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
    ) -> torch.Tensor:
        """
        Interactive segmentation with point prompts.
        
        Args:
            volume: Input volume [C, D, H, W] or [D, H, W].
            point_coords: Point coordinates [N, 3] (z, y, x).
            point_labels: Point labels [N] (0=bg, 1=fg, -1=ignore).
            patch_size: Size of patch around points.
        
        Returns:
            Binary segmentation mask [D, H, W].
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Add channel dim if needed: [D, H, W] -> [1, D, H, W]
        if volume.dim() == 3:
            volume = rearrange(volume, "d h w -> 1 d h w")
        
        C, D, H, W = volume.shape
        pd, ph, pw = patch_size
        
        # Initialize output
        output = torch.zeros((D, H, W), device=device)
        
        # Process each point
        point_coords = point_coords.to(device)
        point_labels = point_labels.to(device)
        
        # Get center point for patch extraction
        center = point_coords.mean(dim=0).long()
        z, y, x = center.tolist()
        
        # Extract patch centered on points
        d_start = max(0, z - pd // 2)
        h_start = max(0, y - ph // 2)
        w_start = max(0, x - pw // 2)
        
        d_end = min(D, d_start + pd)
        h_end = min(H, h_start + ph)
        w_end = min(W, w_start + pw)
        
        # Adjust start if we hit the boundary
        d_start = max(0, d_end - pd)
        h_start = max(0, h_end - ph)
        w_start = max(0, w_end - pw)
        
        patch = volume[
            :,
            d_start:d_end,
            h_start:h_end,
            w_start:w_end,
        ]
        
        # Adjust point coordinates to patch space
        offset = torch.tensor([d_start, h_start, w_start], device=device)
        local_coords = point_coords - offset
        
        # Forward pass: [C, D, H, W] -> [1, C, D, H, W]
        patch_input = rearrange(patch, "c d h w -> 1 c d h w")
        # [N, 3] -> [1, N, 3], [N] -> [1, N]
        outputs = self.forward(
            patch_input,
            point_coords=rearrange(local_coords, "n c -> 1 n c"),
            point_labels=rearrange(point_labels, "n -> 1 n"),
        )
        
        # Extract prediction: [1, num_classes, D, H, W] -> [num_classes, D, H, W]
        logits = rearrange(outputs["logits"], "1 c d h w -> c d h w")
        pred = logits.argmax(dim=0)
        
        # Place in output
        output[d_start:d_end, h_start:h_end, w_start:w_end] = pred.float()
        
        return output

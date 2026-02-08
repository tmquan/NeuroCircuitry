"""
TensorBoard Volume Visualization Callback.

Logs 3D volume slices to TensorBoard during training and validation.
Includes:
- Input images and ground truth labels
- Semantic predictions
- Instance embeddings (PCA projection to RGB)
- Instance segmentation results (clustered instances)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange
import numpy as np


class TensorBoardVolumeCallback(Callback):
    """
    Callback to visualize 3D volume slices in TensorBoard.
    
    Logs input images, ground truth labels, model predictions, instance
    embeddings, and instance segmentation as 2D slices at specified intervals.
    
    Args:
        log_every_n_epochs: Log every N epochs (default: 1).
        num_slices: Number of slices to log per axis (default: 3).
        slice_axes: Axes to slice along ("axial", "sagittal", "coronal").
        max_samples: Maximum number of samples to log per batch (default: 4).
        cmap_image: Colormap for images ("gray" or "viridis").
        cmap_label: Colormap for labels ("tab10" or custom).
        normalize_images: Normalize images to [0, 1] for display.
        log_train: Log training samples (default: True).
        log_val: Log validation samples (default: True).
        log_embeddings: Log instance embeddings as PCA projection (default: True).
        log_instances: Log clustered instance segmentation (default: True).
        clustering_bandwidth: Bandwidth for mean-shift clustering (default: 0.5).
    
    Example:
        >>> callback = TensorBoardVolumeCallback(
        ...     log_every_n_epochs=5,
        ...     num_slices=5,
        ...     max_samples=2,
        ...     log_embeddings=True,
        ...     log_instances=True,
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])
    """
    
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        num_slices: int = 3,
        slice_axes: Tuple[str, ...] = ("axial",),
        max_samples: int = 4,
        cmap_image: str = "gray",
        cmap_label: str = "tab10",
        normalize_images: bool = True,
        log_train: bool = True,
        log_val: bool = True,
        log_embeddings: bool = True,
        log_instances: bool = True,
        clustering_bandwidth: float = 0.5,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_slices = num_slices
        self.slice_axes = slice_axes
        self.max_samples = max_samples
        self.cmap_image = cmap_image
        self.cmap_label = cmap_label
        self.normalize_images = normalize_images
        self.log_train = log_train
        self.log_val = log_val
        self.log_embeddings = log_embeddings
        self.log_instances = log_instances
        self.clustering_bandwidth = clustering_bandwidth
        
        # Store batch data for logging
        self._train_batch: Optional[Dict[str, torch.Tensor]] = None
        self._val_batch: Optional[Dict[str, torch.Tensor]] = None
        self._train_outputs: Optional[Dict[str, torch.Tensor]] = None
        self._val_outputs: Optional[Dict[str, torch.Tensor]] = None
    
    def _should_log(self, trainer: pl.Trainer) -> bool:
        """Check if we should log this epoch."""
        return (trainer.current_epoch + 1) % self.log_every_n_epochs == 0
    
    def _get_tensorboard_logger(
        self, trainer: pl.Trainer
    ) -> Optional[TensorBoardLogger]:
        """Get TensorBoard logger from trainer."""
        if trainer.logger is None:
            return None
        
        if isinstance(trainer.logger, TensorBoardLogger):
            return trainer.logger
        
        # Check if it's a LoggerCollection
        if hasattr(trainer.logger, "experiment"):
            return trainer.logger
        
        return None
    
    def _normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Normalize volume to [0, 1] range."""
        v_min = volume.min()
        v_max = volume.max()
        if v_max - v_min > 1e-8:
            return (volume - v_min) / (v_max - v_min)
        return volume - v_min
    
    def _get_instance_colors(self, num_colors: int = 64) -> torch.Tensor:
        """Generate distinct colors for instance visualization."""
        # Use a combination of hue cycling for more colors
        colors = [[0.0, 0.0, 0.0]]  # Background is black
        
        for i in range(num_colors - 1):
            # Use golden ratio for hue distribution
            hue = (i * 0.618033988749895) % 1.0
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 4) * 0.05
            
            # HSV to RGB conversion
            c = value * saturation
            x = c * (1 - abs((hue * 6) % 2 - 1))
            m = value - c
            
            if hue < 1/6:
                r, g, b = c, x, 0
            elif hue < 2/6:
                r, g, b = x, c, 0
            elif hue < 3/6:
                r, g, b = 0, c, x
            elif hue < 4/6:
                r, g, b = 0, x, c
            elif hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append([r + m, g + m, b + m])
        
        return torch.tensor(colors, dtype=torch.float32)
    
    def _embedding_to_rgb(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert high-dimensional embedding to RGB using PCA projection.
        
        Args:
            embedding: Embedding tensor [E, H, W] where E is embedding dim.
        
        Returns:
            RGB image [3, H, W] with values in [0, 1].
        """
        E, H, W = embedding.shape
        device = embedding.device
        N = H * W
        
        # Flatten spatial dimensions: [E, H, W] -> [N, E] (transpose for PCA)
        emb_flat = rearrange(embedding, "e h w -> (h w) e")  # [N, E]
        
        # Center the data along pixel dimension
        emb_centered = emb_flat - emb_flat.mean(dim=0, keepdim=True)  # [N, E]
        
        # Simple PCA: get top 3 principal components
        if E >= 3:
            try:
                # Use SVD for PCA on [N, E] matrix
                # U: [N, min(N,E)], S: [min(N,E)], Vh: [min(N,E), E]
                U, S, Vh = torch.linalg.svd(emb_centered, full_matrices=False)
                # Project to first 3 components: take first 3 columns of U * S
                # projected = U[:, :3] * S[:3] gives [N, 3]
                projected = U[:, :3] * S[:3].unsqueeze(0)  # [N, 3]
            except Exception:
                # Fallback: just take first 3 channels
                projected = emb_flat[:, :3]  # [N, 3]
        else:
            # If embedding dim < 3, pad with zeros
            projected = torch.zeros(N, 3, device=device, dtype=embedding.dtype)
            projected[:, :E] = emb_flat
        
        # Normalize each channel to [0, 1]
        for i in range(3):
            channel = projected[:, i]
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min > 1e-8:
                projected[:, i] = (channel - c_min) / (c_max - c_min)
            else:
                projected[:, i] = 0.5
        
        # Reshape back: [N, 3] -> [3, H, W]
        rgb = rearrange(projected, "(h w) c -> c h w", h=H, w=W)
        
        return rgb.clamp(0, 1)
    
    def _apply_colormap(
        self,
        image: torch.Tensor,
        cmap: str = "gray",
        is_label: bool = False,
        is_instance: bool = False,
    ) -> torch.Tensor:
        """
        Apply colormap to single-channel image.
        
        Args:
            image: Single-channel image [H, W] in [0, 1].
            cmap: Colormap name.
            is_label: Whether this is a semantic label (use discrete colors).
            is_instance: Whether this is an instance label (use many colors).
        
        Returns:
            RGB image [3, H, W].
        """
        if is_instance:
            # Use many distinct colors for instances
            colors = self._get_instance_colors(64).to(image.device)
            labels = image.long() % len(colors)
            rgb = colors[labels]
            return rearrange(rgb, "h w c -> c h w")
        
        elif is_label:
            # Use discrete colors for semantic labels
            colors = torch.tensor([
                [0.0, 0.0, 0.0],      # 0: Black (background)
                [0.12, 0.47, 0.71],   # 1: Blue
                [1.0, 0.5, 0.05],     # 2: Orange
                [0.17, 0.63, 0.17],   # 3: Green
                [0.84, 0.15, 0.16],   # 4: Red
                [0.58, 0.40, 0.74],   # 5: Purple
                [0.55, 0.34, 0.29],   # 6: Brown
                [0.89, 0.47, 0.76],   # 7: Pink
                [0.5, 0.5, 0.5],      # 8: Gray
                [0.74, 0.74, 0.13],   # 9: Yellow
            ], device=image.device, dtype=torch.float32)
            
            # Clamp to valid range
            labels = image.long().clamp(0, len(colors) - 1)
            rgb = colors[labels]
            # [H, W, 3] -> [3, H, W]
            return rearrange(rgb, "h w c -> c h w")
        else:
            # Grayscale: repeat channel
            image = image.float()
            if self.normalize_images:
                image = self._normalize_volume(image)
            return rearrange(image, "h w -> 1 h w").repeat(3, 1, 1)
    
    def _extract_slices(
        self,
        volume: torch.Tensor,
        axis: str,
    ) -> List[torch.Tensor]:
        """
        Extract evenly spaced slices from a 3D volume.
        
        For EM connectomics data, tensors are [Z, Y, X] where:
        - Z = depth (slices through tissue, ~100 for SNEMI3D)
        - Y = height (1024 for SNEMI3D)
        - X = width (1024 for SNEMI3D)
        
        Axis naming for EM data:
        - "axial": Slice through Z, view Y-X plane - typical EM microscope view
        - "coronal": Slice through Y, view Z-X plane
        - "sagittal": Slice through X, view Z-Y plane
        
        Args:
            volume: 3D volume [Z, Y, X] or [C, Z, Y, X].
            axis: Slice axis ("axial", "coronal", "sagittal").
        
        Returns:
            List of 2D slices.
        """
        # Handle channel dimension
        if volume.dim() == 4:
            volume = volume[0]  # Take first channel
        
        Z, Y, X = volume.shape
        
        # Normalize axis names
        axis = axis.lower()
        if axis in ["axial", "xy"]:
            axis = "axial"
        elif axis in ["coronal", "xz"]:
            axis = "coronal"
        elif axis in ["sagittal", "yz"]:
            axis = "sagittal"
        
        # Get axis dimension
        if axis == "axial":
            dim_size = Z
        elif axis == "coronal":
            dim_size = Y
        elif axis == "sagittal":
            dim_size = X
        else:
            raise ValueError(f"Unknown axis: {axis}. Use 'axial', 'coronal', or 'sagittal'")
        
        # Find valid (non-empty) slice indices
        valid_indices = []
        for idx in range(dim_size):
            if axis == "axial":
                slice_data = volume[idx, :, :]
            elif axis == "coronal":
                slice_data = volume[:, idx, :]
            elif axis == "sagittal":
                slice_data = volume[:, :, idx]
            
            # Check if slice has content (not all zeros/padding)
            if slice_data.abs().sum() > 0:
                valid_indices.append(idx)
        
        # If no valid slices, fall back to middle slices
        if not valid_indices:
            valid_indices = list(range(dim_size // 4, 3 * dim_size // 4))
        
        # Sample evenly spaced indices from valid range
        if len(valid_indices) >= self.num_slices:
            step = len(valid_indices) // self.num_slices
            selected_indices = [valid_indices[i * step] for i in range(self.num_slices)]
        else:
            selected_indices = valid_indices[:self.num_slices]
        
        # Extract slices
        slices = []
        for idx in selected_indices:
            if axis == "axial":
                slices.append(volume[idx, :, :])  # [Y, X]
            elif axis == "coronal":
                slices.append(volume[:, idx, :])  # [Z, X]
            elif axis == "sagittal":
                slices.append(volume[:, :, idx])  # [Z, Y]
        
        return slices
    
    def _create_grid(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: Optional[torch.Tensor],
        axis: str,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create a grid of image/label/prediction slices.
        
        Layout: Input Image | Target Instance Labels | Predicted Instance Labels
        
        Args:
            images: Image volume [B, C, Z, Y, X].
            labels: Instance label volume [B, Z, Y, X] or [B, 1, Z, Y, X].
            predictions: Prediction logits [B, num_classes, Z, Y, X] (for foreground mask).
            axis: Slice axis ("axial", "coronal", "sagittal").
            embeddings: Instance embeddings [B, E, Z, Y, X] for clustering.
        
        Returns:
            Grid image [3, grid_H, grid_W].
        """
        from neurocircuitry.utils.labels import cluster_embeddings_meanshift
        
        batch_size = min(images.shape[0], self.max_samples)
        
        # Handle label dimensions
        if labels.dim() == 5:
            labels = rearrange(labels, "b 1 z y x -> b z y x")
        
        # Get foreground mask from semantic predictions
        if predictions is not None:
            fg_mask = predictions.argmax(dim=1)  # [B, Z, Y, X]
        else:
            fg_mask = None
        
        rows = []
        for b in range(batch_size):
            # Extract slices for image and labels
            img_slices = self._extract_slices(images[b], axis)
            lbl_slices = self._extract_slices(labels[b], axis)
            
            # Cluster embeddings to get predicted instances
            pred_instances = None
            if embeddings is not None and fg_mask is not None:
                try:
                    pred_instances = cluster_embeddings_meanshift(
                        embeddings[b],
                        foreground_mask=fg_mask[b],
                        bandwidth=self.clustering_bandwidth,
                        min_cluster_size=50,
                    )
                except Exception:
                    # Fallback: use foreground mask as single instance
                    pred_instances = fg_mask[b].long()
            
            if pred_instances is not None:
                pred_slices = self._extract_slices(pred_instances, axis)
            else:
                pred_slices = [None] * len(img_slices)
            
            # Create row for this sample
            for img, lbl, pred in zip(img_slices, lbl_slices, pred_slices):
                # Input image: grayscale
                img_rgb = self._apply_colormap(img, self.cmap_image, is_label=False)
                
                # Target instance labels: use instance colors (many distinct colors)
                lbl_rgb = self._apply_colormap(lbl, is_instance=True)
                
                if pred is not None:
                    # Predicted instance labels: clustered from embeddings
                    pred_rgb = self._apply_colormap(pred, is_instance=True)
                    row = torch.cat([img_rgb, lbl_rgb, pred_rgb], dim=2)  # Concat along W
                else:
                    row = torch.cat([img_rgb, lbl_rgb], dim=2)
                
                rows.append(row)
        
        # Stack rows vertically
        if rows:
            grid = torch.cat(rows, dim=1)  # Concat along H
            return grid
        
        # Return empty tensor if no rows
        return torch.zeros(3, 64, 64)
    
    def _create_embedding_grid(
        self,
        embeddings: torch.Tensor,
        axis: str,
    ) -> torch.Tensor:
        """
        Create a grid of embedding visualizations (PCA projected to RGB).
        
        Args:
            embeddings: Embedding volume [B, E, Z, Y, X].
            axis: Slice axis ("axial", "coronal", "sagittal").
        
        Returns:
            Grid image [3, grid_H, grid_W].
        """
        batch_size = min(embeddings.shape[0], self.max_samples)
        
        # Normalize axis names
        axis = axis.lower()
        if axis in ["axial", "xy"]:
            axis = "axial"
        elif axis in ["coronal", "xz"]:
            axis = "coronal"
        elif axis in ["sagittal", "yz"]:
            axis = "sagittal"
        
        # Use first sample only
        emb = embeddings[0]  # [E, Z, Y, X]
        E, Z, Y, X = emb.shape
        
        if axis == "axial":
            dim_size = Z
        elif axis == "coronal":
            dim_size = Y
        elif axis == "sagittal":
            dim_size = X
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        # Find valid slice indices (where embedding has content)
        valid_indices = []
        for idx in range(dim_size):
            if axis == "axial":
                slice_data = emb[:, idx, :, :]
            elif axis == "coronal":
                slice_data = emb[:, :, idx, :]
            elif axis == "sagittal":
                slice_data = emb[:, :, :, idx]
            
            if slice_data.abs().sum() > 0:
                valid_indices.append(idx)
        
        if not valid_indices:
            valid_indices = list(range(dim_size // 4, 3 * dim_size // 4))
        
        # Sample evenly spaced indices
        if len(valid_indices) >= self.num_slices:
            step = len(valid_indices) // self.num_slices
            selected_indices = [valid_indices[i * step] for i in range(self.num_slices)]
        else:
            selected_indices = valid_indices[:self.num_slices]
        
        slice_imgs = []
        for idx in selected_indices:
            if axis == "axial":
                emb_slice = emb[:, idx, :, :]  # [E, Y, X]
            elif axis == "coronal":
                emb_slice = emb[:, :, idx, :]  # [E, Z, X]
            elif axis == "sagittal":
                emb_slice = emb[:, :, :, idx]  # [E, Z, Y]
            
            rgb = self._embedding_to_rgb(emb_slice)
            slice_imgs.append(rgb)
        
        if slice_imgs:
            return torch.cat(slice_imgs, dim=2)  # Concatenate horizontally
        
        return torch.zeros(3, 64, 64)
    
    def _create_volume_grid(
        self,
        images: torch.Tensor,
        axis: str,
    ) -> torch.Tensor:
        """
        Create a grid of input volume slices (grayscale).
        
        Args:
            images: Image volume [B, C, Z, Y, X].
            axis: Slice axis ("axial", "coronal", "sagittal").
        
        Returns:
            Grid image [3, grid_H, grid_W].
        """
        # Only use first sample for cleaner visualization
        img_slices = self._extract_slices(images[0], axis)
        
        slice_imgs = []
        for img_slice in img_slices:
            # Convert grayscale to RGB
            img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            img_rgb = rearrange(img_norm, "h w -> 1 h w").repeat(3, 1, 1)
            slice_imgs.append(img_rgb)
        
        if slice_imgs:
            # Concatenate horizontally (single row)
            return torch.cat(slice_imgs, dim=2)
        
        return torch.zeros(3, 64, 64)
    
    def _create_label_grid(
        self,
        labels: torch.Tensor,
        axis: str,
        is_instance: bool = True,
    ) -> torch.Tensor:
        """
        Create a grid of label slices.
        
        Args:
            labels: Label volume [B, Z, Y, X] or [B, 1, Z, Y, X].
            axis: Slice axis ("axial", "coronal", "sagittal").
            is_instance: If True, use instance colormap with distinct colors.
        
        Returns:
            Grid image [3, grid_H, grid_W].
        """
        # Handle label dimensions
        if labels.dim() == 5:
            labels = rearrange(labels, "b 1 z y x -> b z y x")
        
        # Only use first sample for cleaner visualization
        lbl_slices = self._extract_slices(labels[0], axis)
        
        slice_imgs = []
        for lbl_slice in lbl_slices:
            lbl_rgb = self._apply_colormap(lbl_slice, is_instance=is_instance)
            slice_imgs.append(lbl_rgb)
        
        if slice_imgs:
            # Concatenate horizontally (single row)
            return torch.cat(slice_imgs, dim=2)
        
        return torch.zeros(3, 64, 64)
    
    def _create_predicted_instance_grid(
        self,
        embeddings: torch.Tensor,
        predictions: torch.Tensor,
        axis: str,
    ) -> torch.Tensor:
        """
        Create a grid of predicted instance labels from clustering embeddings.
        
        Args:
            embeddings: Embedding volume [B, E, Z, Y, X].
            predictions: Semantic predictions [B, num_classes, Z, Y, X] for foreground mask.
            axis: Slice axis ("axial", "coronal", "sagittal").
        
        Returns:
            Grid image [3, grid_H, grid_W].
        """
        from neurocircuitry.utils.labels import cluster_embeddings_meanshift
        
        # Get foreground mask from semantic predictions (first sample only)
        fg_mask = predictions.argmax(dim=1)  # [B, Z, Y, X]
        
        emb = embeddings[0]  # [E, Z, Y, X]
        fg = fg_mask[0]  # [Z, Y, X]
        
        # Cluster embeddings to get predicted instances
        try:
            pred_instances = cluster_embeddings_meanshift(
                emb,
                foreground_mask=fg,
                bandwidth=self.clustering_bandwidth,
                min_cluster_size=50,
            )
        except Exception:
            # Fallback: use foreground mask as single instance
            pred_instances = fg.long()
        
        pred_slices = self._extract_slices(pred_instances, axis)
        
        slice_imgs = []
        for pred_slice in pred_slices:
            pred_rgb = self._apply_colormap(pred_slice, is_instance=True)
            slice_imgs.append(pred_rgb)
        
        if slice_imgs:
            # Concatenate horizontally (single row)
            return torch.cat(slice_imgs, dim=2)
        
        return torch.zeros(3, 64, 64)
    
    def _create_instance_grid(
        self,
        embeddings: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        axis: str,
    ) -> torch.Tensor:
        """
        Create a grid showing ground truth instances vs predicted instances.
        
        Args:
            embeddings: Embedding volume [B, E, Z, Y, X].
            predictions: Semantic predictions [B, num_classes, Z, Y, X].
            labels: Ground truth instance labels [B, Z, Y, X].
            axis: Slice axis ("axial", "coronal", "sagittal").
        
        Returns:
            Grid image [3, grid_H, grid_W] with GT instances | Predicted instances.
        """
        from neurocircuitry.utils.labels import cluster_embeddings_meanshift
        
        batch_size = min(embeddings.shape[0], self.max_samples, 1)  # Only first sample
        
        # Handle label dimensions
        if labels.dim() == 5:
            labels = rearrange(labels, "b 1 z y x -> b z y x")
        
        # Get foreground mask from semantic predictions
        fg_mask = predictions.argmax(dim=1)  # [B, Z, Y, X]
        
        rows = []
        for b in range(batch_size):
            emb = embeddings[b]  # [E, D, H, W]
            lbl = labels[b]  # [D, H, W]
            fg = fg_mask[b]  # [D, H, W]
            
            # Cluster embeddings to get predicted instances
            try:
                pred_instances = cluster_embeddings_meanshift(
                    emb,
                    foreground_mask=fg,
                    bandwidth=self.clustering_bandwidth,
                    min_cluster_size=50,
                )
            except Exception:
                # Fallback: use foreground mask as single instance
                pred_instances = fg.long()
            
            # Extract slices
            gt_slices = self._extract_slices(lbl, axis)
            pred_inst_slices = self._extract_slices(pred_instances, axis)
            
            for gt_slice, pred_slice in zip(gt_slices, pred_inst_slices):
                # Apply instance colormap
                gt_rgb = self._apply_colormap(gt_slice, is_instance=True)
                pred_rgb = self._apply_colormap(pred_slice, is_instance=True)
                
                # Concatenate: GT | Predicted
                row = torch.cat([gt_rgb, pred_rgb], dim=2)
                rows.append(row)
        
        if rows:
            grid = torch.cat(rows, dim=1)
            return grid
        
        return torch.zeros(3, 64, 64)
    
    def _log_volumes(
        self,
        trainer: pl.Trainer,
        batch: Dict[str, torch.Tensor],
        outputs: Optional[Dict[str, torch.Tensor]],
        prefix: str,
    ) -> None:
        """Log volume visualizations to TensorBoard."""
        logger = self._get_tensorboard_logger(trainer)
        if logger is None:
            return
        
        images = batch.get("image")
        # Instance segmentation challenge format:
        # - 'label': indexed instance segmentation (IDs: 1, 2, 3, ...)
        # - 'class_ids': semantic class mapping (background=0, foreground=1)
        instance_labels = batch.get("label")
        
        if images is None or instance_labels is None:
            return
        
        # Ensure images are 5D: [B, C, Z, Y, X]
        if images.dim() == 4:
            images = rearrange(images, "b z y x -> b 1 z y x")
        
        # Get predictions and embeddings from outputs
        if isinstance(outputs, dict):
            predictions = outputs.get("logits")
            embeddings = outputs.get("embeds")
        else:
            predictions = outputs
            embeddings = None
        
        step = trainer.global_step
        
        # Map axis names to plane notation for clearer logging
        # For EM data with [Z, Y, X]:
        # - axial: slicing through Z, viewing Y-X plane (typical EM view)
        # - coronal: slicing through Y, viewing Z-X plane
        # - sagittal: slicing through X, viewing Z-Y plane
        axis_to_plane = {
            "axial": "YX",
            "coronal": "ZX",
            "sagittal": "ZY",
        }
        
        # Log slices for each axis
        for axis in self.slice_axes:
            # Normalize axis name
            axis_normalized = axis.lower()
            if axis_normalized in ["axial", "xy"]:
                axis_normalized = "axial"
            elif axis_normalized in ["sagittal", "yz"]:
                axis_normalized = "sagittal"
            elif axis_normalized in ["coronal", "xz"]:
                axis_normalized = "coronal"
            
            plane_name = axis_to_plane.get(axis_normalized, axis)
            
            # 1. Log input volume (grayscale EM image)
            try:
                input_grid = self._create_volume_grid(
                    images.detach().cpu(),
                    axis_normalized,
                )
                tag = f"{prefix}/1_input_{plane_name}"
                if hasattr(logger, "experiment"):
                    logger.experiment.add_image(tag, input_grid, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to log {prefix}/1_input_{plane_name}: {e}")
            
            # 2. Log ground truth instance labels
            try:
                gt_grid = self._create_label_grid(
                    instance_labels.detach().cpu(),
                    axis_normalized,
                    is_instance=True,
                )
                tag = f"{prefix}/2_gt_instance_{plane_name}"
                if hasattr(logger, "experiment"):
                    logger.experiment.add_image(tag, gt_grid, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to log {prefix}/2_gt_instance_{plane_name}: {e}")
            
            # 3. Log predicted instance labels (from clustering embeddings)
            if self.log_instances and embeddings is not None and predictions is not None:
                try:
                    pred_grid = self._create_predicted_instance_grid(
                        embeddings.detach().cpu(),
                        predictions.detach().cpu(),
                        axis_normalized,
                    )
                    tag = f"{prefix}/3_pred_instance_{plane_name}"
                    if hasattr(logger, "experiment"):
                        logger.experiment.add_image(tag, pred_grid, global_step=step)
                except Exception as e:
                    print(f"Warning: Failed to log {prefix}/3_pred_instance_{plane_name}: {e}")
            
            # 4. Log embedding visualization (PCA projection to RGB)
            if self.log_embeddings and embeddings is not None:
                try:
                    emb_grid = self._create_embedding_grid(
                        embeddings.detach().cpu(),
                        axis_normalized,
                    )
                    tag = f"{prefix}/4_embedding_{plane_name}"
                    if hasattr(logger, "experiment"):
                        logger.experiment.add_image(tag, emb_grid, global_step=step)
                except Exception as e:
                    print(f"Warning: Failed to log {prefix}/4_embedding_{plane_name}: {e}")
    
    # =========================================================================
    # Training hooks
    # =========================================================================
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Store training batch for end-of-epoch logging."""
        if not self.log_train:
            return
        
        # Only store first batch of epoch
        if batch_idx == 0:
            self._train_batch = batch
            
            # Get predictions and embeddings
            with torch.no_grad():
                images = batch.get("image")
                if images is not None:
                    if images.dim() == 4:
                        images = rearrange(images, "b d h w -> b 1 d h w")
                    model_outputs = pl_module(images)
                    # Store full outputs dict (includes logits and embedding)
                    self._train_outputs = model_outputs
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log training volumes at end of epoch."""
        if not self.log_train or not self._should_log(trainer):
            return
        
        if self._train_batch is not None:
            self._log_volumes(
                trainer,
                self._train_batch,
                self._train_outputs,
                prefix="train",
            )
        
        # Clear stored data
        self._train_batch = None
        self._train_outputs = None
    
    # =========================================================================
    # Validation hooks
    # =========================================================================
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Store validation batch for end-of-epoch logging."""
        if not self.log_val:
            return
        
        # Only store first batch
        if batch_idx == 0:
            self._val_batch = batch
            
            # Get predictions and embeddings
            with torch.no_grad():
                images = batch.get("image")
                if images is not None:
                    if images.dim() == 4:
                        images = rearrange(images, "b d h w -> b 1 d h w")
                    model_outputs = pl_module(images)
                    # Store full outputs dict (includes logits and embedding)
                    self._val_outputs = model_outputs
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log validation volumes at end of epoch."""
        if not self.log_val or not self._should_log(trainer):
            return
        
        if self._val_batch is not None:
            self._log_volumes(
                trainer,
                self._val_batch,
                self._val_outputs,
                prefix="val",
            )
        
        # Clear stored data
        self._val_batch = None
        self._val_outputs = None

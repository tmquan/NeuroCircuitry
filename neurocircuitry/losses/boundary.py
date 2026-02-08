"""
Boundary-aware loss functions for connectomics segmentation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for improving edge predictions.
    
    Computes additional loss weight on boundary pixels to encourage
    accurate boundary delineation.
    
    Args:
        boundary_weight: Extra weight for boundary pixels (default: 5.0).
        kernel_size: Size of morphological kernel for boundary detection (default: 3).
    
    Example:
        >>> loss_fn = BoundaryLoss(boundary_weight=5.0)
        >>> logits = torch.randn(4, 2, 256, 256)
        >>> labels = torch.randint(0, 2, (4, 256, 256))
        >>> loss = loss_fn(logits, labels)
    """
    
    def __init__(
        self,
        boundary_weight: float = 5.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        
        # Create erosion kernel
        self.register_buffer(
            "kernel",
            torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2),
        )
    
    def _compute_boundaries(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary mask from instance labels.
        
        Args:
            labels: Instance labels [B, H, W].
        
        Returns:
            Boundary mask [B, H, W] with 1 at boundaries.
        """
        # Add channel dimension for conv
        labels_float = labels.float().unsqueeze(1)
        
        # Erosion: min pooling
        eroded = -F.max_pool2d(
            -labels_float,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        
        # Dilation: max pooling
        dilated = F.max_pool2d(
            labels_float,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        
        # Boundary = dilated - eroded
        boundary = (dilated != eroded).float().squeeze(1)
        
        return boundary
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary-weighted cross entropy loss.
        
        Args:
            logits: Prediction logits [B, C, H, W].
            labels: Ground truth labels [B, H, W].
        
        Returns:
            Boundary-weighted loss scalar.
        """
        # Compute per-pixel cross entropy
        ce_loss = F.cross_entropy(logits, labels.long(), reduction="none")
        
        # Compute boundary mask
        boundaries = self._compute_boundaries(labels)
        
        # Apply boundary weighting
        weights = 1.0 + (self.boundary_weight - 1.0) * boundaries
        
        weighted_loss = (ce_loss * weights).mean()
        
        return weighted_loss


class BoundaryAwareCrossEntropy(nn.Module):
    """
    Cross entropy loss with distance-based boundary weighting.
    
    Pixels closer to instance boundaries receive higher weights,
    encouraging the model to focus on accurate boundary predictions.
    
    Args:
        base_weight: Base weight for all pixels (default: 1.0).
        boundary_weight: Maximum extra weight at boundaries (default: 5.0).
        sigma: Gaussian decay for distance weighting (default: 2.0).
        num_classes: Number of classes (default: 2).
    
    Example:
        >>> loss_fn = BoundaryAwareCrossEntropy(boundary_weight=5.0)
        >>> logits = torch.randn(4, 2, 256, 256)
        >>> labels = torch.randint(0, 2, (4, 256, 256))
        >>> loss = loss_fn(logits, labels)
    """
    
    def __init__(
        self,
        base_weight: float = 1.0,
        boundary_weight: float = 5.0,
        sigma: float = 2.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.boundary_weight = boundary_weight
        self.sigma = sigma
        self.num_classes = num_classes
    
    def _compute_distance_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute distance-based weights from boundaries.
        
        Uses Sobel-like gradient detection as approximation.
        
        Args:
            labels: Instance labels [B, H, W].
        
        Returns:
            Weight map [B, H, W].
        """
        # Convert to float
        labels_float = labels.float().unsqueeze(1)
        
        # Sobel gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=labels.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=labels.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        grad_x = F.conv2d(labels_float, sobel_x, padding=1)
        grad_y = F.conv2d(labels_float, sobel_y, padding=1)
        
        # Gradient magnitude (boundary strength)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad_mag = grad_mag.squeeze(1)
        
        # Normalize and apply Gaussian weighting
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        
        # Weight = base + (boundary - base) * gradient
        weights = self.base_weight + (self.boundary_weight - self.base_weight) * grad_mag
        
        return weights
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary-aware cross entropy loss.
        
        Args:
            logits: Prediction logits [B, C, H, W].
            labels: Ground truth labels [B, H, W].
        
        Returns:
            Weighted loss scalar.
        """
        # Compute per-pixel cross entropy
        ce_loss = F.cross_entropy(logits, labels.long(), reduction="none")
        
        # Compute distance-based weights
        weights = self._compute_distance_weights(labels)
        
        # Apply weighting
        weighted_loss = (ce_loss * weights).mean()
        
        return weighted_loss

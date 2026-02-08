"""
SegResNet wrapper with customizable heads for connectomics tasks.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from neurocircuitry.models.base import BaseModel


class SegResNetWrapper(BaseModel):
    """
    MONAI SegResNet with customizable output heads for connectomics.
    
    Provides a flexible encoder-decoder architecture with support for:
    - Semantic segmentation (single output head)
    - Instance segmentation (semantic + embedding heads)
    - Multi-task learning (multiple output heads)
    
    Args:
        in_channels: Number of input channels (default: 1).
        out_channels: Number of semantic classes (default: 2).
        spatial_dims: Spatial dimensions, 2 or 3 (default: 2).
        init_filters: Initial convolution filters (default: 32).
        feature_dim: Feature dimension from backbone (default: 64).
        emb_dim: Instance embedding dimension (default: 16).
        dropout: Dropout probability (default: 0.2).
        blocks_down: Encoder block depths (default: (1, 2, 2, 4)).
        blocks_up: Decoder block depths (default: (1, 1, 1)).
        use_instance_head: Include instance embedding head (default: False).
        use_boundary_head: Include boundary prediction head (default: False).
    
    Example:
        >>> # Semantic segmentation only
        >>> model = SegResNetWrapper(
        ...     in_channels=1,
        ...     out_channels=2,
        ...     spatial_dims=2
        ... )
        >>> x = torch.randn(4, 1, 256, 256)
        >>> output = model(x)
        >>> print(output["logits"].shape)  # [4, 2, 256, 256]
        >>> 
        >>> # With instance embedding head
        >>> model = SegResNetWrapper(
        ...     in_channels=1,
        ...     out_channels=2,
        ...     use_instance_head=True,
        ...     emb_dim=16
        ... )
        >>> output = model(x)
        >>> print(output["embedding"].shape)  # [4, 16, 256, 256]
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        spatial_dims: int = 2,
        init_filters: int = 32,
        feature_dim: int = 64,
        emb_dim: int = 16,
        dropout: float = 0.2,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        use_instance_head: bool = False,
        use_boundary_head: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
        )
        
        self.init_filters = init_filters
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.use_instance_head = use_instance_head
        self.use_boundary_head = use_boundary_head
        
        # Import MONAI SegResNet
        from monai.networks.nets import SegResNet
        
        # Backbone encoder-decoder
        self.backbone = SegResNet(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=feature_dim,
            dropout_prob=dropout,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
        )
        
        # Semantic segmentation head
        conv_class = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        bn_class = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
        
        self.semantic_head = nn.Sequential(
            conv_class(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            bn_class(feature_dim // 2),
            nn.ReLU(inplace=True),
            conv_class(feature_dim // 2, out_channels, kernel_size=1),
        )
        
        # Optional instance embedding head
        if use_instance_head:
            self.instance_head = nn.Sequential(
                conv_class(feature_dim, feature_dim, kernel_size=3, padding=1),
                bn_class(feature_dim),
                nn.ReLU(inplace=True),
                conv_class(feature_dim, emb_dim, kernel_size=1),
            )
        
        # Optional boundary prediction head
        if use_boundary_head:
            self.boundary_head = nn.Sequential(
                conv_class(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
                bn_class(feature_dim // 2),
                nn.ReLU(inplace=True),
                conv_class(feature_dim // 2, 1, kernel_size=1),
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning all output heads.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W].
        
        Returns:
            Dictionary with:
                - 'features': Backbone features
                - 'logits': Semantic segmentation logits
                - 'embedding': Instance embeddings (if use_instance_head)
                - 'boundary': Boundary predictions (if use_boundary_head)
        """
        # Backbone features
        features = self.backbone(x)
        
        outputs = {
            "features": features,
            "logits": self.semantic_head(features),
        }
        
        if self.use_instance_head:
            outputs["embedding"] = self.instance_head(features)
        
        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(features)
        
        return outputs
    
    def get_output_channels(self) -> int:
        return self.out_channels
    
    def freeze_encoder(self) -> None:
        """Freeze backbone encoder layers."""
        # SegResNet doesn't have separate encoder attribute,
        # freeze early layers instead
        for name, param in self.backbone.named_parameters():
            if "down" in name or "conv_initial" in name:
                param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze backbone encoder layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True

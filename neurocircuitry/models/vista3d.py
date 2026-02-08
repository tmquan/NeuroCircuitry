"""
Vista3D model wrapper for connectomics segmentation.

Vista3D is NVIDIA's 3D foundation model for medical image segmentation,
available in MONAI. This wrapper adapts it for connectomics applications.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from neurocircuitry.models.base import BaseModel


class Vista3DWrapper(BaseModel):
    """
    Wrapper for MONAI Vista3D foundation model adapted for connectomics.
    
    Vista3D is a 3D segmentation foundation model that supports:
    - Interactive segmentation with point/box prompts
    - Automatic segmentation for batch processing
    - Zero-shot capability on unseen structures
    - Fine-tuning for domain-specific tasks
    
    This wrapper adapts Vista3D for connectomics applications including:
    - Neuron instance segmentation
    - Synapse detection
    - Organelle segmentation
    
    Args:
        in_channels: Number of input channels (default: 1 for EM).
        num_classes: Number of output classes (default: 2 for bg/neuron).
        pretrained: Load pretrained weights (default: True).
        freeze_encoder: Freeze encoder for fine-tuning (default: False).
        encoder_name: Encoder architecture ('segresnet' or 'swin').
        feature_size: Base feature size for encoder.
        use_point_prompts: Enable interactive point prompts.
        use_automatic_mode: Enable automatic segmentation mode.
    
    Note:
        Vista3D requires MONAI >= 1.3.0 with vista3d extras.
        Install with: pip install 'monai[vista3d]'
    
    Example:
        >>> model = Vista3DWrapper(
        ...     in_channels=1,
        ...     num_classes=2,
        ...     pretrained=True,
        ...     freeze_encoder=True
        ... )
        >>> 
        >>> # Automatic segmentation
        >>> x = torch.randn(1, 1, 64, 128, 128)
        >>> output = model(x)
        >>> logits = output["logits"]  # [1, 2, 64, 128, 128]
        >>> 
        >>> # Interactive segmentation with point prompts
        >>> points = torch.tensor([[[32, 64, 64]]])  # [B, N, 3]
        >>> labels = torch.tensor([[1]])  # [B, N] - foreground points
        >>> output = model(x, point_coords=points, point_labels=labels)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        encoder_name: str = "segresnet",
        feature_size: int = 48,
        use_point_prompts: bool = False,
        use_automatic_mode: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=num_classes,
            spatial_dims=3,
        )
        
        self.pretrained = pretrained
        self.freeze_encoder_flag = freeze_encoder
        self.encoder_name = encoder_name
        self.feature_size = feature_size
        self.use_point_prompts = use_point_prompts
        self.use_automatic_mode = use_automatic_mode
        
        # Initialize Vista3D model
        self._build_model(**kwargs)
        
        if freeze_encoder:
            self.freeze_encoder()
    
    def _build_model(self, **kwargs) -> None:
        """
        Build the Vista3D model architecture.
        
        Falls back to SegResNet if Vista3D is not available.
        """
        try:
            # Try to import Vista3D from MONAI
            from monai.networks.nets import vista3d
            
            self.vista3d = vista3d.Vista3D(
                in_channels=self.in_channels,
                encoder_name=self.encoder_name,
                feature_size=self.feature_size,
                **kwargs,
            )
            
            # Add output head for connectomics
            self.output_head = nn.Conv3d(
                self.feature_size,
                self.out_channels,
                kernel_size=1,
            )
            
            self._has_vista3d = True
            
        except (ImportError, AttributeError):
            # Fallback to SegResNet-based architecture
            import warnings
            warnings.warn(
                "Vista3D not available in this MONAI version. "
                "Using SegResNet backbone instead. "
                "Install MONAI >= 1.3.0 with vista3d extras for full support."
            )
            
            from monai.networks.nets import SegResNet
            
            self.backbone = SegResNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.feature_size,
                init_filters=self.feature_size,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
            )
            
            self.output_head = nn.Conv3d(
                self.feature_size,
                self.out_channels,
                kernel_size=1,
            )
            
            self._has_vista3d = False
    
    def forward(
        self,
        x: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional interactive prompts.
        
        Args:
            x: Input tensor [B, C, D, H, W].
            point_coords: Point prompt coordinates [B, N, 3] (z, y, x).
            point_labels: Point labels [B, N] (0=background, 1=foreground).
            boxes: Box prompts [B, M, 6] (z1, y1, x1, z2, y2, x2).
            class_ids: Class IDs for automatic mode [B] or [B, K].
        
        Returns:
            Dictionary with:
                - 'logits': Segmentation logits [B, num_classes, D, H, W]
                - 'features': Intermediate features (if available)
                - 'embeddings': Point/patch embeddings (if using prompts)
        """
        outputs = {}
        
        if self._has_vista3d:
            # Use Vista3D forward pass
            if point_coords is not None or boxes is not None:
                # Interactive mode with prompts
                vista_out = self.vista3d(
                    x,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    boxes=boxes,
                )
            else:
                # Automatic mode
                vista_out = self.vista3d(x, class_ids=class_ids)
            
            # Extract features and apply output head
            if isinstance(vista_out, dict):
                features = vista_out.get("features", vista_out.get("logits"))
                outputs["embeddings"] = vista_out.get("embeddings")
            else:
                features = vista_out
            
            outputs["features"] = features
            outputs["logits"] = self.output_head(features)
            
        else:
            # SegResNet fallback
            features = self.backbone(x)
            outputs["features"] = features
            outputs["logits"] = self.output_head(features)
        
        return outputs
    
    def get_output_channels(self) -> int:
        return self.out_channels
    
    def freeze_encoder(self) -> None:
        """Freeze encoder/backbone parameters for fine-tuning."""
        if self._has_vista3d:
            for param in self.vista3d.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.freeze_encoder_flag = True
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder/backbone parameters."""
        if self._has_vista3d:
            for param in self.vista3d.encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        self.freeze_encoder_flag = False
    
    def load_pretrained_weights(self, weights_path: str) -> None:
        """
        Load pretrained weights from file.
        
        Args:
            weights_path: Path to weights file (.pt or .pth).
        """
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Handle different state dict formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Load with strict=False to handle architecture differences
        self.load_state_dict(state_dict, strict=False)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "vista3d_segresnet",
        **kwargs,
    ) -> "Vista3DWrapper":
        """
        Load a pretrained Vista3D model.
        
        Args:
            model_name: Name of pretrained model configuration.
            **kwargs: Additional arguments passed to constructor.
        
        Returns:
            Initialized Vista3DWrapper with pretrained weights.
        """
        # Predefined configurations
        configs = {
            "vista3d_segresnet": {
                "encoder_name": "segresnet",
                "feature_size": 48,
            },
            "vista3d_swin": {
                "encoder_name": "swin",
                "feature_size": 48,
            },
        }
        
        if model_name in configs:
            config = configs[model_name]
            config.update(kwargs)
            return cls(pretrained=True, **config)
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available: {list(configs.keys())}"
            )

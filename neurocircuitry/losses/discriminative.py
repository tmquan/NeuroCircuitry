"""
Discriminative Loss for Instance Segmentation.

Based on "Semantic Instance Segmentation with a Discriminative Loss Function"
by De Brabandere et al. (2017).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for learning pixel embeddings for instance segmentation.
    
    The loss encourages:
    - L_var: Embeddings of same instance to be close (within delta_var)
    - L_dist: Embeddings of different instances to be far apart (beyond delta_dist)
    - L_reg: Embeddings to stay close to origin (regularization)
    
    Supports both 2D and 3D inputs.
    
    Args:
        delta_var: Margin for variance term (default: 0.5).
        delta_dist: Margin for distance term (default: 1.5).
        norm: Norm type for distance computation (default: 2).
        alpha: Weight for variance term (default: 1.0).
        beta: Weight for distance term (default: 1.0).
        gamma: Weight for regularization term (default: 0.001).
    
    Example:
        >>> loss_fn = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5)
        >>> embedding = torch.randn(4, 16, 256, 256)  # [B, E, H, W]
        >>> labels = torch.randint(0, 10, (4, 256, 256))  # [B, H, W]
        >>> 
        >>> total_loss, L_var, L_dist, L_reg = loss_fn(embedding, labels)
    """
    
    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        norm: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
    ):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def _flatten_spatial(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Flatten spatial dimensions for both 2D and 3D inputs.
        
        Args:
            embedding: [B, E, H, W] for 2D or [B, E, D, H, W] for 3D.
            instance_mask: [B, 1, H, W]/[B, H, W] for 2D or [B, 1, D, H, W]/[B, D, H, W] for 3D.
        
        Returns:
            Tuple of (emb_flat [B, E, N], mask_flat [B, N], is_3d).
        """
        is_3d = embedding.dim() == 5
        
        if is_3d:
            emb_flat = rearrange(embedding, "b e d h w -> b e (d h w)")
            if instance_mask.dim() == 5:
                mask_flat = rearrange(instance_mask, "b 1 d h w -> b (d h w)")
            else:
                mask_flat = rearrange(instance_mask, "b d h w -> b (d h w)")
        else:
            emb_flat = rearrange(embedding, "b e h w -> b e (h w)")
            if instance_mask.dim() == 4:
                mask_flat = rearrange(instance_mask, "b 1 h w -> b (h w)")
            else:
                mask_flat = rearrange(instance_mask, "b h w -> b (h w)")
        
        return emb_flat, mask_flat, is_3d
    
    def _compute_cluster_means(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cluster centers for each instance."""
        # Ensure float32
        emb = emb.float()
        
        centers = []
        
        for inst_id in unique_instances:
            inst_mask = inst == inst_id
            if inst_mask.sum() == 0:
                continue
            
            inst_embeddings = emb[:, inst_mask]
            center = inst_embeddings.mean(dim=1)
            centers.append(center)
        
        if len(centers) == 0:
            return torch.zeros((0, emb.shape[0]), device=emb.device, dtype=torch.float32)
        
        return torch.stack(centers)
    
    def _variance_loss(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Compute variance loss: pull embeddings toward their cluster center."""
        num_instances = len(unique_instances)
        
        if num_instances == 0:
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)
        
        # Ensure float32
        emb = emb.float()
        cluster_centers = cluster_centers.float()
        
        loss_var = torch.tensor(0.0, device=emb.device, dtype=torch.float32)
        
        for idx, inst_id in enumerate(unique_instances):
            inst_mask = inst == inst_id
            if inst_mask.sum() == 0:
                continue
            
            inst_embeddings = emb[:, inst_mask]
            center = cluster_centers[idx]
            center_broadcast = rearrange(center, "e -> e 1")
            
            distances = torch.norm(inst_embeddings - center_broadcast, p=self.norm, dim=0)
            hinged = F.relu(distances - self.delta_var) ** 2
            loss_var = loss_var + hinged.mean()
        
        return loss_var / num_instances
    
    def _distance_loss(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """Compute distance loss: push different instance centers apart."""
        num_instances = cluster_centers.shape[0]
        
        if num_instances <= 1:
            return torch.tensor(0.0, device=cluster_centers.device, dtype=torch.float32)
        
        # Ensure float32
        cluster_centers = cluster_centers.float()
        
        loss_dist = torch.tensor(0.0, device=cluster_centers.device, dtype=torch.float32)
        n_pairs = 0
        
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                dist = torch.norm(cluster_centers[i] - cluster_centers[j], p=self.norm)
                hinged = F.relu(2 * self.delta_dist - dist) ** 2
                loss_dist = loss_dist + hinged
                n_pairs += 1
        
        if n_pairs > 0:
            loss_dist = loss_dist / n_pairs
        
        return loss_dist
    
    def _regularization_loss(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss: keep centers near origin."""
        if cluster_centers.shape[0] == 0:
            return torch.tensor(0.0, device=cluster_centers.device, dtype=torch.float32)
        
        # Ensure float32
        cluster_centers = cluster_centers.float()
        
        norms = torch.norm(cluster_centers, p=self.norm, dim=1)
        return norms.mean()
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute discriminative loss.
        
        Args:
            embedding: Pixel embeddings [B, E, H, W] (2D) or [B, E, D, H, W] (3D).
            instance_mask: Instance labels [B, 1, H, W]/[B, H, W] (2D) or 
                          [B, 1, D, H, W]/[B, D, H, W] (3D).
                          Value 0 = background, >0 = instance IDs.
        
        Returns:
            Tuple of (total_loss, L_var, L_dist, L_reg).
        """
        batch_size = embedding.shape[0]
        
        emb_flat, mask_flat, _ = self._flatten_spatial(embedding, instance_mask)
        
        # Ensure float32 for loss accumulation
        loss_var_total = torch.tensor(0.0, device=embedding.device, dtype=torch.float32)
        loss_dist_total = torch.tensor(0.0, device=embedding.device, dtype=torch.float32)
        loss_reg_total = torch.tensor(0.0, device=embedding.device, dtype=torch.float32)
        
        valid_batches = 0
        
        for b in range(batch_size):
            emb = emb_flat[b]
            inst = mask_flat[b]
            
            unique_instances = torch.unique(inst)
            unique_instances = unique_instances[unique_instances > 0]
            
            if len(unique_instances) == 0:
                continue
            
            valid_batches += 1
            
            cluster_centers = self._compute_cluster_means(emb, inst, unique_instances)
            
            loss_var = self._variance_loss(emb, inst, unique_instances, cluster_centers)
            loss_dist = self._distance_loss(cluster_centers)
            loss_reg = self._regularization_loss(cluster_centers)
            
            loss_var_total = loss_var_total + loss_var
            loss_dist_total = loss_dist_total + loss_dist
            loss_reg_total = loss_reg_total + loss_reg
        
        if valid_batches > 0:
            loss_var_total = loss_var_total / valid_batches
            loss_dist_total = loss_dist_total / valid_batches
            loss_reg_total = loss_reg_total / valid_batches
        
        total_loss = (
            self.alpha * loss_var_total +
            self.beta * loss_dist_total +
            self.gamma * loss_reg_total
        )
        
        return total_loss, loss_var_total, loss_dist_total, loss_reg_total
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"delta_var={self.delta_var}, "
            f"delta_dist={self.delta_dist}, "
            f"norm={self.norm}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"gamma={self.gamma})"
        )


class DiscriminativeLossVectorized(DiscriminativeLoss):
    """
    Vectorized implementation using einops for better GPU efficiency.
    
    Uses scatter operations instead of explicit Python loops where possible.
    """
    
    def _compute_cluster_means(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cluster means using scatter operations."""
        num_instances = len(unique_instances)
        emb_dim = emb.shape[0]
        
        # Ensure emb is a regular tensor with consistent dtype
        emb = emb.float()
        
        max_inst = int(inst.max().item()) + 1
        inst_to_idx = torch.full((max_inst,), -1, device=emb.device, dtype=torch.long)
        
        for idx, inst_id in enumerate(unique_instances):
            inst_to_idx[inst_id.long()] = idx
        
        cluster_indices = inst_to_idx[inst.long()]
        valid_mask = cluster_indices >= 0
        
        if not valid_mask.any():
            return torch.zeros((num_instances, emb_dim), device=emb.device, dtype=emb.dtype)
        
        valid_emb = emb[:, valid_mask]
        valid_idx = cluster_indices[valid_mask]
        
        # Create cluster_sums with same dtype as emb
        cluster_sums = torch.zeros((num_instances, emb_dim), device=emb.device, dtype=emb.dtype)
        cluster_counts = torch.zeros(num_instances, device=emb.device, dtype=emb.dtype)
        
        valid_emb_t = rearrange(valid_emb, "e n -> n e")
        for e in range(emb_dim):
            cluster_sums[:, e].scatter_add_(0, valid_idx, valid_emb_t[:, e])
        cluster_counts.scatter_add_(0, valid_idx, torch.ones(valid_idx.shape[0], device=emb.device, dtype=emb.dtype))
        
        cluster_counts = torch.clamp(cluster_counts, min=1)
        cluster_means = cluster_sums / rearrange(cluster_counts, "c -> c 1")
        
        return cluster_means
    
    def _variance_loss(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_means: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized variance loss using scatter operations."""
        num_instances = len(unique_instances)
        
        if num_instances == 0:
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)
        
        # Ensure consistent dtype
        emb = emb.float()
        cluster_means = cluster_means.float()
        
        max_inst = int(inst.max().item()) + 1
        inst_to_idx = torch.full((max_inst,), -1, device=emb.device, dtype=torch.long)
        
        for idx, inst_id in enumerate(unique_instances):
            inst_to_idx[inst_id.long()] = idx
        
        cluster_indices = inst_to_idx[inst.long()]
        valid_mask = cluster_indices >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)
        
        valid_emb = emb[:, valid_mask]
        valid_idx = cluster_indices[valid_mask]
        
        gathered_means = cluster_means[valid_idx]
        gathered_means = rearrange(gathered_means, "n e -> e n")
        
        diff = valid_emb - gathered_means
        distances = torch.norm(diff, p=self.norm, dim=0)
        
        hinged = F.relu(distances - self.delta_var) ** 2
        
        # Create tensors with consistent dtype
        cluster_losses = torch.zeros(num_instances, device=emb.device, dtype=torch.float32)
        cluster_counts = torch.zeros(num_instances, device=emb.device, dtype=torch.float32)
        
        cluster_losses.scatter_add_(0, valid_idx, hinged.float())
        cluster_counts.scatter_add_(0, valid_idx, torch.ones_like(hinged, dtype=torch.float32))
        
        cluster_counts = torch.clamp(cluster_counts, min=1)
        per_cluster_loss = cluster_losses / cluster_counts
        var_loss = reduce(per_cluster_loss, "c -> ", "mean")
        
        return var_loss
    
    def _distance_loss(self, cluster_means: torch.Tensor) -> torch.Tensor:
        """Vectorized distance loss between cluster centers."""
        num_instances = cluster_means.shape[0]
        
        if num_instances <= 1:
            return torch.tensor(0.0, device=cluster_means.device, dtype=torch.float32)
        
        # Ensure float32
        cluster_means = cluster_means.float()
        
        means_i = rearrange(cluster_means, "c e -> c 1 e")
        means_j = rearrange(cluster_means, "c e -> 1 c e")
        
        diff = means_i - means_j
        pairwise_dist = torch.norm(diff, p=self.norm, dim=2)
        
        triu_indices = torch.triu_indices(
            num_instances, num_instances, offset=1, device=cluster_means.device
        )
        upper_dists = pairwise_dist[triu_indices[0], triu_indices[1]]
        
        hinged = F.relu(2 * self.delta_dist - upper_dists) ** 2
        dist_loss = reduce(hinged, "n -> ", "mean")
        
        return dist_loss

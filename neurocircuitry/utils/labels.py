"""
Label utilities for connectomics segmentation.

Provides functions for:
- Relabeling/reindexing instance labels after cropping
- Connected component relabeling
- Instance segmentation metrics (ARI, AMI)
"""

from typing import Optional, Tuple, Union

import torch
import numpy as np
from einops import rearrange


def relabel_sequential(
    labels: torch.Tensor,
    start_label: int = 1,
) -> torch.Tensor:
    """
    Relabel instance labels to be sequential starting from start_label.
    
    Background (0) is preserved. All other unique labels are mapped to
    consecutive integers starting from start_label.
    
    Args:
        labels: Instance labels tensor of any shape.
        start_label: Starting label for foreground instances (default: 1).
    
    Returns:
        Relabeled tensor with sequential labels.
    
    Example:
        >>> labels = torch.tensor([0, 5, 0, 5, 12, 12, 0])
        >>> relabel_sequential(labels)
        tensor([0, 1, 0, 1, 2, 2, 0])
    """
    device = labels.device
    dtype = labels.dtype
    
    # Get unique labels
    unique_labels = torch.unique(labels)
    
    # Separate background and foreground
    fg_labels = unique_labels[unique_labels > 0]
    
    if len(fg_labels) == 0:
        return labels.clone()
    
    # Create mapping: old_label -> new_label
    max_label = int(labels.max().item()) + 1
    label_map = torch.zeros(max_label, device=device, dtype=dtype)
    
    for new_idx, old_label in enumerate(fg_labels):
        label_map[old_label.long()] = start_label + new_idx
    
    # Apply mapping (background stays 0)
    relabeled = label_map[labels.long().clamp(0, max_label - 1)]
    
    return relabeled


def relabel_connected_components_3d(
    labels: torch.Tensor,
    connectivity: int = 6,
) -> torch.Tensor:
    """
    Relabel 3D volume by finding connected components.
    
    After cropping, a single instance label might represent multiple
    disconnected components. This function assigns unique labels to
    each connected component.
    
    Args:
        labels: 3D label volume [D, H, W] or [B, D, H, W].
        connectivity: Connectivity for finding components (6, 18, or 26).
    
    Returns:
        Relabeled volume with unique labels for each connected component.
    """
    # Handle batch dimension
    if labels.dim() == 4:
        batch_results = []
        for b in range(labels.shape[0]):
            result = relabel_connected_components_3d(labels[b], connectivity)
            batch_results.append(result)
        return torch.stack(batch_results)
    
    device = labels.device
    labels_np = labels.cpu().numpy().astype(np.int32)
    
    try:
        from scipy import ndimage
        
        # Find connected components for each unique label
        unique_labels = np.unique(labels_np)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        relabeled = np.zeros_like(labels_np)
        next_label = 1
        
        # Define structure based on connectivity
        if connectivity == 6:
            structure = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 18:
            structure = ndimage.generate_binary_structure(3, 2)
        else:  # 26
            structure = ndimage.generate_binary_structure(3, 3)
        
        for old_label in unique_labels:
            mask = labels_np == old_label
            labeled_mask, num_features = ndimage.label(mask, structure=structure)
            
            for i in range(1, num_features + 1):
                relabeled[labeled_mask == i] = next_label
                next_label += 1
        
        return torch.from_numpy(relabeled).to(device=device, dtype=labels.dtype)
    
    except ImportError:
        # Fallback: just relabel sequentially without splitting
        return relabel_sequential(labels)


def relabel_connected_components_2d(
    labels: torch.Tensor,
    connectivity: int = 4,
) -> torch.Tensor:
    """
    Relabel 2D image by finding connected components.
    
    Args:
        labels: 2D label image [H, W] or [B, H, W].
        connectivity: Connectivity for finding components (4 or 8).
    
    Returns:
        Relabeled image with unique labels for each connected component.
    """
    # Handle batch dimension
    if labels.dim() == 3:
        batch_results = []
        for b in range(labels.shape[0]):
            result = relabel_connected_components_2d(labels[b], connectivity)
            batch_results.append(result)
        return torch.stack(batch_results)
    
    device = labels.device
    labels_np = labels.cpu().numpy().astype(np.int32)
    
    try:
        from scipy import ndimage
        
        unique_labels = np.unique(labels_np)
        unique_labels = unique_labels[unique_labels > 0]
        
        relabeled = np.zeros_like(labels_np)
        next_label = 1
        
        if connectivity == 4:
            structure = ndimage.generate_binary_structure(2, 1)
        else:  # 8
            structure = ndimage.generate_binary_structure(2, 2)
        
        for old_label in unique_labels:
            mask = labels_np == old_label
            labeled_mask, num_features = ndimage.label(mask, structure=structure)
            
            for i in range(1, num_features + 1):
                relabeled[labeled_mask == i] = next_label
                next_label += 1
        
        return torch.from_numpy(relabeled).to(device=device, dtype=labels.dtype)
    
    except ImportError:
        return relabel_sequential(labels)


def relabel_after_crop(
    labels: torch.Tensor,
    spatial_dims: int = 3,
    connectivity: Optional[int] = None,
) -> torch.Tensor:
    """
    Relabel instance labels after cropping.
    
    After cropping a volume/image, some instances may be split into
    disconnected components, or some may be entirely removed. This
    function:
    1. Finds connected components (to separate split instances)
    2. Relabels sequentially (to have consecutive IDs)
    
    Args:
        labels: Label tensor [D, H, W], [B, D, H, W], [H, W], or [B, H, W].
        spatial_dims: Number of spatial dimensions (2 or 3).
        connectivity: Connectivity for component detection (default: 6 for 3D, 4 for 2D).
    
    Returns:
        Relabeled tensor with sequential unique labels per component.
    
    Example:
        >>> # After cropping, label 5 might be split into two regions
        >>> cropped_labels = crop_function(original_labels)
        >>> relabeled = relabel_after_crop(cropped_labels, spatial_dims=3)
        >>> # Now each connected region has its own unique label
    """
    if spatial_dims == 3:
        if connectivity is None:
            connectivity = 6
        return relabel_connected_components_3d(labels, connectivity)
    elif spatial_dims == 2:
        if connectivity is None:
            connectivity = 4
        return relabel_connected_components_2d(labels, connectivity)
    else:
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")


def compute_ari_ami(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> Tuple[float, float]:
    """
    Compute Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI).
    
    These metrics evaluate the quality of instance segmentation by comparing
    predicted and ground truth label assignments.
    
    Args:
        pred_labels: Predicted instance labels [H, W] or [D, H, W].
        true_labels: Ground truth instance labels, same shape as pred_labels.
        ignore_background: If True, exclude background (label 0) from computation.
    
    Returns:
        Tuple of (ARI, AMI), both in range [0, 1] (clamped from [-1, 1]).
    
    Note:
        Requires sklearn. Returns (0.0, 0.0) if sklearn is not available or
        if there are no foreground pixels.
    """
    try:
        from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    except ImportError:
        return 0.0, 0.0
    
    # Flatten to 1D
    pred_flat = pred_labels.detach().cpu().numpy().ravel()
    true_flat = true_labels.detach().cpu().numpy().ravel()
    
    if ignore_background:
        # Only consider foreground pixels in either prediction or ground truth
        fg_mask = (pred_flat > 0) | (true_flat > 0)
        if not fg_mask.any():
            return 0.0, 0.0
        pred_flat = pred_flat[fg_mask]
        true_flat = true_flat[fg_mask]
    
    if len(pred_flat) == 0:
        return 0.0, 0.0
    
    # Compute metrics and clamp to [0, 1]
    ari = max(0.0, adjusted_rand_score(true_flat, pred_flat))
    ami = max(0.0, adjusted_mutual_info_score(true_flat, pred_flat))
    
    return ari, ami


def compute_batch_ari_ami(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    ignore_background: bool = True,
) -> Tuple[float, float]:
    """
    Compute ARI and AMI averaged over a batch.
    
    Args:
        pred_labels: Predicted labels [B, ...] (any spatial dims).
        true_labels: Ground truth labels [B, ...].
        ignore_background: If True, exclude background (label 0).
    
    Returns:
        Tuple of (mean_ARI, mean_AMI) over the batch.
    """
    batch_size = pred_labels.shape[0]
    ari_sum, ami_sum = 0.0, 0.0
    valid_count = 0
    
    for b in range(batch_size):
        ari, ami = compute_ari_ami(
            pred_labels[b], true_labels[b], ignore_background
        )
        if ari > 0 or ami > 0:  # Skip empty samples
            ari_sum += ari
            ami_sum += ami
            valid_count += 1
    
    if valid_count == 0:
        return 0.0, 0.0
    
    return ari_sum / valid_count, ami_sum / valid_count


def cluster_embeddings_meanshift(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
) -> torch.Tensor:
    """
    Cluster pixel embeddings using mean-shift clustering.
    
    Args:
        embedding: Pixel embeddings [E, D, H, W] or [E, H, W].
        foreground_mask: Binary mask, same spatial shape as embedding.
        bandwidth: Mean-shift bandwidth (related to delta_var).
        min_cluster_size: Minimum pixels per cluster.
    
    Returns:
        Instance labels with same spatial shape as embedding.
    """
    try:
        from sklearn.cluster import MeanShift
    except ImportError:
        # Fallback: all foreground is one instance
        if foreground_mask is not None:
            return foreground_mask.long()
        return (embedding[0] > 0).long()
    
    device = embedding.device
    is_3d = embedding.dim() == 4
    
    if is_3d:
        E, D, H, W = embedding.shape
        emb_flat = rearrange(embedding, "e d h w -> (d h w) e").cpu().numpy()
        spatial_shape = (D, H, W)
    else:
        E, H, W = embedding.shape
        emb_flat = rearrange(embedding, "e h w -> (h w) e").cpu().numpy()
        spatial_shape = (H, W)
    
    # Apply foreground mask
    if foreground_mask is not None:
        fg_flat = foreground_mask.cpu().numpy().ravel() > 0
    else:
        fg_flat = np.ones(emb_flat.shape[0], dtype=bool)
    
    fg_indices = np.where(fg_flat)[0]
    
    if len(fg_indices) == 0:
        return torch.zeros(spatial_shape, device=device, dtype=torch.long)
    
    emb_fg = emb_flat[fg_indices]
    
    # Cluster
    try:
        clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels_fg = clusterer.fit_predict(emb_fg) + 1  # +1 to reserve 0 for background
    except ValueError:
        # "No point was within bandwidth" - assign all to one cluster
        labels_fg = np.ones(len(emb_fg), dtype=np.int32)
    
    # Filter small clusters
    unique_labels, counts = np.unique(labels_fg, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label > 0 and count < min_cluster_size:
            labels_fg[labels_fg == label] = 0
    
    # Relabel sequentially
    unique_labels = np.unique(labels_fg)
    unique_labels = unique_labels[unique_labels > 0]
    label_map = {old: new + 1 for new, old in enumerate(unique_labels)}
    label_map[0] = 0
    labels_fg = np.array([label_map.get(l, 0) for l in labels_fg], dtype=np.int32)
    
    # Create full label map
    labels_full = np.zeros(emb_flat.shape[0], dtype=np.int32)
    labels_full[fg_indices] = labels_fg
    
    labels_out = labels_full.reshape(spatial_shape)
    
    return torch.from_numpy(labels_out).to(device=device, dtype=torch.long)

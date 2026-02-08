#!/usr/bin/env python
"""
Vista3D Inference Script with Sliding Window and Agglomeration.

Supports:
- Automatic segmentation of full volumes
- Interactive segmentation with point prompts
- Sliding window inference with configurable overlap
- Multiple aggregation strategies (gaussian, average, max)

Usage:
    # Automatic segmentation
    python scripts/infer_vista3d.py \
        --checkpoint checkpoints/vista3d/best.ckpt \
        --input data/SNEMI3D/AC3_inputs.h5 \
        --output output/AC3_segmentation.h5
    
    # Interactive segmentation with points
    python scripts/infer_vista3d.py \
        --checkpoint checkpoints/vista3d/best.ckpt \
        --input data/volume.h5 \
        --output output/segmentation.h5 \
        --mode interactive \
        --points "[[50,100,100],[60,110,110]]" \
        --point-labels "[1,1]"
    
    # Custom sliding window settings
    python scripts/infer_vista3d.py \
        --checkpoint checkpoints/vista3d/best.ckpt \
        --input data/volume.h5 \
        --output output/segmentation.h5 \
        --patch-size 128 128 128 \
        --stride 64 64 64 \
        --aggregation gaussian
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


def load_volume(path: str, key: Optional[str] = None) -> np.ndarray:
    """Load volume from file (HDF5, TIFF, NRRD, NPY)."""
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix in [".h5", ".hdf5"]:
        import h5py
        with h5py.File(path, "r") as f:
            if key is None:
                key = list(f.keys())[0]
            return f[key][:]
    
    elif suffix in [".tiff", ".tif"]:
        import tifffile
        return tifffile.imread(str(path))
    
    elif suffix == ".nrrd":
        import nrrd
        data, _ = nrrd.read(str(path))
        return data
    
    elif suffix == ".npy":
        return np.load(str(path))
    
    elif suffix == ".npz":
        data = np.load(str(path))
        if key is None:
            key = list(data.keys())[0]
        return data[key]
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_volume(
    data: np.ndarray,
    path: str,
    key: str = "segmentation",
) -> None:
    """Save volume to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    
    if suffix in [".h5", ".hdf5"]:
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset(key, data=data, compression="gzip")
    
    elif suffix in [".tiff", ".tif"]:
        import tifffile
        tifffile.imwrite(str(path), data)
    
    elif suffix == ".nrrd":
        import nrrd
        nrrd.write(str(path), data)
    
    elif suffix == ".npy":
        np.save(str(path), data)
    
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def create_gaussian_weight(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create 3D Gaussian weight map for blending."""
    pd, ph, pw = patch_size
    sigma = min(patch_size) * sigma_scale
    
    z = torch.arange(pd, device=device).float() - pd / 2
    y = torch.arange(ph, device=device).float() - ph / 2
    x = torch.arange(pw, device=device).float() - pw / 2
    
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    gaussian = torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))
    
    # Normalize
    gaussian = gaussian / gaussian.max()
    
    return gaussian


def sliding_window_inference(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    stride: Optional[Tuple[int, int, int]] = None,
    aggregation: str = "gaussian",
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
    progress: bool = True,
) -> torch.Tensor:
    """
    Perform sliding window inference on a 3D volume.
    
    Args:
        model: Segmentation model.
        volume: Input volume [C, D, H, W] or [D, H, W].
        patch_size: Size of patches (D, H, W).
        stride: Stride between patches. Default: patch_size // 2.
        aggregation: Blending mode ('gaussian', 'average', 'max').
        batch_size: Number of patches to process at once.
        device: Device for inference.
        progress: Show progress bar.
    
    Returns:
        Segmentation probabilities [num_classes, D, H, W].
    """
    model.eval()
    
    # Add channel dim if needed: [D, H, W] -> [1, D, H, W]
    if volume.dim() == 3:
        volume = rearrange(volume, "d h w -> 1 d h w")
    
    volume = volume.to(device)
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    # Default stride: 50% overlap
    if stride is None:
        stride = (pd // 2, ph // 2, pw // 2)
    sd, sh, sw = stride
    
    # Calculate grid dimensions
    nd = max(1, (D - pd + sd) // sd)
    nh = max(1, (H - ph + sh) // sh)
    nw = max(1, (W - pw + sw) // sw)
    
    # Pad volume to fit grid
    pad_d = max(0, (nd - 1) * sd + pd - D)
    pad_h = max(0, (nh - 1) * sh + ph - H)
    pad_w = max(0, (nw - 1) * sw + pw - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect")
        D_pad, H_pad, W_pad = D + pad_d, H + pad_h, W + pad_w
    else:
        D_pad, H_pad, W_pad = D, H, W
    
    # Determine output shape
    with torch.no_grad():
        # [C, pd, ph, pw] -> [1, C, pd, ph, pw]
        dummy_input = rearrange(volume[:, :pd, :ph, :pw], "c d h w -> 1 c d h w")
        dummy_output = model(dummy_input)
        if isinstance(dummy_output, dict):
            dummy_logits = dummy_output["logits"]
        else:
            dummy_logits = dummy_output
        num_classes = dummy_logits.shape[1]
    
    # Initialize output and weight tensors
    output = torch.zeros((num_classes, D_pad, H_pad, W_pad), device=device)
    weight = torch.zeros((1, D_pad, H_pad, W_pad), device=device)
    
    # Create weight map for blending
    if aggregation == "gaussian":
        patch_weight = create_gaussian_weight(patch_size, device=device)
    else:
        patch_weight = torch.ones(patch_size, device=device)
    
    # Collect all patch positions
    positions = []
    for i in range(nd):
        for j in range(nh):
            for k in range(nw):
                d_start = min(i * sd, D_pad - pd)
                h_start = min(j * sh, H_pad - ph)
                w_start = min(k * sw, W_pad - pw)
                positions.append((d_start, h_start, w_start))
    
    # Process patches in batches
    total_patches = len(positions)
    
    iterator = range(0, total_patches, batch_size)
    if progress:
        iterator = tqdm(iterator, desc="Sliding window inference", total=(total_patches + batch_size - 1) // batch_size)
    
    with torch.no_grad():
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, total_patches)
            batch_positions = positions[batch_start:batch_end]
            
            # Extract patches
            patches = []
            for d_start, h_start, w_start in batch_positions:
                patch = volume[
                    :,
                    d_start:d_start + pd,
                    h_start:h_start + ph,
                    w_start:w_start + pw,
                ]
                patches.append(patch)
            
            patches = torch.stack(patches, dim=0)  # [B, C, D, H, W]
            
            # Forward pass
            outputs = model(patches)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            
            # Apply softmax
            probs = F.softmax(logits, dim=1)
            
            # Aggregate results
            for idx, (d_start, h_start, w_start) in enumerate(batch_positions):
                if aggregation == "max":
                    # Take max probability
                    current = output[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ]
                    output[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] = torch.max(current, probs[idx])
                else:
                    # Weighted average
                    output[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += probs[idx] * patch_weight
                    
                    weight[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += patch_weight
    
    # Normalize by weight (for average/gaussian)
    if aggregation != "max":
        output = output / (weight + 1e-8)
    
    # Remove padding
    output = output[:, :D, :H, :W]
    
    return output


def interactive_inference(
    model: torch.nn.Module,
    volume: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Perform interactive segmentation with point prompts.
    
    Args:
        model: Segmentation model.
        volume: Input volume [C, D, H, W] or [D, H, W].
        point_coords: Point coordinates [N, 3] (z, y, x).
        point_labels: Point labels [N] (0=bg, 1=fg).
        patch_size: Size of patch around points.
        device: Device for inference.
    
    Returns:
        Binary segmentation mask [D, H, W].
    """
    model.eval()
    
    # Add channel dim if needed: [D, H, W] -> [1, D, H, W]
    if volume.dim() == 3:
        volume = rearrange(volume, "d h w -> 1 d h w")
    
    volume = volume.to(device)
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    point_coords = point_coords.to(device)
    point_labels = point_labels.to(device)
    
    # Get center of points
    center = point_coords.float().mean(dim=0).long()
    cz, cy, cx = center.tolist()
    
    # Calculate patch bounds centered on points
    d_start = max(0, cz - pd // 2)
    h_start = max(0, cy - ph // 2)
    w_start = max(0, cx - pw // 2)
    
    d_end = min(D, d_start + pd)
    h_end = min(H, h_start + ph)
    w_end = min(W, w_start + pw)
    
    # Adjust if we hit boundaries
    if d_end - d_start < pd:
        d_start = max(0, d_end - pd)
    if h_end - h_start < ph:
        h_start = max(0, h_end - ph)
    if w_end - w_start < pw:
        w_start = max(0, w_end - pw)
    
    # Extract patch
    patch = volume[
        :,
        d_start:d_start + pd,
        h_start:h_start + ph,
        w_start:w_start + pw,
    ]
    
    # Adjust point coordinates to patch space
    offset = torch.tensor([d_start, h_start, w_start], device=device)
    local_coords = point_coords - offset
    
    # Forward pass
    with torch.no_grad():
        # [C, D, H, W] -> [1, C, D, H, W], [N, 3] -> [1, N, 3], [N] -> [1, N]
        outputs = model(
            rearrange(patch, "c d h w -> 1 c d h w"),
            point_coords=rearrange(local_coords, "n c -> 1 n c").float(),
            point_labels=rearrange(point_labels, "n -> 1 n"),
        )
        
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
    
    # Get prediction: [1, num_classes, D, H, W] -> [num_classes, D, H, W]
    pred = rearrange(logits, "1 c d h w -> c d h w").argmax(dim=0)
    
    # Place in full output
    output = torch.zeros((D, H, W), device=device, dtype=torch.long)
    output[
        d_start:d_start + pd,
        h_start:h_start + ph,
        w_start:w_start + pw,
    ] = pred
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Vista3D Inference")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input volume")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output segmentation")
    
    # Inference mode
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "interactive"],
                        help="Inference mode")
    
    # Sliding window settings
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128],
                        help="Patch size (D H W)")
    parser.add_argument("--stride", type=int, nargs=3, default=None,
                        help="Stride between patches (D H W). Default: patch_size // 2")
    parser.add_argument("--aggregation", type=str, default="gaussian",
                        choices=["gaussian", "average", "max"],
                        help="Aggregation method for overlapping patches")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for patch processing")
    
    # Interactive mode settings
    parser.add_argument("--points", type=str, default=None,
                        help="Point coordinates as JSON: [[z1,y1,x1],[z2,y2,x2],...]")
    parser.add_argument("--point-labels", type=str, default=None,
                        help="Point labels as JSON: [1,1,0,...] (0=bg, 1=fg)")
    
    # Other settings
    parser.add_argument("--input-key", type=str, default=None,
                        help="Key for HDF5/NPZ input files")
    parser.add_argument("--output-key", type=str, default="segmentation",
                        help="Key for HDF5 output")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--output-probs", action="store_true",
                        help="Output probabilities instead of labels")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    from neurocircuitry.modules.vista3d_module import Vista3DModule
    
    module = Vista3DModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model = module.model.to(device)
    model.eval()
    
    # Load input volume
    print(f"Loading input: {args.input}")
    volume = load_volume(args.input, key=args.input_key)
    print(f"Input shape: {volume.shape}, dtype: {volume.dtype}")
    
    # Convert to tensor
    volume_tensor = torch.from_numpy(volume.astype(np.float32))
    
    # Normalize
    volume_tensor = (volume_tensor - volume_tensor.min()) / (volume_tensor.max() - volume_tensor.min() + 1e-8)
    
    # Run inference
    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride) if args.stride else None
    
    if args.mode == "auto":
        print(f"\nRunning automatic segmentation...")
        print(f"  Patch size: {patch_size}")
        print(f"  Stride: {stride or 'auto (50% overlap)'}")
        print(f"  Aggregation: {args.aggregation}")
        
        probs = sliding_window_inference(
            model,
            volume_tensor,
            patch_size=patch_size,
            stride=stride,
            aggregation=args.aggregation,
            batch_size=args.batch_size,
            device=device,
            progress=True,
        )
        
        if args.output_probs:
            output = probs.cpu().numpy()
        else:
            output = probs.argmax(dim=0).cpu().numpy().astype(np.uint16)
    
    elif args.mode == "interactive":
        if args.points is None or args.point_labels is None:
            raise ValueError("Interactive mode requires --points and --point-labels")
        
        points = torch.tensor(json.loads(args.points), dtype=torch.long)
        labels = torch.tensor(json.loads(args.point_labels), dtype=torch.long)
        
        print(f"\nRunning interactive segmentation...")
        print(f"  Points: {points.shape[0]}")
        print(f"  Foreground points: {(labels == 1).sum().item()}")
        print(f"  Background points: {(labels == 0).sum().item()}")
        
        output = interactive_inference(
            model,
            volume_tensor,
            points,
            labels,
            patch_size=patch_size,
            device=device,
        ).cpu().numpy().astype(np.uint16)
    
    # Save output
    print(f"\nSaving output to: {args.output}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    save_volume(output, args.output, key=args.output_key)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()

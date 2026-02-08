"""
I/O utilities for loading and saving connectomics data.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
    
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_volume(
    path: Union[str, Path],
    format: Optional[str] = None,
    key: str = "main",
) -> np.ndarray:
    """
    Load volume data from file.
    
    Automatically detects format based on extension if not specified.
    
    Args:
        path: Path to volume file.
        format: File format ('h5', 'tiff', 'nrrd'). Auto-detected if None.
        key: Dataset key for HDF5 files.
    
    Returns:
        Numpy array containing volume data.
    """
    path = Path(path)
    
    if format is None:
        format = path.suffix.lower().lstrip(".")
    
    if format in ["h5", "hdf5", "hdf"]:
        import h5py
        with h5py.File(path, "r") as f:
            if key in f:
                return f[key][:]
            else:
                # Try to find first dataset
                keys = list(f.keys())
                if keys:
                    return f[keys[0]][:]
                raise KeyError(f"No datasets found in {path}")
    
    elif format in ["tiff", "tif"]:
        import tifffile
        return tifffile.imread(str(path))
    
    elif format == "nrrd":
        import nrrd
        data, _ = nrrd.read(str(path))
        return data
    
    elif format == "npy":
        return np.load(path)
    
    elif format == "npz":
        data = np.load(path)
        keys = list(data.keys())
        if keys:
            return data[keys[0]]
        raise KeyError(f"No arrays found in {path}")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_volume(
    data: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    format: Optional[str] = None,
    key: str = "main",
    compression: Optional[str] = "gzip",
) -> None:
    """
    Save volume data to file.
    
    Automatically detects format based on extension if not specified.
    
    Args:
        data: Volume data as numpy array or torch tensor.
        path: Output file path.
        format: File format ('h5', 'tiff', 'nrrd'). Auto-detected if None.
        key: Dataset key for HDF5 files.
        compression: Compression for HDF5 files.
    """
    path = Path(path)
    ensure_directory(path.parent)
    
    # Convert tensor to numpy
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    if format is None:
        format = path.suffix.lower().lstrip(".")
    
    if format in ["h5", "hdf5", "hdf"]:
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset(key, data=data, compression=compression)
    
    elif format in ["tiff", "tif"]:
        import tifffile
        tifffile.imwrite(str(path), data)
    
    elif format == "nrrd":
        import nrrd
        nrrd.write(str(path), data)
    
    elif format == "npy":
        np.save(path, data)
    
    elif format == "npz":
        np.savez_compressed(path, **{key: data})
    
    else:
        raise ValueError(f"Unsupported format: {format}")

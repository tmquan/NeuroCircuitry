"""
SNEMI3D Dataset for neuron segmentation.

The SNEMI3D challenge dataset from the Kasthuri et al. (2015) study.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from neurocircuitry.datasets.base import BaseConnectomicsDataset
from neurocircuitry.preprocessors import HDF5Preprocessor, TIFFPreprocessor


class SNEMI3DDataset(BaseConnectomicsDataset):
    """
    SNEMI3D Dataset for neuron segmentation in electron microscopy images.
    
    Dataset from the SNEMI3D challenge (ISBI 2013) based on Kasthuri et al. (2015).
    Contains serial section electron microscopy images of mouse cortex with
    neuron instance segmentation labels.
    
    Dataset Structure:
        - Training: AC4 volume (100 slices, 1024x1024)
        - Testing: AC3 volume (100 slices, 1024x1024)
        - Resolution: 6x6x30 nm (anisotropic)
    
    Expected file structure:
        root_dir/
            AC3_inputs.h5 or AC3_inputs.tiff    # Test volume
            AC3_labels.h5 or AC3_labels.tiff    # Test labels (if available)
            AC4_inputs.h5 or AC4_inputs.tiff    # Train volume
            AC4_labels.h5 or AC4_labels.tiff    # Train labels
    
    Args:
        root_dir: Path to directory containing SNEMI3D data files.
        split: Data split ('train', 'valid', 'test').
        transform: Optional MONAI transforms to apply.
        cache_rate: Fraction of data to cache in memory (default: 1.0).
        train_val_split: Fraction for validation split (default: 0.2).
        slice_mode: If True, return individual 2D slices; if False, return
            3D volume patches (default: True).
    
    Example:
        >>> from neurocircuitry.datasets import SNEMI3DDataset
        >>> dataset = SNEMI3DDataset(
        ...     root_dir="/path/to/snemi3d",
        ...     split="train",
        ...     cache_rate=0.5
        ... )
        >>> sample = dataset[0]
        >>> print(sample["image"].shape)  # (1024, 1024)
    """
    
    # Class-level metadata
    _paper = (
        "Kasthuri, N., et al. (2015). Saturated Reconstruction of a Volume of "
        "Neocortex. Cell, 162(3), 648-661. doi:10.1016/j.cell.2015.06.054"
    )
    _resolution = {"x": 6.0, "y": 6.0, "z": 30.0}  # nanometers
    _labels = ["background", "neuron"]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        slice_mode: bool = True,
        num_workers: int = 0,
    ):
        self.slice_mode = slice_mode
        self._hdf5_preprocessor = HDF5Preprocessor()
        self._tiff_preprocessor = TIFFPreprocessor()
        
        super().__init__(
            root_dir=root_dir,
            split=split,
            transform=transform,
            cache_rate=cache_rate,
            train_val_split=train_val_split,
            num_workers=num_workers,
        )
    
    @property
    def paper(self) -> str:
        return self._paper
    
    @property
    def resolution(self) -> Dict[str, float]:
        return self._resolution.copy()
    
    @property
    def labels(self) -> List[str]:
        return self._labels.copy()
    
    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        """Return expected data files based on current split."""
        if self.split in ["train", "valid"]:
            return {
                "vol": "AC4_inputs",
                "seg": "AC4_labels",
            }
        else:  # test
            return {
                "vol": "AC3_inputs",
                "seg": "AC3_labels",
            }
    
    def _find_data_file(self, base_name: str) -> Optional[Path]:
        """
        Find data file with supported extension.
        
        Args:
            base_name: Base filename without extension.
        
        Returns:
            Path to file if found, None otherwise.
        """
        for ext in [".h5", ".hdf5", ".tiff", ".tif"]:
            path = self.root_dir / f"{base_name}{ext}"
            if path.exists():
                return path
        return None
    
    def _load_volume(self, base_name: str) -> np.ndarray:
        """
        Load volume data from file.
        
        Args:
            base_name: Base filename without extension.
        
        Returns:
            Numpy array containing volume data.
        
        Raises:
            FileNotFoundError: If no matching file is found.
        """
        path = self._find_data_file(base_name)
        
        if path is None:
            raise FileNotFoundError(
                f"Could not find data file '{base_name}' in {self.root_dir}.\n"
                f"Expected one of: {base_name}.h5, {base_name}.hdf5, "
                f"{base_name}.tiff, {base_name}.tif"
            )
        
        suffix = path.suffix.lower()
        if suffix in [".h5", ".hdf5"]:
            return self._hdf5_preprocessor.load(str(path))
        else:  # .tiff, .tif
            return self._tiff_preprocessor.load(str(path))
    
    def _prepare_data(self) -> List[Dict[str, Any]]:
        """
        Prepare data dictionaries based on split.
        
        For train/valid: Uses AC4 volume with train_val_split.
        For test: Uses AC3 volume.
        
        Returns:
            List of dictionaries with 'image', 'label', and metadata.
        """
        data_list = []
        files = self.data_files
        
        # Load volumes
        try:
            inputs = self._load_volume(files["vol"])
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"SNEMI3D volume data not found.\n{e}\n"
                f"Please download the SNEMI3D dataset from: "
                f"https://snemi3d.grand-challenge.org/"
            )
        
        # Load labels (may not exist for test split)
        labels = None
        if self.split in ["train", "valid"]:
            try:
                labels = self._load_volume(files["seg"])
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"SNEMI3D label data not found for {self.split} split.\n{e}"
                )
        else:
            # Test split - labels may or may not exist
            try:
                labels = self._load_volume(files["seg"])
            except FileNotFoundError:
                labels = None  # No labels for test set
        
        # Calculate split indices
        n_total = inputs.shape[0]
        n_train = int(n_total * (1.0 - self.train_val_split))
        
        if self.split == "train":
            slice_range = range(n_train)
            volume_name = "AC4_train"
        elif self.split == "valid":
            slice_range = range(n_train, n_total)
            volume_name = "AC4_valid"
        else:  # test
            slice_range = range(n_total)
            volume_name = "AC3_test"
        
        # Create data dictionaries
        if self.slice_mode:
            # Return individual 2D slices
            for i in slice_range:
                data_dict = {
                    "image": inputs[i],
                    "slice_idx": i,
                    "volume": volume_name,
                    "idx": i,
                }
                
                if labels is not None:
                    data_dict["label"] = labels[i]
                
                data_list.append(data_dict)
        else:
            # Return full 3D volume (or split portion)
            vol_inputs = inputs[list(slice_range)]
            data_dict = {
                "image": vol_inputs,
                "volume": volume_name,
                "idx": 0,
            }
            
            if labels is not None:
                vol_labels = labels[list(slice_range)]
                data_dict["label"] = vol_labels
            
            data_list.append(data_dict)
        
        return data_list

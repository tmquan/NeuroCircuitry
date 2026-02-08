"""
CREMI3D Dataset for neuron and synapse segmentation.

The CREMI challenge dataset from MICCAI 2016.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from neurocircuitry.datasets.base import BaseConnectomicsDataset
from neurocircuitry.preprocessors import HDF5Preprocessor, TIFFPreprocessor


class CREMI3DDataset(BaseConnectomicsDataset):
    """
    CREMI3D Dataset for neuron and synapse segmentation.
    
    Dataset from the CREMI challenge (MICCAI 2016) containing electron
    microscopy images of Drosophila brain with neuron instance segmentation
    and synaptic cleft annotations.
    
    Dataset Structure:
        - Samples: A, B, C (each 125 slices, 1250x1250)
        - Annotations: Neuron segmentation + synaptic cleft labels
        - Resolution: 4x4x40 nm (anisotropic)
    
    Expected file structure:
        root_dir/
            sample_A_20160501.hdf or sample_A.h5
            sample_B_20160501.hdf or sample_B.h5
            sample_C_20160501.hdf or sample_C.h5
    
    HDF5 structure:
        /volumes/raw          - Raw EM images
        /volumes/labels/neuron_ids   - Neuron segmentation
        /volumes/labels/clefts       - Synaptic cleft labels
    
    Args:
        root_dir: Path to directory containing CREMI data files.
        split: Data split ('train', 'valid', 'test').
        transform: Optional MONAI transforms to apply.
        cache_rate: Fraction of data to cache in memory (default: 1.0).
        train_val_split: Fraction for validation split (default: 0.2).
        samples: List of samples to use ('A', 'B', 'C'). Default: all.
        include_synapses: Whether to include synaptic cleft labels (default: True).
        slice_mode: If True, return individual 2D slices (default: True).
    
    Example:
        >>> from neurocircuitry.datasets import CREMI3DDataset
        >>> dataset = CREMI3DDataset(
        ...     root_dir="/path/to/cremi",
        ...     split="train",
        ...     samples=["A", "B"],
        ...     include_synapses=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample["image"].shape)  # (1250, 1250)
    """
    
    # Class-level metadata
    _paper = (
        "Funke, J., et al. (2016). CREMI: MICCAI Challenge on Circuit "
        "Reconstruction from Electron Microscopy Images. "
        "https://cremi.org/"
    )
    _resolution = {"x": 4.0, "y": 4.0, "z": 40.0}  # nanometers
    _labels_base = ["background", "neuron"]
    _labels_with_synapse = ["background", "neuron", "synapse_cleft"]
    
    # HDF5 dataset paths
    _RAW_PATH = "volumes/raw"
    _NEURON_PATH = "volumes/labels/neuron_ids"
    _CLEFT_PATH = "volumes/labels/clefts"
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        samples: Optional[List[str]] = None,
        include_synapses: bool = True,
        slice_mode: bool = True,
        num_workers: int = 0,
    ):
        self.samples = samples if samples is not None else ["A", "B", "C"]
        self.include_synapses = include_synapses
        self.slice_mode = slice_mode
        self._hdf5_preprocessor = HDF5Preprocessor()
        
        # Validate samples
        valid_samples = ["A", "B", "C"]
        for s in self.samples:
            if s.upper() not in valid_samples:
                raise ValueError(
                    f"Invalid sample '{s}'. Must be one of {valid_samples}"
                )
        self.samples = [s.upper() for s in self.samples]
        
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
        if self.include_synapses:
            return self._labels_with_synapse.copy()
        return self._labels_base.copy()
    
    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        """Return expected data files for all samples."""
        return {
            "vol": [f"sample_{s}" for s in self.samples],
            "seg": [f"sample_{s}" for s in self.samples],
        }
    
    def _find_sample_file(self, sample: str) -> Optional[Path]:
        """
        Find sample file with supported naming conventions.
        
        Args:
            sample: Sample identifier ('A', 'B', or 'C').
        
        Returns:
            Path to file if found, None otherwise.
        """
        # Try different naming conventions
        patterns = [
            f"sample_{sample}_20160501.hdf",
            f"sample_{sample}_20160501.h5",
            f"sample_{sample}.hdf",
            f"sample_{sample}.h5",
            f"sample_{sample.lower()}.hdf",
            f"sample_{sample.lower()}.h5",
        ]
        
        for pattern in patterns:
            path = self.root_dir / pattern
            if path.exists():
                return path
        
        return None
    
    def _load_sample(self, sample: str) -> Dict[str, np.ndarray]:
        """
        Load all data from a CREMI sample file.
        
        Args:
            sample: Sample identifier ('A', 'B', or 'C').
        
        Returns:
            Dictionary with 'raw', 'neuron_ids', and optionally 'clefts'.
        
        Raises:
            FileNotFoundError: If sample file is not found.
        """
        path = self._find_sample_file(sample)
        
        if path is None:
            raise FileNotFoundError(
                f"Could not find CREMI sample_{sample} in {self.root_dir}.\n"
                f"Expected files like: sample_{sample}_20160501.hdf, "
                f"sample_{sample}.h5, etc.\n"
                f"Please download from: https://cremi.org/data/"
            )
        
        import h5py
        
        data = {}
        with h5py.File(path, "r") as f:
            # Load raw data
            if self._RAW_PATH in f:
                data["raw"] = f[self._RAW_PATH][:]
            else:
                raise KeyError(
                    f"Raw data not found at '{self._RAW_PATH}' in {path}"
                )
            
            # Load neuron segmentation
            if self._NEURON_PATH in f:
                data["neuron_ids"] = f[self._NEURON_PATH][:]
            else:
                raise KeyError(
                    f"Neuron labels not found at '{self._NEURON_PATH}' in {path}"
                )
            
            # Load synaptic clefts (optional)
            if self.include_synapses and self._CLEFT_PATH in f:
                data["clefts"] = f[self._CLEFT_PATH][:]
        
        return data
    
    def _prepare_data(self) -> List[Dict[str, Any]]:
        """
        Prepare data dictionaries based on split.
        
        Combines data from all specified samples and splits according
        to train_val_split.
        
        Returns:
            List of dictionaries with 'image', 'label', and metadata.
        """
        data_list = []
        
        # Determine which samples to use for each split
        # Use all samples for train/valid, split within each sample
        for sample in self.samples:
            try:
                sample_data = self._load_sample(sample)
            except FileNotFoundError as e:
                if self.split == "test":
                    continue  # Skip missing samples in test mode
                raise
            
            raw = sample_data["raw"]
            neuron_ids = sample_data["neuron_ids"]
            clefts = sample_data.get("clefts", None)
            
            n_total = raw.shape[0]
            n_train = int(n_total * (1.0 - self.train_val_split))
            
            if self.split == "train":
                slice_range = range(n_train)
            elif self.split == "valid":
                slice_range = range(n_train, n_total)
            else:  # test
                slice_range = range(n_total)
            
            if self.slice_mode:
                # Return individual 2D slices
                for i in slice_range:
                    data_dict = {
                        "image": raw[i],
                        "label": neuron_ids[i],
                        "slice_idx": i,
                        "sample": sample,
                        "idx": len(data_list),
                    }
                    
                    if clefts is not None:
                        data_dict["clefts"] = clefts[i]
                    
                    data_list.append(data_dict)
            else:
                # Return 3D volume portion
                vol_raw = raw[list(slice_range)]
                vol_neuron = neuron_ids[list(slice_range)]
                
                data_dict = {
                    "image": vol_raw,
                    "label": vol_neuron,
                    "sample": sample,
                    "idx": len(data_list),
                }
                
                if clefts is not None:
                    data_dict["clefts"] = clefts[list(slice_range)]
                
                data_list.append(data_dict)
        
        return data_list

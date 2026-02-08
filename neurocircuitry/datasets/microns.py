"""
MICRONS Dataset for large-scale connectomics.

The MICrONS (Machine Intelligence from Cortical Networks) dataset.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from neurocircuitry.datasets.base import BaseConnectomicsDataset
from neurocircuitry.preprocessors import HDF5Preprocessor, TIFFPreprocessor, NRRDPreprocessor


class MICRONSDataset(BaseConnectomicsDataset):
    """
    MICRONS Dataset for large-scale cortical connectomics.
    
    Dataset from the MICrONS (Machine Intelligence from Cortical Networks)
    consortium containing petascale electron microscopy imaging of mouse
    visual cortex with dense neuron segmentation and synapse annotations.
    
    Dataset Structure:
        - Multiple cortical regions with varying sizes
        - Annotations: Neuron segmentation, synapses, mitochondria
        - Resolution: 4x4x40 nm (typical, varies by acquisition)
    
    Note: MICRONS data is typically accessed via cloud APIs (e.g., CAVEclient).
    This dataset class supports local file-based access for downloaded subvolumes.
    
    Expected file structure:
        root_dir/
            volume.h5 or volume.tiff    # EM volume data
            segmentation.h5             # Neuron segmentation
            synapses.h5                 # Synapse annotations (optional)
            mitochondria.h5             # Mitochondria labels (optional)
    
    Args:
        root_dir: Path to directory containing MICRONS data files.
        split: Data split ('train', 'valid', 'test').
        transform: Optional MONAI transforms to apply.
        cache_rate: Fraction of data to cache in memory (default: 1.0).
        train_val_split: Fraction for validation split (default: 0.2).
        volume_file: Name of volume file (default: 'volume').
        segmentation_file: Name of segmentation file (default: 'segmentation').
        include_synapses: Whether to load synapse annotations (default: False).
        include_mitochondria: Whether to load mitochondria labels (default: False).
        slice_mode: If True, return individual 2D slices (default: True).
        patch_size: If not None, return 3D patches of this size (z, y, x).
        patch_overlap: Overlap between patches (default: 0.25).
    
    Example:
        >>> from neurocircuitry.datasets import MICRONSDataset
        >>> dataset = MICRONSDataset(
        ...     root_dir="/path/to/microns_subvolume",
        ...     split="train",
        ...     include_synapses=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample["image"].shape)
    """
    
    # Class-level metadata
    _paper = (
        "MICrONS Consortium (2021). Functional connectomics spanning multiple "
        "areas of mouse visual cortex. bioRxiv. doi:10.1101/2021.07.28.454025"
    )
    _resolution = {"x": 4.0, "y": 4.0, "z": 40.0}  # nanometers (typical)
    _labels_base = ["background", "neuron"]
    _labels_extended = ["background", "neuron", "synapse", "mitochondria"]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        volume_file: str = "volume",
        segmentation_file: str = "segmentation",
        include_synapses: bool = False,
        include_mitochondria: bool = False,
        slice_mode: bool = True,
        patch_size: Optional[Tuple[int, int, int]] = None,
        patch_overlap: float = 0.25,
        num_workers: int = 0,
    ):
        self.volume_file = volume_file
        self.segmentation_file = segmentation_file
        self.include_synapses = include_synapses
        self.include_mitochondria = include_mitochondria
        self.slice_mode = slice_mode
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        self._hdf5_preprocessor = HDF5Preprocessor()
        self._tiff_preprocessor = TIFFPreprocessor()
        self._nrrd_preprocessor = NRRDPreprocessor()
        
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
        if self.include_synapses or self.include_mitochondria:
            return self._labels_extended.copy()
        return self._labels_base.copy()
    
    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        """Return expected data files."""
        files = {
            "vol": self.volume_file,
            "seg": self.segmentation_file,
        }
        if self.include_synapses:
            files["synapses"] = "synapses"
        if self.include_mitochondria:
            files["mitochondria"] = "mitochondria"
        return files
    
    def _find_data_file(self, base_name: str) -> Optional[Path]:
        """
        Find data file with supported extension.
        
        Args:
            base_name: Base filename without extension.
        
        Returns:
            Path to file if found, None otherwise.
        """
        for ext in [".h5", ".hdf5", ".tiff", ".tif", ".nrrd"]:
            path = self.root_dir / f"{base_name}{ext}"
            if path.exists():
                return path
        return None
    
    def _load_volume(self, base_name: str, required: bool = True) -> Optional[np.ndarray]:
        """
        Load volume data from file.
        
        Args:
            base_name: Base filename without extension.
            required: If True, raise error when not found.
        
        Returns:
            Numpy array containing volume data, or None if not found and not required.
        
        Raises:
            FileNotFoundError: If required and no matching file is found.
        """
        path = self._find_data_file(base_name)
        
        if path is None:
            if required:
                raise FileNotFoundError(
                    f"Could not find data file '{base_name}' in {self.root_dir}.\n"
                    f"Expected one of: {base_name}.h5, {base_name}.tiff, {base_name}.nrrd"
                )
            return None
        
        suffix = path.suffix.lower()
        if suffix in [".h5", ".hdf5"]:
            return self._hdf5_preprocessor.load(str(path))
        elif suffix in [".tiff", ".tif"]:
            return self._tiff_preprocessor.load(str(path))
        else:  # .nrrd
            return self._nrrd_preprocessor.load(str(path))
    
    def _generate_patch_indices(
        self,
        volume_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        overlap: float,
    ) -> List[Tuple[slice, slice, slice]]:
        """
        Generate indices for extracting overlapping 3D patches.
        
        Args:
            volume_shape: Shape of the full volume (z, y, x).
            patch_size: Size of each patch (z, y, x).
            overlap: Overlap fraction between patches.
        
        Returns:
            List of (z_slice, y_slice, x_slice) tuples.
        """
        indices = []
        
        for dim in range(3):
            vol_size = volume_shape[dim]
            patch_dim = patch_size[dim]
            stride = int(patch_dim * (1 - overlap))
            stride = max(1, stride)  # Ensure at least stride of 1
            
            dim_indices = []
            start = 0
            while start < vol_size:
                end = min(start + patch_dim, vol_size)
                # Adjust start if patch goes beyond volume
                if end - start < patch_dim and start > 0:
                    start = max(0, end - patch_dim)
                dim_indices.append((start, end))
                if end >= vol_size:
                    break
                start += stride
            
            indices.append(dim_indices)
        
        # Generate all combinations
        patch_indices = []
        for z_start, z_end in indices[0]:
            for y_start, y_end in indices[1]:
                for x_start, x_end in indices[2]:
                    patch_indices.append((
                        slice(z_start, z_end),
                        slice(y_start, y_end),
                        slice(x_start, x_end),
                    ))
        
        return patch_indices
    
    def _prepare_data(self) -> List[Dict[str, Any]]:
        """
        Prepare data dictionaries based on split.
        
        Returns:
            List of dictionaries with 'image', 'label', and metadata.
        """
        data_list = []
        files = self.data_files
        
        # Load main volumes
        try:
            inputs = self._load_volume(files["vol"])
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"MICRONS volume data not found.\n{e}\n"
                f"Please ensure MICRONS data is downloaded to {self.root_dir}"
            )
        
        # Load segmentation (required for train/valid)
        labels = None
        if self.split in ["train", "valid"]:
            labels = self._load_volume(files["seg"], required=True)
        else:
            labels = self._load_volume(files["seg"], required=False)
        
        # Load optional annotations
        synapses = None
        mitochondria = None
        if self.include_synapses:
            synapses = self._load_volume("synapses", required=False)
        if self.include_mitochondria:
            mitochondria = self._load_volume("mitochondria", required=False)
        
        # Calculate split indices along z-axis
        n_total = inputs.shape[0]
        n_train = int(n_total * (1.0 - self.train_val_split))
        
        if self.split == "train":
            z_range = range(n_train)
        elif self.split == "valid":
            z_range = range(n_train, n_total)
        else:  # test
            z_range = range(n_total)
        
        # Extract relevant portion
        inputs_split = inputs[list(z_range)]
        labels_split = labels[list(z_range)] if labels is not None else None
        synapses_split = synapses[list(z_range)] if synapses is not None else None
        mito_split = mitochondria[list(z_range)] if mitochondria is not None else None
        
        if self.slice_mode:
            # Return individual 2D slices
            for i in range(inputs_split.shape[0]):
                data_dict = {
                    "image": inputs_split[i],
                    "slice_idx": z_range[i] if isinstance(z_range, range) else i,
                    "idx": len(data_list),
                }
                
                if labels_split is not None:
                    data_dict["label"] = labels_split[i]
                if synapses_split is not None:
                    data_dict["synapses"] = synapses_split[i]
                if mito_split is not None:
                    data_dict["mitochondria"] = mito_split[i]
                
                data_list.append(data_dict)
        
        elif self.patch_size is not None:
            # Return 3D patches
            patch_indices = self._generate_patch_indices(
                inputs_split.shape,
                self.patch_size,
                self.patch_overlap,
            )
            
            for idx, (z_sl, y_sl, x_sl) in enumerate(patch_indices):
                data_dict = {
                    "image": inputs_split[z_sl, y_sl, x_sl],
                    "patch_idx": idx,
                    "patch_location": (z_sl.start, y_sl.start, x_sl.start),
                    "idx": len(data_list),
                }
                
                if labels_split is not None:
                    data_dict["label"] = labels_split[z_sl, y_sl, x_sl]
                if synapses_split is not None:
                    data_dict["synapses"] = synapses_split[z_sl, y_sl, x_sl]
                if mito_split is not None:
                    data_dict["mitochondria"] = mito_split[z_sl, y_sl, x_sl]
                
                data_list.append(data_dict)
        
        else:
            # Return full volume
            data_dict = {
                "image": inputs_split,
                "idx": 0,
            }
            
            if labels_split is not None:
                data_dict["label"] = labels_split
            if synapses_split is not None:
                data_dict["synapses"] = synapses_split
            if mito_split is not None:
                data_dict["mitochondria"] = mito_split
            
            data_list.append(data_dict)
        
        return data_list

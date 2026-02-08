"""
CREMI3D Dataset for connectomics instance segmentation.

CREMI (Circuit Reconstruction from Electron Microscopy Images) Challenge:
- 3 volumes: A, B, C (training A, B have labels; C is test)
- Resolution: 4nm x 4nm x 40nm (anisotropic)
- Annotations: neurons, synaptic clefts, (optionally mitochondria)

Data format:
- image: raw EM volume [Z, Y, X]
- label: instance segmentation (neuron IDs, cleft IDs merged with offsets)
- class_ids: semantic class per pixel (background=0, neuron=1, cleft=2, mito=3)

Reference:
- https://cremi.org/
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseConnectomicsDataset


class CREMI3DDataset(BaseConnectomicsDataset):
    """
    CREMI3D dataset for neuron and synapse segmentation.
    
    Expected directory structure:
        data_root/
        ├── sample_A.h5 (or sample_A/)
        │   ├── volumes/raw (EM image)
        │   ├── volumes/labels/neuron_ids
        │   └── volumes/labels/clefts (optional)
        ├── sample_B.h5
        └── sample_C.h5 (test, no labels)
    
    Or alternative structure:
        data_root/
        ├── sample_A_raw.h5
        ├── sample_A_neuron_ids.h5
        ├── sample_A_clefts.h5
        └── ...
    
    Attributes:
        data_root: Root directory containing CREMI data.
        volumes: List of volume names to load ["A", "B", "C"].
        include_clefts: Whether to include synaptic cleft annotations.
        include_mito: Whether to include mitochondria annotations.
        id_offset: Offset to add to instance IDs for each class.
    """
    
    # Class ID offsets for merging different annotation types
    NEURON_ID_OFFSET = 0
    CLEFT_ID_OFFSET = 100000
    MITO_ID_OFFSET = 200000
    
    # Semantic class IDs
    CLASS_BACKGROUND = 0
    CLASS_NEURON = 1
    CLASS_CLEFT = 2
    CLASS_MITO = 3
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform=None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        num_workers: int = 0,
        volumes: List[str] = ["A", "B"],
        include_clefts: bool = True,
        include_mito: bool = False,
        **kwargs,
    ):
        """
        Initialize CREMI3D dataset.
        
        Args:
            root_dir: Root directory containing CREMI data.
            split: "train", "valid", or "test".
            transform: Optional transforms to apply.
            cache_rate: Fraction of data to cache in memory.
            train_val_split: Fraction for validation split.
            num_workers: Number of workers for data loading.
            volumes: List of volume names ["A", "B", "C"].
            include_clefts: Include synaptic cleft annotations.
            include_mito: Include mitochondria annotations.
        """
        self.volumes = volumes
        self.include_clefts = include_clefts
        self.include_mito = include_mito
        self._image_data = None
        self._label_data = None
        
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
        """Reference for CREMI dataset."""
        return "CREMI Challenge - https://cremi.org/"
    
    @property
    def resolution(self) -> Dict[str, float]:
        """Voxel resolution in nanometers (4nm x 4nm x 40nm)."""
        return {"x": 4.0, "y": 4.0, "z": 40.0}
    
    @property
    def labels(self) -> List[str]:
        """List of segmentation class labels."""
        return ["background", "neuron", "cleft", "mito"]
    
    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        """Dictionary specifying volume and segmentation data sources."""
        # Return cached data arrays if available
        if self._image_data is not None and self._label_data is not None:
            return {"vol": self._image_data, "seg": self._label_data}
        # Otherwise return file pattern
        return {
            "vol": f"sample_*.h5/volumes/raw",
            "seg": f"sample_*.h5/volumes/labels/neuron_ids",
        }
    
    def _prepare_data(self) -> List[Dict[str, Any]]:
        """Prepare list of data dictionaries for each sample."""
        # Load all volumes
        image, label = self._load_data()
        self._image_data = image
        self._label_data = label
        
        # Calculate split indices based on Z dimension
        total_slices = image.shape[0]
        
        # Determine volume name for metadata
        volumes_str = "_".join(self.volumes)
        
        if self.split == "test":
            # Test uses last 20% or separate test volume
            start_idx = int(total_slices * (1 - self.train_val_split))
            image = image[start_idx:]
            label = label[start_idx:]
            volume_name = f"CREMI_{volumes_str}_test"
        elif self.split == "valid":
            # Validation uses train_val_split fraction
            val_start = int(total_slices * (1 - self.train_val_split))
            image = image[val_start:]
            label = label[val_start:]
            volume_name = f"CREMI_{volumes_str}_valid"
        else:  # train
            # Training uses first (1 - train_val_split) fraction
            train_end = int(total_slices * (1 - self.train_val_split))
            image = image[:train_end]
            label = label[:train_end]
            volume_name = f"CREMI_{volumes_str}_train"
        
        # Return single data dict (whole volume as one sample)
        # Include 'volume' and 'idx' keys to match SNEMI3D format
        return [{
            "image": image,
            "label": label,
            "volume": volume_name,
            "idx": 0,
        }]
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and merge CREMI volumes."""
        all_images = []
        all_labels = []
        
        for vol_name in self.volumes:
            image, label = self._load_volume(vol_name)
            if image is not None:
                all_images.append(image)
                all_labels.append(label)
        
        if not all_images:
            raise ValueError(f"No data found in {self.root_dir}")
        
        # Concatenate volumes along Z axis
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return images, labels
    
    def _load_volume(
        self,
        vol_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single CREMI volume."""
        # Try different file naming conventions used by CREMI
        possible_paths = [
            # Standard CREMI challenge format
            self.root_dir / f"sample_{vol_name}_20160501.hdf",
            self.root_dir / f"sample_{vol_name}+_20160601.hdf",
            self.root_dir / f"sample_{vol_name}_padded_20160501.hdf",
            # Common variants
            self.root_dir / f"sample_{vol_name}.h5",
            self.root_dir / f"sample_{vol_name}.hdf5",
            self.root_dir / f"sample_{vol_name}.hdf",
            self.root_dir / vol_name / "sample.h5",
        ]
        
        h5_path = None
        for path in possible_paths:
            if path.exists():
                h5_path = path
                break
        
        if h5_path is None:
            # Try separate files
            return self._load_volume_separate_files(vol_name)
        
        # Load from single HDF5 file
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                # Standard CREMI format
                if "volumes/raw" in f:
                    image = f["volumes/raw"][:]
                elif "raw" in f:
                    image = f["raw"][:]
                else:
                    print(f"Warning: No raw data found in {h5_path}")
                    return None, None
                
                # Load neuron labels
                label = np.zeros_like(image, dtype=np.int64)
                
                if "volumes/labels/neuron_ids" in f:
                    neuron_ids = f["volumes/labels/neuron_ids"][:]
                    label[neuron_ids > 0] = neuron_ids[neuron_ids > 0] + self.NEURON_ID_OFFSET
                elif "neuron_ids" in f:
                    neuron_ids = f["neuron_ids"][:]
                    label[neuron_ids > 0] = neuron_ids[neuron_ids > 0] + self.NEURON_ID_OFFSET
                
                # Load cleft labels (optional)
                if self.include_clefts:
                    if "volumes/labels/clefts" in f:
                        cleft_ids = f["volumes/labels/clefts"][:]
                        label[cleft_ids > 0] = cleft_ids[cleft_ids > 0] + self.CLEFT_ID_OFFSET
                    elif "clefts" in f:
                        cleft_ids = f["clefts"][:]
                        label[cleft_ids > 0] = cleft_ids[cleft_ids > 0] + self.CLEFT_ID_OFFSET
                
                # Load mito labels (optional)
                if self.include_mito:
                    if "volumes/labels/mitochondria" in f:
                        mito_ids = f["volumes/labels/mitochondria"][:]
                        label[mito_ids > 0] = mito_ids[mito_ids > 0] + self.MITO_ID_OFFSET
                    elif "mitochondria" in f:
                        mito_ids = f["mitochondria"][:]
                        label[mito_ids > 0] = mito_ids[mito_ids > 0] + self.MITO_ID_OFFSET
                
                return image.astype(np.float32), label
                
        except Exception as e:
            print(f"Warning: Failed to load {h5_path}: {e}")
            return None, None
    
    def _load_volume_separate_files(
        self,
        vol_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load volume from separate files."""
        import h5py
        
        def load_h5(path):
            """Load first dataset from HDF5 file."""
            with h5py.File(path, "r") as f:
                # Get first dataset
                def find_dataset(group):
                    for key in group.keys():
                        if isinstance(group[key], h5py.Dataset):
                            return group[key][:]
                        elif isinstance(group[key], h5py.Group):
                            result = find_dataset(group[key])
                            if result is not None:
                                return result
                    return None
                return find_dataset(f)
        
        # Try to load raw image
        raw_paths = [
            self.root_dir / f"sample_{vol_name}_raw.h5",
            self.root_dir / f"{vol_name}_raw.h5",
            self.root_dir / f"{vol_name}_image.h5",
        ]
        
        image = None
        for path in raw_paths:
            if path.exists():
                image = load_h5(str(path))
                break
        
        if image is None:
            return None, None
        
        label = np.zeros_like(image, dtype=np.int64)
        
        # Load neuron labels
        neuron_paths = [
            self.root_dir / f"sample_{vol_name}_neuron_ids.h5",
            self.root_dir / f"{vol_name}_neuron_ids.h5",
            self.root_dir / f"{vol_name}_labels.h5",
        ]
        
        for path in neuron_paths:
            if path.exists():
                neuron_ids = load_h5(str(path))
                label[neuron_ids > 0] = neuron_ids[neuron_ids > 0] + self.NEURON_ID_OFFSET
                break
        
        # Load cleft labels
        if self.include_clefts:
            cleft_paths = [
                self.root_dir / f"sample_{vol_name}_clefts.h5",
                self.root_dir / f"{vol_name}_clefts.h5",
            ]
            for path in cleft_paths:
                if path.exists():
                    cleft_ids = load_h5(str(path))
                    label[cleft_ids > 0] = cleft_ids[cleft_ids > 0] + self.CLEFT_ID_OFFSET
                    break
        
        return image.astype(np.float32), label
    
    def get_class_mapping(self) -> Dict[str, List[int]]:
        """
        Get mapping from class name to instance ID ranges.
        
        Returns:
            Dict mapping class names to ID ranges.
        """
        return {
            "neuron": list(range(1, self.CLEFT_ID_OFFSET)),
            "cleft": list(range(self.CLEFT_ID_OFFSET, self.MITO_ID_OFFSET)),
            "mito": list(range(self.MITO_ID_OFFSET, self.MITO_ID_OFFSET + 100000)),
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return ["background", "neuron", "cleft", "mito"]
    
    @staticmethod
    def instance_id_to_class(instance_id: int) -> int:
        """Map instance ID to semantic class ID."""
        if instance_id == 0:
            return CREMI3DDataset.CLASS_BACKGROUND
        elif instance_id < CREMI3DDataset.CLEFT_ID_OFFSET:
            return CREMI3DDataset.CLASS_NEURON
        elif instance_id < CREMI3DDataset.MITO_ID_OFFSET:
            return CREMI3DDataset.CLASS_CLEFT
        else:
            return CREMI3DDataset.CLASS_MITO

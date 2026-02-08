"""
Tests for dataset base classes and common functionality.

Individual dataset tests are in separate files:
- test_snemi3d_dataset.py
- test_cremi3d_dataset.py (TODO)
- test_microns_dataset.py (TODO)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from neurocircuitry.datasets.base import BaseConnectomicsDataset


class TestBaseConnectomicsDataset:
    """Tests for BaseConnectomicsDataset abstract class."""
    
    def test_abstract_class_cannot_instantiate(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseConnectomicsDataset(root_dir=".")
    
    def test_invalid_split_raises_error(self):
        """Test that invalid split raises ValueError."""
        
        class MinimalDataset(BaseConnectomicsDataset):
            """Minimal concrete implementation for testing."""
            
            @property
            def paper(self):
                return "Test Paper"
            
            @property
            def resolution(self):
                return {"x": 1.0, "y": 1.0, "z": 1.0}
            
            @property
            def labels(self):
                return ["background", "foreground"]
            
            @property
            def data_files(self):
                return {"vol": "test.h5", "seg": "test.h5"}
            
            def _prepare_data(self):
                return []
        
        with pytest.raises(ValueError, match="split must be"):
            MinimalDataset(root_dir=".", split="invalid")
    
    def test_valid_splits_accepted(self):
        """Test that valid splits are accepted."""
        
        class MinimalDataset(BaseConnectomicsDataset):
            @property
            def paper(self):
                return "Test"
            
            @property
            def resolution(self):
                return {"x": 1.0, "y": 1.0, "z": 1.0}
            
            @property
            def labels(self):
                return ["bg", "fg"]
            
            @property
            def data_files(self):
                return {"vol": "test.h5"}
            
            def _prepare_data(self):
                return [{"image": np.zeros((10, 10)), "label": np.zeros((10, 10))}]
        
        for split in ["train", "valid", "test"]:
            dataset = MinimalDataset(root_dir=".", split=split, cache_rate=0.0)
            assert dataset.split == split
    
    def test_required_properties(self):
        """Test that subclasses must implement required properties."""
        
        # Missing paper property
        class MissingPaper(BaseConnectomicsDataset):
            @property
            def resolution(self):
                return {"x": 1.0, "y": 1.0, "z": 1.0}
            
            @property
            def labels(self):
                return ["bg"]
            
            @property
            def data_files(self):
                return {}
            
            def _prepare_data(self):
                return []
        
        with pytest.raises(TypeError):
            MissingPaper(root_dir=".")
    
    def test_resolution_tuple_method(self):
        """Test get_resolution_tuple method."""
        
        class TestDataset(BaseConnectomicsDataset):
            @property
            def paper(self):
                return "Test"
            
            @property
            def resolution(self):
                return {"x": 4.0, "y": 4.0, "z": 40.0}
            
            @property
            def labels(self):
                return ["bg"]
            
            @property
            def data_files(self):
                return {}
            
            def _prepare_data(self):
                return [{"image": np.zeros((10, 10))}]
        
        dataset = TestDataset(root_dir=".", cache_rate=0.0)
        res_tuple = dataset.get_resolution_tuple()
        
        # Should return (z, y, x) order
        assert res_tuple == (40.0, 4.0, 4.0)
    
    def test_anisotropy_factor(self):
        """Test get_anisotropy_factor method."""
        
        class TestDataset(BaseConnectomicsDataset):
            @property
            def paper(self):
                return "Test"
            
            @property
            def resolution(self):
                return {"x": 6.0, "y": 6.0, "z": 30.0}
            
            @property
            def labels(self):
                return ["bg"]
            
            @property
            def data_files(self):
                return {}
            
            def _prepare_data(self):
                return [{"image": np.zeros((10, 10))}]
        
        dataset = TestDataset(root_dir=".", cache_rate=0.0)
        anisotropy = dataset.get_anisotropy_factor()
        
        # z / min(x, y) = 30 / 6 = 5.0
        assert anisotropy == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

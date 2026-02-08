"""
Tests for dataset classes.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from neurocircuitry.datasets.base import BaseConnectomicsDataset
from neurocircuitry.datasets.snemi3d import SNEMI3DDataset
from neurocircuitry.datasets.cremi3d import CREMI3DDataset
from neurocircuitry.datasets.microns import MICRONSDataset


class TestBaseConnectomicsDataset:
    """Tests for BaseConnectomicsDataset abstract class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            BaseConnectomicsDataset(root_dir=".")
    
    def test_invalid_split(self):
        """Test that invalid split raises error."""
        
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
                return {"vol": "test.h5", "seg": "test.h5"}
            
            def _prepare_data(self):
                return []
        
        with pytest.raises(ValueError, match="split must be"):
            MinimalDataset(root_dir=".", split="invalid")


class TestSNEMI3DDataset:
    """Tests for SNEMI3DDataset."""
    
    def test_properties(self):
        """Test that required properties are defined correctly."""
        # Check class-level properties
        assert SNEMI3DDataset._paper is not None
        assert "Kasthuri" in SNEMI3DDataset._paper
        
        assert SNEMI3DDataset._resolution == {"x": 6.0, "y": 6.0, "z": 30.0}
        assert SNEMI3DDataset._labels == ["background", "neuron"]
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_data_loading(self, mock_exists, mock_load):
        """Test data loading with mocked file system."""
        mock_exists.return_value = True
        
        # Create mock data
        mock_data = np.random.randint(0, 255, (100, 64, 64), dtype=np.uint8)
        mock_load.return_value = mock_data
        
        # Test train split
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="train",
            train_val_split=0.2,
        )
        
        # Should have 80 samples (80% of 100 slices)
        assert len(dataset) == 80
    
    def test_resolution_helpers(self):
        """Test resolution helper methods."""
        with patch.object(SNEMI3DDataset, '_load_volume') as mock_load:
            with patch('pathlib.Path.exists', return_value=True):
                mock_load.return_value = np.zeros((10, 64, 64))
                
                dataset = SNEMI3DDataset(root_dir="/fake/path", split="train")
                
                # Test resolution tuple (z, y, x order)
                res_tuple = dataset.get_resolution_tuple()
                assert res_tuple == (30.0, 6.0, 6.0)
                
                # Test anisotropy factor
                anisotropy = dataset.get_anisotropy_factor()
                assert anisotropy == 30.0 / 6.0


class TestCREMI3DDataset:
    """Tests for CREMI3DDataset."""
    
    def test_properties(self):
        """Test that required properties are defined correctly."""
        assert CREMI3DDataset._paper is not None
        assert "CREMI" in CREMI3DDataset._paper
        
        assert CREMI3DDataset._resolution == {"x": 4.0, "y": 4.0, "z": 40.0}
    
    def test_sample_validation(self):
        """Test that invalid samples raise error."""
        with pytest.raises(ValueError, match="Invalid sample"):
            CREMI3DDataset(root_dir=".", samples=["X"])
    
    def test_labels_with_synapses(self):
        """Test that labels change based on include_synapses."""
        with patch.object(CREMI3DDataset, '_load_sample') as mock_load:
            with patch('pathlib.Path.exists', return_value=True):
                mock_load.return_value = {
                    "raw": np.zeros((10, 64, 64)),
                    "neuron_ids": np.zeros((10, 64, 64)),
                    "clefts": np.zeros((10, 64, 64)),
                }
                
                # With synapses
                dataset = CREMI3DDataset(
                    root_dir="/fake",
                    samples=["A"],
                    include_synapses=True,
                )
                assert "synapse_cleft" in dataset.labels
                
                # Without synapses
                dataset = CREMI3DDataset(
                    root_dir="/fake",
                    samples=["A"],
                    include_synapses=False,
                )
                assert "synapse_cleft" not in dataset.labels


class TestMICRONSDataset:
    """Tests for MICRONSDataset."""
    
    def test_properties(self):
        """Test that required properties are defined correctly."""
        assert MICRONSDataset._paper is not None
        assert "MICrONS" in MICRONSDataset._paper
        
        assert MICRONSDataset._resolution == {"x": 4.0, "y": 4.0, "z": 40.0}
    
    def test_patch_generation(self):
        """Test 3D patch index generation."""
        with patch.object(MICRONSDataset, '_load_volume') as mock_load:
            with patch('pathlib.Path.exists', return_value=True):
                mock_load.return_value = np.zeros((20, 128, 128))
                
                dataset = MICRONSDataset(
                    root_dir="/fake",
                    split="train",
                    slice_mode=False,
                    patch_size=(10, 64, 64),
                    patch_overlap=0.5,
                )
                
                # Should generate multiple patches
                assert len(dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for SNEMI3D Dataset.

This module contains both mock-based tests (for CI/CD) and real data tests
(skipped if data is not available).
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from neurocircuitry.datasets.snemi3d import SNEMI3DDataset


# =============================================================================
# Mock-based Tests (no real data required)
# =============================================================================

class TestSNEMI3DDatasetMock:
    """Tests for SNEMI3DDataset using mocked data."""
    
    def test_class_properties(self):
        """Test that class-level properties are defined correctly."""
        assert SNEMI3DDataset._paper is not None
        assert "Kasthuri" in SNEMI3DDataset._paper
        assert "2015" in SNEMI3DDataset._paper
        
        assert SNEMI3DDataset._resolution == {"x": 6.0, "y": 6.0, "z": 30.0}
        assert SNEMI3DDataset._labels == ["background", "neuron"]
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_train_split_loading(self, mock_exists, mock_load):
        """Test loading train split with mocked file system."""
        mock_exists.return_value = True
        
        # Create mock data (100 slices)
        mock_data = np.random.randint(0, 255, (100, 64, 64), dtype=np.uint8)
        mock_load.return_value = mock_data
        
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="train",
            train_val_split=0.2,
            cache_rate=0.0,
        )
        
        # Should have 80 samples (80% of 100 slices)
        assert len(dataset) == 80
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_valid_split_loading(self, mock_exists, mock_load):
        """Test loading valid split with mocked file system."""
        mock_exists.return_value = True
        
        mock_data = np.random.randint(0, 255, (100, 64, 64), dtype=np.uint8)
        mock_load.return_value = mock_data
        
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="valid",
            train_val_split=0.2,
            cache_rate=0.0,
        )
        
        # Should have 20 samples (20% of 100 slices)
        assert len(dataset) == 20
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_test_split_loading(self, mock_exists, mock_load):
        """Test loading test split with mocked file system."""
        mock_exists.return_value = True
        
        mock_data = np.random.randint(0, 255, (100, 64, 64), dtype=np.uint8)
        mock_load.return_value = mock_data
        
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="test",
            cache_rate=0.0,
        )
        
        # Should have all 100 slices for test
        assert len(dataset) == 100
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_full_training_no_validation(self, mock_exists, mock_load):
        """Test train_val_split=1.0 uses all data for training."""
        mock_exists.return_value = True
        
        mock_data = np.random.randint(0, 255, (100, 64, 64), dtype=np.uint8)
        mock_load.return_value = mock_data
        
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="train",
            train_val_split=0.0,  # No validation split
            cache_rate=0.0,
        )
        
        # Should have all 100 slices for training
        assert len(dataset) == 100
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_sample_keys(self, mock_exists, mock_load):
        """Test that samples have required keys."""
        mock_exists.return_value = True
        
        mock_inputs = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
        mock_labels = np.random.randint(0, 100, (10, 64, 64), dtype=np.uint32)
        mock_load.side_effect = [mock_inputs, mock_labels]
        
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="train",
            train_val_split=0.0,
            cache_rate=0.0,
        )
        
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert "slice_idx" in sample
        assert "volume" in sample
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_sample_shapes(self, mock_exists, mock_load):
        """Test that sample shapes are correct."""
        mock_exists.return_value = True
        
        height, width = 128, 128
        mock_inputs = np.random.randint(0, 255, (10, height, width), dtype=np.uint8)
        mock_labels = np.random.randint(0, 100, (10, height, width), dtype=np.uint32)
        mock_load.side_effect = [mock_inputs, mock_labels]
        
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="train",
            train_val_split=0.0,
            cache_rate=0.0,
        )
        
        sample = dataset[0]
        assert sample["image"].shape == (height, width)
        assert sample["label"].shape == (height, width)
    
    @patch.object(SNEMI3DDataset, '_load_volume')
    @patch('pathlib.Path.exists')
    def test_3d_volume_mode(self, mock_exists, mock_load):
        """Test slice_mode=False returns 3D volume."""
        mock_exists.return_value = True
        
        depth, height, width = 20, 64, 64
        mock_inputs = np.random.randint(0, 255, (depth, height, width), dtype=np.uint8)
        mock_labels = np.random.randint(0, 100, (depth, height, width), dtype=np.uint32)
        mock_load.side_effect = [mock_inputs, mock_labels]
        
        dataset = SNEMI3DDataset(
            root_dir="/fake/path",
            split="train",
            train_val_split=0.0,
            slice_mode=False,
            cache_rate=0.0,
        )
        
        # Should have 1 sample (the full volume)
        assert len(dataset) == 1
        
        sample = dataset[0]
        assert sample["image"].shape == (depth, height, width)
        assert sample["label"].shape == (depth, height, width)
    
    def test_resolution_helpers(self):
        """Test resolution helper methods."""
        with patch.object(SNEMI3DDataset, '_load_volume') as mock_load:
            with patch('pathlib.Path.exists', return_value=True):
                mock_load.return_value = np.zeros((10, 64, 64))
                
                dataset = SNEMI3DDataset(
                    root_dir="/fake/path",
                    split="train",
                    cache_rate=0.0,
                )
                
                # Test resolution tuple (z, y, x order)
                res_tuple = dataset.get_resolution_tuple()
                assert res_tuple == (30.0, 6.0, 6.0)
                
                # Test anisotropy factor
                anisotropy = dataset.get_anisotropy_factor()
                assert anisotropy == 30.0 / 6.0  # 5.0
    
    def test_data_files_train(self):
        """Test data_files property for train split."""
        with patch.object(SNEMI3DDataset, '_load_volume') as mock_load:
            with patch('pathlib.Path.exists', return_value=True):
                mock_load.return_value = np.zeros((10, 64, 64))
                
                dataset = SNEMI3DDataset(
                    root_dir="/fake/path",
                    split="train",
                    cache_rate=0.0,
                )
                
                files = dataset.data_files
                assert files["vol"] == "AC4_inputs"
                assert files["seg"] == "AC4_labels"
    
    def test_data_files_test(self):
        """Test data_files property for test split."""
        with patch.object(SNEMI3DDataset, '_load_volume') as mock_load:
            with patch('pathlib.Path.exists', return_value=True):
                mock_load.return_value = np.zeros((10, 64, 64))
                
                dataset = SNEMI3DDataset(
                    root_dir="/fake/path",
                    split="test",
                    cache_rate=0.0,
                )
                
                files = dataset.data_files
                assert files["vol"] == "AC3_inputs"
                assert files["seg"] == "AC3_labels"
    
    def test_invalid_split_raises_error(self):
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be"):
            with patch.object(SNEMI3DDataset, '_load_volume') as mock_load:
                with patch('pathlib.Path.exists', return_value=True):
                    mock_load.return_value = np.zeros((10, 64, 64))
                    SNEMI3DDataset(root_dir="/fake/path", split="invalid")


# =============================================================================
# Real Data Tests (skipped if data not available)
# =============================================================================

# Path to actual SNEMI3D data
SNEMI3D_DATA_PATH = Path(__file__).parent.parent / "data" / "SNEMI3D"


@pytest.mark.skipif(
    not SNEMI3D_DATA_PATH.exists(),
    reason=f"SNEMI3D data not found at {SNEMI3D_DATA_PATH}"
)
class TestSNEMI3DDatasetReal:
    """Tests for SNEMI3DDataset with actual data from ../data/SNEMI3D."""
    
    def test_dataset_loads_train(self):
        """Test that train dataset loads successfully."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=0.2,
            cache_rate=0.0,
        )
        assert len(dataset) > 0
        print(f"Train dataset loaded with {len(dataset)} samples")
    
    def test_dataset_loads_valid(self):
        """Test that valid dataset loads successfully."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="valid",
            train_val_split=0.2,
            cache_rate=0.0,
        )
        assert len(dataset) > 0
        print(f"Valid dataset loaded with {len(dataset)} samples")
    
    def test_dataset_loads_test(self):
        """Test that test dataset loads successfully."""
        try:
            dataset = SNEMI3DDataset(
                root_dir=str(SNEMI3D_DATA_PATH),
                split="test",
                cache_rate=0.0,
            )
            print(f"Test dataset loaded with {len(dataset)} samples")
        except FileNotFoundError:
            pytest.skip("AC3 test data not available")
    
    def test_sample_content(self):
        """Test that samples have correct content."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=0.2,
            cache_rate=0.0,
        )
        
        sample = dataset[0]
        
        # Check keys
        assert "image" in sample
        assert "label" in sample
        assert "slice_idx" in sample
        assert "volume" in sample
        
        # Check shapes
        image = sample["image"]
        label = sample["label"]
        assert image.shape == label.shape
        assert len(image.shape) == 2  # 2D slice
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
        print(f"Label dtype: {label.dtype}, unique labels: {len(np.unique(label))}")
    
    def test_train_val_split_ratio(self):
        """Test train/validation split ratio is correct."""
        split_ratio = 0.2
        
        train_dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=split_ratio,
            cache_rate=0.0,
        )
        
        val_dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="valid",
            train_val_split=split_ratio,
            cache_rate=0.0,
        )
        
        total = len(train_dataset) + len(val_dataset)
        actual_train_ratio = len(train_dataset) / total
        expected_train_ratio = 1.0 - split_ratio
        
        assert abs(actual_train_ratio - expected_train_ratio) < 0.05
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"Actual train ratio: {actual_train_ratio:.2f}")
    
    def test_no_validation_split(self):
        """Test train_val_split=0 uses all data for training."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=0.0,
            cache_rate=0.0,
        )
        
        # Should have all slices (typically 100 for AC4)
        print(f"Full training dataset: {len(dataset)} samples")
        assert len(dataset) >= 80  # At least 80 slices expected
    
    def test_3d_volume_mode(self):
        """Test loading as 3D volume instead of slices."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=0.0,
            slice_mode=False,
            cache_rate=0.0,
        )
        
        assert len(dataset) == 1  # Single volume
        
        sample = dataset[0]
        assert len(sample["image"].shape) == 3
        print(f"3D volume shape: {sample['image'].shape}")
    
    def test_data_consistency(self):
        """Test that image and label data are consistent."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=0.2,
            cache_rate=0.0,
        )
        
        # Check multiple samples
        for i in [0, len(dataset) // 2, len(dataset) - 1]:
            sample = dataset[i]
            image = sample["image"]
            label = sample["label"]
            
            # Image and label should have same spatial dimensions
            assert image.shape == label.shape
            
            # Image should be in valid range
            assert image.min() >= 0
            
            # Label should be non-negative integers
            assert label.min() >= 0
            assert label.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, 
                                   np.int8, np.int16, np.int32, np.int64]
    
    def test_slice_indices_sequential(self):
        """Test that slice indices are sequential."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=0.2,
            cache_rate=0.0,
        )
        
        indices = [dataset[i]["slice_idx"] for i in range(min(10, len(dataset)))]
        
        # Indices should be sequential
        for i in range(1, len(indices)):
            assert indices[i] == indices[i-1] + 1
    
    def test_volume_name_correct(self):
        """Test that volume name is set correctly."""
        train_dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            cache_rate=0.0,
        )
        assert "AC4" in train_dataset[0]["volume"]
        
        val_dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="valid",
            cache_rate=0.0,
        )
        assert "AC4" in val_dataset[0]["volume"]


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.skipif(
    not SNEMI3D_DATA_PATH.exists(),
    reason=f"SNEMI3D data not found at {SNEMI3D_DATA_PATH}"
)
class TestSNEMI3DDatasetParametrized:
    """Parametrized tests for different configurations."""
    
    @pytest.mark.parametrize("train_val_split", [0.1, 0.2, 0.3])
    def test_different_split_ratios(self, train_val_split):
        """Test different train/validation split ratios."""
        train_dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="train",
            train_val_split=train_val_split,
            cache_rate=0.0,
        )
        
        val_dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split="valid",
            train_val_split=train_val_split,
            cache_rate=0.0,
        )
        
        total = len(train_dataset) + len(val_dataset)
        actual_val_ratio = len(val_dataset) / total
        
        assert abs(actual_val_ratio - train_val_split) < 0.05
        print(f"Split {train_val_split}: train={len(train_dataset)}, val={len(val_dataset)}")
    
    @pytest.mark.parametrize("split", ["train", "valid"])
    def test_all_train_valid_splits(self, split):
        """Test that train and valid splits load correctly."""
        dataset = SNEMI3DDataset(
            root_dir=str(SNEMI3D_DATA_PATH),
            split=split,
            train_val_split=0.2,
            cache_rate=0.0,
        )
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

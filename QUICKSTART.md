# NeuroCircuitry Quickstart Guide

Get up and running with NeuroCircuitry in minutes.

## Prerequisites

- Linux (tested on Ubuntu)
- NVIDIA GPU with CUDA support
- Conda (Miniconda or Anaconda)
- Python 3.12+

## Installation

### 1. Create Conda Environment

```bash
conda create -n connectomics python=3.12
conda activate connectomics
```

### 2. Install UV (Fast Package Manager)

```bash
pip install uv
```

### 3. Install PyTorch with CUDA Support

For CUDA 13.0 (adjust based on your CUDA version):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Other CUDA versions:
- CUDA 12.4: `https://download.pytorch.org/whl/cu124`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`

### 4. Verify PyTorch Installation

```bash
python -c "
import torch; 
print(f'PyTorch version: {torch.__version__}'); 
print(f'CUDA available: {torch.cuda.is_available()}'); 
print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); 
print(f'Device 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}');
"
```

Expected output:
```
PyTorch version: 2.10.0+cu130
CUDA available: True
Device count: 8
Device 0: NVIDIA B200
```

### 5. Install NeuroCircuitry Dependencies

```bash
uv pip install -r requirements.txt
```

This installs all required packages including:
- `pytorch-lightning` - Training framework
- `monai` - Medical imaging transforms and models
- `einops` - Tensor operations
- `hydra-core` / `omegaconf` - Configuration management
- `h5py`, `tifffile`, `pynrrd` - Data format support
- `wandb`, `tensorboard` - Experiment tracking

### 6. Install NeuroCircuitry Package (Development Mode)

```bash
pip install -e .
```

## Quick Verification

```python
# Test imports
from neurocircuitry.datasets import SNEMI3DDataset, CREMI3DDataset, MICRONSDataset
from neurocircuitry.preprocessors import TIFFPreprocessor, HDF5Preprocessor
from neurocircuitry.modules import SemanticSegmentationModule, InstanceSegmentationModule

print("NeuroCircuitry installed successfully!")
```

## Download Sample Data

### SNEMI3D
```bash
mkdir -p data/snemi3d
# Download from: https://snemi3d.grand-challenge.org/
```

### CREMI
```bash
mkdir -p data/cremi
# Download from: https://cremi.org/data/
wget https://cremi.org/static/data/sample_A_20160501.hdf -P data/cremi/
wget https://cremi.org/static/data/sample_B_20160501.hdf -P data/cremi/
wget https://cremi.org/static/data/sample_C_20160501.hdf -P data/cremi/
```

## Run Training

### Basic Training (SNEMI3D)

```bash
python scripts/train.py --config configs/snemi3d.yaml
```

### Training with Overrides

```bash
python scripts/train.py \
    --config configs/snemi3d.yaml \
    --data.batch_size 4 \
    --training.max_epochs 100 \
    --training.precision 16-mixed
```

### Multi-GPU Training

```bash
python scripts/train.py \
    --config configs/snemi3d.yaml \
    --training.devices 4 \
    --training.strategy ddp
```

## Explore Data (Jupyter Notebooks)

```bash
pip install jupyter
cd notebooks
jupyter notebook
```

Open `01_snemi3d_eda.ipynb` to visualize dataset slices and label distributions.

## Project Structure

```
NeuroCircuitry/
├── neurocircuitry/
│   ├── datasets/       # SNEMI3D, CREMI3D, MICRONS dataset classes
│   ├── datamodules/    # PyTorch Lightning DataModules
│   ├── modules/        # Lightning training modules
│   ├── models/         # Vista3D, SegResNet wrappers
│   ├── losses/         # Discriminative, boundary losses
│   ├── preprocessors/  # TIFF, HDF5, NRRD loaders
│   ├── transforms/     # EM-specific augmentations
│   └── utils/          # I/O utilities
├── configs/            # Hydra/OmegaConf YAML configs
├── scripts/            # Training entry points
├── notebooks/          # EDA Jupyter notebooks
└── tests/              # Unit tests
```

## Next Steps

1. **Explore EDA notebooks** - Understand your data
2. **Customize configs** - Adjust for your dataset
3. **Train models** - Start with semantic segmentation
4. **Evaluate** - Use built-in metrics (Dice, IoU)

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Enable mixed precision: `training.precision: 16-mixed`
- Reduce `patch_size` or `cache_rate`

### Slow Data Loading
- Increase `num_workers` in config
- Increase `cache_rate` (uses more RAM)
- Use SSD storage for data

### Import Errors
- Ensure conda environment is activated: `conda activate connectomics`
- Reinstall package: `pip install -e .`

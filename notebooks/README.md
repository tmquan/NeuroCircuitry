# NeuroCircuitry Notebooks

This folder contains Jupyter notebooks for exploratory data analysis (EDA) of connectomics datasets.

## Notebooks

| Notebook | Dataset | Description |
|----------|---------|-------------|
| `01_snemi3d_eda.ipynb` | SNEMI3D | Neuron segmentation in mouse cortex EM |
| `02_cremi3d_eda.ipynb` | CREMI | Neuron + synapse segmentation in Drosophila |
| `03_microns_eda.ipynb` | MICRONS | Large-scale cortical connectomics |

## Features

Each notebook provides:

1. **Dataset Metadata** - Paper reference, resolution, label information
2. **Volume Loading** - Support for HDF5, TIFF, NRRD formats
3. **Statistics** - Shape, dtype, min/max, memory usage
4. **Slice Visualization** - Multiple slices across the volume
5. **Segmentation Visualization** - Color-coded instance labels
6. **Side-by-Side Comparison** - EM image vs segmentation overlay
7. **Label Analysis** - Instance count, size distribution, coverage
8. **Per-Slice Analysis** - Instances per slice plots
9. **NeuroCircuitry Integration** - Using the dataset classes

## Setup

1. Update `DATA_ROOT` in each notebook to point to your data directory
2. Install dependencies:
   ```bash
   pip install jupyter matplotlib numpy h5py tifffile pynrrd
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Expected Data Structure

### SNEMI3D
```
data/snemi3d/
├── AC4_inputs.h5    # Training volume
├── AC4_labels.h5    # Training labels
├── AC3_inputs.h5    # Test volume
└── AC3_labels.h5    # Test labels (if available)
```

### CREMI
```
data/cremi/
├── sample_A_20160501.hdf
├── sample_B_20160501.hdf
└── sample_C_20160501.hdf
```

### MICRONS
```
data/microns/
├── volume.h5         # EM volume
├── segmentation.h5   # Neuron labels
├── synapses.h5       # Synapse annotations (optional)
└── mitochondria.h5   # Mitochondria labels (optional)
```

## Data Sources

- **SNEMI3D**: https://snemi3d.grand-challenge.org/
- **CREMI**: https://cremi.org/
- **MICRONS**: https://www.microns-explorer.org/ (via CAVEclient API)

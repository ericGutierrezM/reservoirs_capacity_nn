# Predicting Water Volume in Reservoirs using Satellite Imagery in Catalonia

#### Computational Learning and Deep Learning. Barcelona School of Economics.
> Eric Gutiérrez, 23rd  March 2026.

This repository contains the end-to-end pipeline for predicting hydrological volumes from Sentinel-2 MSI data using a custom Convolutional Neural Network.

## 1. Environment Setup
This project uses uv for high-performance dependency management and reproducible environments.

### 1.1 Dependencies
If you have `uv` installed, run:

```Bash
# Sync dependencies and create a locked environment
uv sync
```

### 1.2 CDSE API Credentials
To download satellite imagery, you must have an account with the Copernicus Data Space Ecosystem.

Create a folder named confidential/ in the root directory.

Create two text files: SH_CLIENT_ID.txt and SH_CLIENT_SECRET.txt.

Paste your credentials into these files. The notebooks are configured to read these to avoid hardcoding secrets.

## 2. Pipeline
### 2.1. Data Acquisition (_get_sentinel-2.ipynb_)
This notebook handles the interaction with the CDSE API. It defines the Bounding Box (BBOX) for the 9 reservoirs, filters for <15% cloud cover, and executes a server-side Evalscript to compute NDWI.

> Note: Downloading the full 2016–2026 dataset (approx. 2000+ TIFFs) requires ~2.5GB of disk space.

### 2.2. Label Matching (_get_labels.ipynb_)
This script performs a temporal join between the downloaded `*.tiff` filenames and the `capacity_reservoirs.csv` provided by the ACA. It outputs `master_labels_volume.csv`, which serves as the ground truth for the PyTorch Dataset class.

### 2.3. Training via LOOCV (_cnn_loocv.ipynb_)
This is the core training script. Set LOSS_MODE = 'mape', 'log_mse', or 'mse' to toggle optimization strategies. The script automatically performs Leave-One-Reservoir-Out Cross-Validation.

> Hardware Note: Training 500x500 tensors with a batch size of 32 requires at least 8GB of VRAM. If running on a local CPU, reduce BATCH_SIZE to 4 or 8.

### 2.4. Ensemble Inference (_prediction_ensemble.ipynb_)
Once the 9 models are trained (or downloaded from the models/ folder), this notebook loads them as an Ensemble. It processes "unseen" reservoirs (Escales/Terradets), and applies the Savitzky-Golay filter to the timeline to remove spikes.

> Caution: Ensure the loss_mode parameter matches the weights being loaded to avoid scaling mismatches during the inverse transform.

### 2.5. Visualizations (_plots.ipynb_)
This script generates a plot for each of the "unseen" reservoirs (Escales and Terradets), and displays both the raw and smoothed predictions for the models that use Log-MSE and MAPE.
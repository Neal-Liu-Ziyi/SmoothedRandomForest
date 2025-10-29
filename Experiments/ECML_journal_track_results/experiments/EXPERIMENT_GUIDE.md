# SRFnet_OOB Experiment Guide

## 📋 Overview

The `exp_SRFnet_CV_experiments.py` script runs comprehensive cross-validation experiments to evaluate SRFnet_OOB model selection across multiple datasets.

## 🎯 What It Does

1. **Cross-Validation Model Selection**
   - Tests 4 smoothing modes: `global`, `per_dim`, `per_tree`, `per_tree_dim`
   - Tests 2 kernels: `normal`, `hyperbolic_secant`
   - Selects best model based on validation MSE

2. **Comprehensive Evaluation**
   - Evaluates ALL smoothing modes on test set (not just the best one)
   - Compares against baseline RF and full RF

3. **Saves Everything**
   - Base RF models (10 trees for CV)
   - Full RF models (100 trees for comparison)
   - All predictions and calibrated predictions
   - Smoothing parameters for each mode
   - CV selection results

## 📁 Output Structure

```
results/
└── <data_name>/
    ├── base_rf/              # RF models with 10 trees
    │   └── <data_name>_n50_r0_rf10.joblib
    ├── rf100/                # RF models with 100 trees
    │   └── <data_name>_n50_r0_rf100.joblib
    ├── predictions/          # All predictions
    │   └── <data_name>_n50_r0.csv
    ├── cv_results/           # CV selection info
    │   └── <data_name>_n50_r0.json
    └── smoothing_params/     # Learned parameters
        └── <data_name>_n50_r0.json
```

## 🚀 Quick Start

### Basic Usage

```python
from exp_SRFnet_CV_experiments import run_experiments

# Run on a single dataset
run_experiments(
    data_name_list=['pendulum'],
    n_obs_list=[50, 100],
    r_list=range(5),
    epochs=100,
    n_jobs=4
)
```

### Run All Datasets

```bash
python exp_SRFnet_CV_experiments.py
```

This will run experiments on all 14 datasets with default settings.

## ⚙️ Configuration

### Default Settings

```python
DATA_NAMES = [
    'autompg', 'breastcancer', 'compressive', 'facebook',
    'fertility', 'forest', 'housing', 'machine', 'pendulum',
    'qsar_aquatic_toxicity', 'servo', 'slump', 'stock', 
    'yacht_hydrodynamics'
]

N_OBS_LIST = [50, 100, 200]  # Sample sizes
R_LIST = range(20)            # 20 replications
EPOCHS = 100                  # Training epochs
N_JOBS = 4                    # Parallel jobs
```

### Customization

Modify the `if __name__ == "__main__"` section:

```python
# Test on single dataset quickly
DATA_NAMES = ['pendulum']
N_OBS_LIST = [50]
R_LIST = range(5)
EPOCHS = 50
N_JOBS = 2
```

## 📊 Output Files

### 1. Predictions (`predictions/<data_name>_n<n_obs>_r<r>.csv`)

Contains all predictions:

| Column | Description |
|--------|-------------|
| `y_test` | True test labels |
| `base_rf_pred` | Base RF (10 trees) predictions |
| `rf_full_pred` | Full RF (100 trees) predictions |
| `srf_normal_global_pred` | SRF with normal kernel, global mode |
| `srf_normal_per_dim_pred` | SRF with normal kernel, per_dim mode |
| `srf_normal_per_tree_pred` | SRF with normal kernel, per_tree mode |
| `srf_normal_per_tree_dim_pred` | SRF with normal kernel, per_tree_dim mode |
| `srf_hypsec_global_pred` | SRF with hypsec kernel, global mode |
| ... | (8 SRF variants total) |

### 2. CV Results (`cv_results/<data_name>_n<n_obs>_r<r>.json`)

```json
{
    "data_name": "pendulum",
    "n_obs": 50,
    "r": 0,
    "normal": {
        "best_mode": "per_dim",
        "cv_results": {
            "global": {"valid_mse": 10.5, "smoothing_params": 0.8, ...},
            "per_dim": {"valid_mse": 9.2, "smoothing_params": [0.7, 1.1, ...], ...},
            "per_tree": {"valid_mse": 9.8, ...},
            "per_tree_dim": {"valid_mse": 9.5, ...}
        }
    },
    "hyperbolic_secant": {
        "best_mode": "global",
        "cv_results": {...}
    }
}
```

### 3. Smoothing Parameters (`smoothing_params/<data_name>_n<n_obs>_r<r>.json`)

```json
{
    "normal": {
        "global": 0.85,
        "per_dim": [0.7, 1.2, 0.9, ...],
        "per_tree": [0.8, 0.9, 1.1, ...],
        "per_tree_dim": [[0.7, 0.8, ...], [1.1, 0.9, ...], ...]
    },
    "hyperbolic_secant": {
        ...
    }
}
```

## 🔍 Dataset Size Handling

The script automatically handles datasets with limited samples:

```python
DATASET_MAX_N_OBS = {
    'fertility': 100,    # Will skip n_obs=200
    'machine': 209,      # Will skip n_obs > 209
    'servo': 167,        # Will skip n_obs=200
    'slump': 103,        # Will skip n_obs > 103
    ...
}
```

## 💻 Parallel Processing

The script uses `joblib.Parallel` for efficiency:

```python
# Adjust based on your CPU cores
N_JOBS = 4  # Run 4 experiments simultaneously
```

**Memory Usage:**
- Each job runs independently
- Memory is cleaned up after each experiment
- Safe for large-scale experiments

## 📈 Analyzing Results

### Load Predictions

```python
import pandas as pd

# Load predictions for a specific experiment
df = pd.read_csv('srf_cv_results/pendulum/predictions/pendulum_n50_r0.csv')

# Calculate MSE for each method
from sklearn.metrics import mean_squared_error

mse_base_rf = mean_squared_error(df['y_test'], df['base_rf_pred'])
mse_srf_normal_global = mean_squared_error(df['y_test'], df['srf_normal_global_pred'])
...
```

### Aggregate Results

```python
import glob
import json

# Collect all CV results for a dataset
cv_files = glob.glob('srf_cv_results/pendulum/cv_results/*.json')

best_modes_normal = []
best_modes_hypsec = []

for file in cv_files:
    with open(file, 'r') as f:
        data = json.load(f)
        best_modes_normal.append(data['normal']['best_mode'])
        best_modes_hypsec.append(data['hyperbolic_secant']['best_mode'])

# Count frequency of each mode being selected
from collections import Counter
print("Normal kernel best modes:", Counter(best_modes_normal))
print("Hypsec kernel best modes:", Counter(best_modes_hypsec))
```

## ⏱️ Estimated Runtime

For default settings (14 datasets, 3 sample sizes, 20 reps):
- Total experiments: ~840
- Time per experiment: 2-5 minutes
- Total time (4 parallel jobs): **~7-18 hours**

For quick testing:
```python
DATA_NAMES = ['pendulum']
N_OBS_LIST = [50]
R_LIST = range(5)
# Total: 5 experiments, ~10-25 minutes
```

## 🐛 Troubleshooting

### Issue: "No data for dataset"
- Check that bootstrap sample files exist
- Verify n_obs is within valid range

### Issue: Memory error
- Reduce `N_JOBS`
- Process datasets sequentially

### Issue: Slow convergence
- Reduce `EPOCHS` for testing
- Some datasets may need more epochs for convergence

## 📝 Notes

1. **Reproducibility**: Uses fixed `random_state=42` for RF models
2. **Calibration**: All SRF predictions are calibrated using OOB linear regression
3. **Skip Completed**: Automatically skips experiments that already have results
4. **Error Handling**: Failed experiments are logged but don't stop the pipeline

---

**Created:** 2025-10-20  
**Version:** 1.0


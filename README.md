# Smoothed Random Forest (SRFnet)

A PyTorch implementation of Smoothed Random Forest with kernel smoothing and gradient-based optimization.

## Overview

SRFnet is an enhanced random forest model that applies kernel smoothing to tree predictions, enabling:
- **Differentiable predictions** for gradient-based optimization
- **Multiple smoothing strategies** (global, per-dimension, per-tree, per-tree-dimension)
- **Multiple kernel functions** (Normal, Hyperbolic Secant)
- **Out-of-Bag (OOB) evaluation** for robust model selection
- **Improved prediction accuracy** through optimized smoothing parameters

## Key Features

### Model Components (`models/`)
- **`SRFnet_OOB.py`**: Main SRFnet model with OOB-based optimization
- **`RandomForestRegressor.py`**: Extended sklearn RandomForest with OOB methods
- **`Hypsecant.py`**: Hyperbolic Secant kernel implementation
- **`TreeInfoExtractor.py`**: Utilities for extracting tree structure information
- **`optimizer_configs.py`**: Pre-configured optimizers and schedulers

### Smoothing Modes
- **Global**: Single smoothing parameter for all features and trees
- **Per-dimension**: One parameter per feature
- **Per-tree**: One parameter per tree
- **Per-tree-dimension**: One parameter per tree-feature combination

### Kernel Functions
- **Normal (Gaussian)**: Standard Gaussian kernel
- **Hyperbolic Secant**: Heavy-tailed alternative with robust properties

## Installation

```bash
# Clone the repository
git clone https://github.com/Neal-Liu-Ziyi/SmoothedRandomForest.git
cd SmoothedRandomForest

# Install dependencies
pip install numpy scipy scikit-learn torch pandas joblib tensorboard
```

## Quick Start

There are two ways to use SRFnet depending on how much control you need.

---

### Method 1 — Paper Pipeline (step by step)

This follows the exact procedure used in the paper experiments
(`exp_EST_PD_noCV.py`): train an RF10, optimise smoothing parameters,
apply OOB linear calibration, and produce point predictions.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

from models.RandomForestRegressor import ExtendedRandomForest
from models.SRFnet_OOB import SRFnetOOB
from models.optimizer_configs import get_optimizer_scheduler

# ── Step 1: Train RF10 ────────────────────────────────────────────────
rf = ExtendedRandomForest(n_estimators=10, bootstrap=True)
rf.fit(X_train, y_train)

# ── Step 2: Optimise SRFnet smoothing parameters ──────────────────────
#   Default settings recommended by the paper:
#     smoothing_mode = 'per_tree_dim'
#     srf_kernel     = 'hyperbolic_secant'
model = SRFnetOOB(
    smoothing_mode='per_tree_dim',
    srf_kernel='hyperbolic_secant',
    init_smoothing=0.5,
)
opt_fn, sch_fn = get_optimizer_scheduler('per_tree_dim', total_epochs=100)
model.fit(
    X_train, y_train,
    rf=rf,
    epochs=100,
    optimizer=opt_fn,
    scheduler=sch_fn,
    verbose=False,
)

# ── Step 3: OOB linear calibration ───────────────────────────────────
#   Use per-tree smoothed predictions on the training set to fit a
#   linear calibrator on OOB samples only.
smoothed_trees_pred = model.smoothed_trees_predict(X_train)  # (n_train, n_trees)
smoothed_T = smoothed_trees_pred.T                           # (n_trees, n_train)

oob_mask = np.zeros((len(rf.oob_samples_indices), X_train.shape[0]), dtype=bool)
for i, idx_list in enumerate(rf.oob_samples_indices):
    oob_mask[i, idx_list] = True

oob_preds = smoothed_T[oob_mask]
oob_y     = np.broadcast_to(y_train, smoothed_T.shape)[oob_mask]

calibrator = LinearRegression(fit_intercept=True, copy_X=True)
calibrator.fit(oob_preds.reshape(-1, 1), oob_y.reshape(-1, 1))

# ── Step 4: Prediction (point estimation) ────────────────────────────
srf_pred   = model.predict(X_test)
predictions = calibrator.predict(srf_pred.reshape(-1, 1)).flatten()
```

---

### Method 2 — High-level Wrapper (`SRFnetPredictor`)

`SRFnetPredictor` wraps all four steps above into a single class.
Defaults match the paper's recommended settings.

```python
from models.SRFnet_predictor import SRFnetPredictor

model = SRFnetPredictor()          # per_tree_dim + hyperbolic_secant + RF10
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

You can also pass a pre-trained RF:

```python
from models.RandomForestRegressor import ExtendedRandomForest

rf = ExtendedRandomForest(n_estimators=10, bootstrap=True)
rf.fit(X_train, y_train)

model = SRFnetPredictor()
model.fit(X_train, y_train, rf=rf)
predictions = model.predict(X_test)
```

## Experiments

The `Experiments/` folder contains:
- **`data/`**: Benchmark datasets used in experiments
- **`ECML_journal_track_results/`**: Complete experimental setup
  - `experiments/`: Experimental scripts
  - `analysis/`: Analysis notebooks
  - `results/`: Experimental results (predictions and CV results)

### Running Experiments

```bash
cd Experiments/ECML_journal_track_results/experiments

# Run per_tree_dim mode (loads pre-trained RF10, outputs calibrated predictions + uncertainty)
python exp_EST_PD_noCV.py --data_name winequality-red --n_obs_list 100 200 --n_jobs 5

# Run global/per_dim/per_tree modes (loads pre-trained RF10, outputs calibrated predictions + uncertainty)
python exp_other_modes_noCV.py --data_name winequality-red --n_obs_list 100 200 --n_jobs 5

# Add RF baselines (RF20, RF50)
python add_rf20_rf50_optimized.py --data_name winequality-red

# Add Gaussian Process baseline (Matern kernel, appends gp_pred and gp_std columns)
python add_gp.py --data_name winequality-red --n_obs_list 100 200 --n_jobs 5
```

All experiment scripts share the same CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_name` | all datasets | Dataset name(s) to run |
| `--n_obs_list` | 50 100 200 300 400 500 | Training sample sizes |
| `--r0` / `--r1` | 0 / 100 | Replication index range |
| `--epochs` | 100 | Training epochs |
| `--n_jobs` | 5 | Parallel jobs |
| `--save_dir` | `../results` | Output directory |

**Output structure for `exp_EST_PD_noCV.py` and `exp_other_modes_noCV.py`:**
```
results/<data_name>/
    EST_PD_results_noCV/          # JSON files with smoothing params & calibration coefficients
    EST_PD_predictions_noCV/      # CSV files with predictions and uncertainty columns:
                                  #   srf_{kernel}_EST_PD_pred
                                  #   srf_{kernel}_EST_PD_total_std
                                  #   srf_{kernel}_EST_PD_noise_free_std
                                  #   srf_{kernel}_EST_PD_{intra,inter,model}_var
    other_mode_results_noCV/      # JSON files for global/per_dim/per_tree modes
    other_mode_predictions_noCV/  # CSV files for global/per_dim/per_tree modes
```

## Citation

This work is currently under review. If you use this code in your research, please cite:

```bibtex
@article{liu2025improving,
  title={Improving Random Forests by Smoothing},
  author={Liu, Ziyi and Luong, Phuc and Boley, Mario and Schmidt, Daniel F.},
  note={Manuscript submitted to ECML PKDD Journal Track},
  year={2026}
}
```

**Paper Status:** Under review at ECML PKDD Journal Track

## License

MIT License

## Contact

For questions and feedback, please open an issue on GitHub.


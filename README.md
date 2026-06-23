<p align="center">
  <img src="smoorf_logo.png" alt="SmooRF logo" width="300" />
</p>

# Smoothed Random Forest

A PyTorch implementation of Smoothed Random Forest with kernel smoothing and gradient-based optimization.

## Overview

SRFnet is an enhanced random forest model that applies kernel smoothing to tree predictions, enabling:
- **Differentiable predictions** for gradient-based optimization
- **Multiple smoothing strategies** — `STE`, `STE_PD`, `EST`, `EST_PD` (legacy names `global` / `per_dim` / `per_tree` / `per_tree_dim` are still accepted)
- **Multiple kernel functions** (Normal, Hyperbolic Secant)
- **Out-of-Bag (OOB) evaluation** for robust model selection
- **Improved prediction accuracy** through optimized smoothing parameters

## Key Features

### Model Components (`models/`)
- **`SRFnet_OOB.py`**: Main SRFnet model with OOB-based optimization
- **`SRFRegressor.py`**: High-level scikit-learn-style wrapper (RF + SRFnet + OOB calibration) — recommended entry point
- **`SRFinference.py`**: Inference-only sibling of `SRFnet_OOB` — numpy + scipy, no pytorch; takes a pre-trained bandwidth and skips all optimisation. See Method 3 below.
- **`RandomForestRegressor.py`**: Extended sklearn RandomForest with OOB methods
- **`Hypsecant.py`**: Hyperbolic Secant kernel implementation
- **`TreeInfoExtractor.py`**: Utilities for extracting tree structure information
- **`optimizer_configs.py`**: Pre-configured optimizers and schedulers

### Smoothing Modes
Paper names are used in the API; legacy names from earlier code are still accepted as aliases.

| Paper name | Legacy name | Description |
|---|---|---|
| `STE`     | `global`        | Single smoothing parameter for all features and trees |
| `STE_PD`  | `per_dim`       | One parameter per feature |
| `EST`     | `per_tree`      | One parameter per tree |
| `EST_PD`  | `per_tree_dim`  | One parameter per tree-feature combination (paper default) |

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
#     smoothing_mode = 'EST_PD'   (legacy alias: 'per_tree_dim')
#     srf_kernel     = 'hyperbolic_secant'
model = SRFnetOOB(
    smoothing_mode='EST_PD',
    srf_kernel='hyperbolic_secant',
    init_smoothing=0.5,
)
opt_fn, sch_fn = get_optimizer_scheduler('EST_PD', total_epochs=100)
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

### Method 2 — High-level Wrapper (`SRFRegressor`)

`SRFRegressor` wraps all four steps above into a single class.
Defaults match the paper's recommended settings.

```python
from models.SRFRegressor import SRFRegressor

model = SRFRegressor()             # EST_PD + hyperbolic_secant + RF10
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Already have a `sklearn.ensemble.RandomForestRegressor`? Use
`ExtendedRandomForest.upgrade(...)` to convert it in place — it keeps the
same trained trees and just attaches the OOB machinery SRFnet needs:

```python
from sklearn.ensemble import RandomForestRegressor
from models.RandomForestRegressor import ExtendedRandomForest
from models.SRFRegressor import SRFRegressor

# Existing sklearn RF you already trained somewhere
rf = RandomForestRegressor(n_estimators=10, bootstrap=True)
rf.fit(X_train, y_train)

# Upgrade to SRF-compatible RF (same trees, +OOB indices for SRFnet)
rf = ExtendedRandomForest.upgrade(rf, X_train, y_train)

# Pass it straight to SRFRegressor — SRF training will reuse the trees
model = SRFRegressor()
model.fit(X_train, y_train, rf=rf)
predictions = model.predict(X_test)
```

---

### Method 3 — Lightweight Inference (`SRFInference`)

Pickling a fully-trained `SRFRegressor` / `SRFnetOOB` is heavy: the
pytorch state, optimiser history, tree-info cache, and OOB bookkeeping
all get serialised. If you only need predictions / effective-kernel
matrices / variance decomposition at *inference time*, you can save just
**two** small artefacts and rebuild the predictor on the fly:

1. the trained `ExtendedRandomForest` (`joblib.dump(rf, ...)`)
2. the smoothing-bandwidth array (`np.savez(..., bandwidth=model._srf.get_smoothing_params())`)
3. *(optional)* the OOB linear calibration coefficients
   (`model.calibration_coef_`, `model.calibration_intercept_`)

`SRFInference` is a pure numpy + scipy implementation that reproduces
the same effective kernel and variance decomposition as `SRFnet_OOB` to
~1e-6 (the only gap is the `+1e-6` softplus pad inside SRFnetOOB). It
does **not** train smoothing parameters and does **not** fit the OOB
calibration — both must be supplied externally.

**Save side (training machine):**

```python
import numpy as np
from joblib import dump
from models.SRFRegressor import SRFRegressor

model = SRFRegressor().fit(X_train, y_train)

# Three small artefacts:
dump(model._rf, 'rf.joblib')                              # tens of KB
np.savez('bandwidth.npz',
         bandwidth = model._srf.get_smoothing_params())   # < 1 KB
np.savez('calibration.npz',
         coef      = model.calibration_coef_,
         intercept = model.calibration_intercept_)        # < 1 KB
```

**Load side (inference machine):**

```python
import numpy as np
from joblib import load
from models.SRFinference import SRFInference
from models.Hypsecant import HyperbolicSecant

rf        = load('rf.joblib')                            # ExtendedRandomForest
bandwidth = np.load('bandwidth.npz')['bandwidth']
calib     = np.load('calibration.npz')
coef, intercept = float(calib['coef']), float(calib['intercept'])

inf = SRFInference(
    smoothing_params=bandwidth,
    kernel=HyperbolicSecant,         # or scipy.stats.norm for 'normal'
    per_tree=True,                    # True for EST / EST_PD; False for STE / STE_PD
)
inf.fit(X_train, y_train, rf=rf, coef=coef, intercept=intercept)

# Calibrated predictions + uncertainty (matches SRFnet_OOB exactly)
pred, total_std, noise_free_std = inf.predict(
    X_test,
    return_uncertainty=True,
    return_noise_free_uncertainty=True,
    calibrate=True,
)

# Per-test-point variance decomposition (uncalibrated, matches SRFnet_OOB.get_detailed_uncertainty)
intra_var, inter_var, model_var = inf.get_detailed_uncertainty(X_test)

# Effective kernel matrix (n_test, n_train) — handy for downstream EL / conformal CIs
K_test = inf.get_effective_kernel(X_test)
```

`per_tree` should match how the bandwidth was trained:

| Smoothing mode | `per_tree` |
|---|---|
| `STE`, `STE_PD` | `False` |
| `EST`, `EST_PD` | `True` |

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

# Run EST_PD mode (loads pre-trained RF10, outputs calibrated predictions + uncertainty)
python exp_EST_PD_noCV.py --data_name winequality-red --n_obs_list 100 200 --n_jobs 5

# Run STE / STE_PD / EST modes (loads pre-trained RF10, outputs calibrated predictions + uncertainty)
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
    other_mode_results_noCV/      # JSON files for STE / STE_PD / EST modes
                                  # (CSV/JSON column names use legacy keys global/per_dim/per_tree)
    other_mode_predictions_noCV/  # CSV files for STE / STE_PD / EST modes
```

## Citation

This work has been published in *Machine Learning* (Springer Nature, 2026).
If you use this code in your research, please cite:

```bibtex
@article{liu2026improving,
  title     = {Improving Random Forests by Smoothing},
  author    = {Liu, Ziyi and Luong, Phuc and Boley, Mario and Schmidt, Daniel F.},
  journal   = {Machine Learning},
  volume    = {115},
  number    = {152},
  year      = {2026},
  publisher = {Springer Nature},
  doi       = {10.1007/s10994-026-07077-z},
  url       = {https://link.springer.com/article/10.1007/s10994-026-07077-z}
}
```

**Paper:** [Liu et al. (2026), *Machine Learning* 115:152](https://link.springer.com/article/10.1007/s10994-026-07077-z)

## License

MIT License

## Contact

For questions and feedback, please open an issue on GitHub.


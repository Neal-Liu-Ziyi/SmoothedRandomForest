****# Smoothed Random Forest (SRFnet)

A PyTorch implementation of Smoothed Random Forest with gradient-based optimization and uncertainty quantification.

## Overview

SRFnet is an enhanced random forest model that applies kernel smoothing to tree predictions, enabling:
- **Differentiable predictions** for gradient-based optimization
- **Uncertainty quantification** with detailed decomposition
- **Multiple smoothing strategies** (global, per-dimension, per-tree, per-tree-dimension)
- **Multiple kernel functions** (Normal, Hyperbolic Secant)
- **Out-of-Bag (OOB) evaluation** for robust model selection

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
git clone https://github.com/YOUR_USERNAME/SmoothedRandomForest.git
cd SmoothedRandomForest

# Install dependencies
pip install numpy scipy scikit-learn torch pandas joblib tensorboard

# Or use conda environment
conda env create -f environment.yml
conda activate srf-env
```

## Quick Start

```python
from models.SRFnet_OOB import SRFnetOOB
from models.optimizer_configs import get_optimizer_scheduler

# Prepare data
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data

# Create model
model = SRFnetOOB(
    smoothing_mode='per_tree_dim',
    srf_kernel='hyperbolic_secant',
    init_smoothing=0.5
)

# Configure optimizer and scheduler
opt_fn, sch_fn = get_optimizer_scheduler('per_tree_dim', total_epochs=100)

# Train model
model.fit(
    X_train, y_train,
    epochs=100,
    optimizer=opt_fn,
    scheduler=sch_fn,
    verbose=True
)

# Make predictions with uncertainty
predictions, total_uncertainty, noise_free_uncertainty = model.predict(
    X_test,
    return_uncertainty=True,
    return_noise_free_uncertainty=True
)
```

## Uncertainty Quantification

SRFnet provides detailed uncertainty decomposition:

```python
intra_var, inter_var, model_var = model.get_detailed_uncertainty(X_test)
```

- **Intra-tree variance**: Average uncertainty within each tree (smoothing-induced)
- **Inter-tree variance**: Disagreement between different trees (ensemble diversity)
- **Model variance**: Irreducible error from training set MSE

Total uncertainty = √(intra_var + inter_var + model_var)

## Experiments

The `Experiments/` folder contains:
- **`data/`**: Benchmark datasets used in experiments
- **`ECML_journal_track_results/`**: Complete experimental setup
  - `experiments/`: Experimental scripts
  - `analysis/`: Analysis notebooks
  - `results/`: Experimental results (predictions and CV results)

### Running Experiments

```bash
# Run SRFnet experiments
cd Experiments/ECML_journal_track_results/experiments
python exp_SRFnet_CV_experiments.py --data_name pendulum --n_obs_list 50 100 200

# Add RF baselines
python add_rf20_rf50_optimized.py --data_name pendulum
```

## Model Selection

SRFnet uses cross-validation to automatically select the best smoothing mode:

```python
# CV results are saved in results/{dataset}/cv_results/
# Contains validation MSE for all 4 smoothing modes
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{smoothed_random_forest,
  title={Smoothed Random Forest with Uncertainty Quantification},
  author={Your Name},
  journal={ECML Journal Track},
  year={2024}
}
```

## License

MIT License

## Contact

For questions and feedback, please open an issue on GitHub.


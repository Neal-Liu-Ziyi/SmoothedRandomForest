"""
SRFnet_OOB Other Modes Experiments (No CV Model Selection)

This script runs SRFnet experiments using pre-trained RF10 models with
global, per_dim, and per_tree smoothing modes (excluding per_tree_dim).

Key Features:
- Loads pre-trained RF10 models from previous experiments
- Runs global, per_dim, per_tree smoothing modes (no CV model selection)
- Support for 2 kernel types (normal, hyperbolic_secant)
- Parallel processing for efficiency

Directory Structure:
    results/
        <data_name>/
            rf10/                       # Pre-trained RF10 models (loaded)
            other_mode_predictions_noCV/  # Predictions for other modes
            other_mode_results_noCV/      # Results for other modes

Usage:
    python exp_other_modes_noCV.py
"""

import pandas as pd
import os
import sys
import numpy as np
import json
import gc
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from joblib import load, Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('../../../../')

from smoorf.RandomForestRegressor import ExtendedRandomForest
from smoorf.SRFnet_OOB import SRFnetOOB
from optimizer_configs import get_optimizer_scheduler
from exp_wrapper import get_bootstrap_samples


# Dataset configurations: maximum available n_obs for each dataset
DATASET_MAX_N_OBS = {
    'fertility': 100,  # Small dataset
    'forest': 500,
    'qsar_aquatic_toxicity': 500,
    'stock': 500,
    'yacht_hydrodynamics': 300,  # Medium dataset
    'real_estate': 414,
    'winequality-red': 1599,
    'winequality-white': 4898,  # Large dataset
    'qsar_fish_toxicity': 908,
    'Combined_Cycle_Power_Plant': 9568,
}

# Smoothing modes to run (excluding per_tree_dim)
SMOOTHING_MODES = ['global', 'per_dim', 'per_tree']


def get_valid_n_obs_list(data_name, requested_n_obs_list):
    """Filter n_obs_list based on dataset's maximum available samples"""
    max_n_obs = DATASET_MAX_N_OBS.get(data_name, 500)
    valid_list = [n for n in requested_n_obs_list if n <= max_n_obs]
    return valid_list


def real_data(data_name, n_obs, r):
    """Load and prepare data for a single experiment"""
    data = pd.read_csv(f'../../data/{data_name}.csv')
    bootstrap_sample_df = pd.read_csv('../../data/bootstrap_sample.csv')

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, y_train, X_test, y_test = get_bootstrap_samples(
        bootstrap_df=bootstrap_sample_df,
        file_name=data_name,
        q=n_obs,
        r=r,
        X=X,
        y=y
    )

    return X_train, y_train, X_test, y_test


def run_single_smoothing_mode(X_train, y_train, X_test, rf, smoothing_mode, srf_kernel, epochs=150):
    """
    Run SRFnet_OOB for a single smoothing mode

    Returns:
        dict with predictions, uncertainties, smoothing_params, calibration coefficients
    """
    model = SRFnetOOB(smoothing_mode=smoothing_mode, srf_kernel=srf_kernel, init_smoothing=0.5)

    opt_fn, sch_fn = get_optimizer_scheduler(smoothing_mode, total_epochs=epochs)

    model.fit(X_train, y_train, rf=rf,
              epochs=epochs,
              optimizer=opt_fn,
              scheduler=sch_fn,
              verbose=False)

    # Get predictions with both types of uncertainty
    srf_pred, total_uncertainty, noise_free_uncertainty = model.predict(
        X_test,
        return_uncertainty=True,
        return_noise_free_uncertainty=True
    )

    # Get detailed uncertainty decomposition
    intra_var, inter_var, model_var = model.get_detailed_uncertainty(X_test)

    # Get individual tree predictions for calibration
    smoothed_trees_train_pred = model.smoothed_trees_predict(X_train)

    # OOB Linear Regression calibration
    smoothed_trees_train_pred_T = smoothed_trees_train_pred.T
    oob_mask = np.zeros((len(rf.oob_samples_indices), X_train.shape[0]), dtype=bool)
    for i, idx_list in enumerate(rf.oob_samples_indices):
        oob_mask[i, idx_list] = True
    oob_preds = smoothed_trees_train_pred_T[oob_mask]
    oob_y = np.broadcast_to(y_train, smoothed_trees_train_pred_T.shape)[oob_mask]

    lr = LinearRegression(fit_intercept=True, copy_X=True)
    lr.fit(oob_preds.reshape(-1, 1), oob_y.reshape(-1, 1))
    srf_pred_calibrated = lr.predict(srf_pred.reshape(-1, 1))

    # Get calibration coefficients
    coef = lr.coef_.flatten()[0]
    intercept = lr.intercept_.flatten()[0]

    # Calibrate uncertainty components
    intra_var_calibrated = (coef ** 2) * intra_var
    inter_var_calibrated = (coef ** 2) * inter_var

    # For model variance, compute from calibrated training predictions
    train_pred_uncalibrated = smoothed_trees_train_pred.mean(axis=1)
    train_pred_calibrated = coef * train_pred_uncalibrated + intercept
    model_var_calibrated = mean_squared_error(y_train, train_pred_calibrated)

    # Calibrate noise-free uncertainty
    noise_free_uncertainty_calibrated = np.abs(coef) * noise_free_uncertainty

    # Recalculate total uncertainty
    total_variance_calibrated = intra_var_calibrated + inter_var_calibrated + model_var_calibrated
    total_uncertainty_calibrated = np.sqrt(total_variance_calibrated)

    # Get smoothing parameters
    smoothing_params = model.get_smoothing_params()

    return {
        'pred': srf_pred,
        'pred_calibrated': srf_pred_calibrated.flatten(),
        'total_uncertainty': total_uncertainty,
        'noise_free_uncertainty': noise_free_uncertainty,
        'total_uncertainty_calibrated': total_uncertainty_calibrated,
        'noise_free_uncertainty_calibrated': noise_free_uncertainty_calibrated,
        'intra_var': intra_var,
        'inter_var': inter_var,
        'model_var': model_var,
        'intra_var_calibrated': intra_var_calibrated,
        'inter_var_calibrated': inter_var_calibrated,
        'model_var_calibrated': model_var_calibrated,
        'smoothing_params': smoothing_params,
        'coef': lr.coef_.flatten(),
        'intercept': lr.intercept_.flatten(),
    }


def run_single_experiment(data_name, n_obs, r, save_dir, epochs=150):
    """
    Run a single experiment for one (data_name, n_obs, r) combination

    Loads pre-trained RF10 and runs global, per_dim, per_tree with both kernels.
    """
    # Check if already completed
    pred_file = f'{save_dir}/{data_name}/other_mode_predictions_noCV/{data_name}_n{n_obs}_r{r}.csv'
    if os.path.exists(pred_file):
        return None

    # Check if RF10 model exists
    rf10_path = f'{save_dir}/{data_name}/rf10/{data_name}_n{n_obs}_r{r}_rf10.joblib'
    if not os.path.exists(rf10_path):
        raise FileNotFoundError(f"RF10 model not found: {rf10_path}")

    # Load data
    X_train, y_train, X_test, y_test = real_data(data_name, n_obs, r)

    if X_train is None:
        raise ValueError(f"No data for {data_name} n={n_obs} r={r}")

    # Load pre-trained RF10
    rf10 = load(rf10_path)

    # Convert to ExtendedRandomForest for SRFnet
    rf10_extended = ExtendedRandomForest.upgrade(rf10, X_train, y_train)

    # Create prediction DataFrame
    pred_df = pd.DataFrame({
        'y_test': y_test,
    })

    # Store results for JSON
    exp_info = {
        'data_name': data_name,
        'n_obs': n_obs,
        'r': r,
        'smoothing_modes': SMOOTHING_MODES,
    }

    # Run all smoothing modes with both kernels
    for smoothing_mode in SMOOTHING_MODES:
        result_normal = run_single_smoothing_mode(
            X_train, y_train, X_test, rf10_extended,
            smoothing_mode, 'normal', epochs
        )

        result_hypsec = run_single_smoothing_mode(
            X_train, y_train, X_test, rf10_extended,
            smoothing_mode, 'hyperbolic_secant', epochs
        )

        # Add predictions and uncertainties for both kernels
        for kernel_name, result in [('normal', result_normal), ('hypsec', result_hypsec)]:
            col_prefix = f'srf_{kernel_name}_{smoothing_mode}'

            # Predictions
            pred_df[f'{col_prefix}_pred'] = result['pred_calibrated']

            # Uncertainties (calibrated)
            pred_df[f'{col_prefix}_total_std'] = result['total_uncertainty_calibrated']
            pred_df[f'{col_prefix}_noise_free_std'] = result['noise_free_uncertainty_calibrated']

            # Variance components (calibrated)
            pred_df[f'{col_prefix}_intra_var'] = result['intra_var_calibrated']
            pred_df[f'{col_prefix}_inter_var'] = result['inter_var_calibrated']
            pred_df[f'{col_prefix}_model_var'] = result['model_var_calibrated']

        # Store info for JSON (per smoothing_mode)
        exp_info[smoothing_mode] = {
            'normal': {
                'smoothing_params_shape': list(result_normal['smoothing_params'].shape),
                'coef': result_normal['coef'].tolist(),
                'intercept': result_normal['intercept'].tolist()
            },
            'hyperbolic_secant': {
                'smoothing_params_shape': list(result_hypsec['smoothing_params'].shape),
                'coef': result_hypsec['coef'].tolist(),
                'intercept': result_hypsec['intercept'].tolist()
            }
        }

    # Save predictions
    pred_df.to_csv(pred_file, index=False)

    # Save experiment info
    json_file = f'{save_dir}/{data_name}/other_mode_results_noCV/{data_name}_n{n_obs}_r{r}.json'
    with open(json_file, 'w') as f:
        json.dump(exp_info, f, indent=4)

    # Clean up
    gc.collect()

    return True


def run_experiments(data_name_list,
                   n_obs_list=[50, 100, 200, 300, 400, 500],
                   r_list=range(100),
                   save_dir='../results',
                   epochs=100,
                   n_jobs=5):
    """
    Main function to run all experiments
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create directory structure for each dataset
    for data_name in data_name_list:
        data_dir = f'{save_dir}/{data_name}'
        os.makedirs(f'{data_dir}/other_mode_predictions_noCV', exist_ok=True)
        os.makedirs(f'{data_dir}/other_mode_results_noCV', exist_ok=True)

    # Create task list
    tasks = []
    for data_name in data_name_list:
        # Filter n_obs_list based on dataset size
        valid_n_obs = get_valid_n_obs_list(data_name, n_obs_list)

        for n_obs in valid_n_obs:
            for r in r_list:
                tasks.append((data_name, n_obs, r))

    # Run tasks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_single_experiment)(data_name, n_obs, r, save_dir, epochs)
        for data_name, n_obs, r in tasks
    )

    completed = sum(1 for r in results if r is not None)


def main():
    """Main function with command-line argument support"""
    import argparse

    parser = argparse.ArgumentParser(description='Run SRFnet other modes experiments (no CV)')

    # Dataset arguments
    parser.add_argument('--data_name', type=str, nargs='+', default=None,
                       help='Dataset name(s) to run. (default: run all datasets)')

    # Sample size arguments
    parser.add_argument('--n_obs_list', type=int, nargs='+', default=[50, 100, 200, 300, 400, 500],
                       help='List of sample sizes to test')

    # Replication arguments
    parser.add_argument('--r0', type=int, default=0, help='Start replication index')
    parser.add_argument('--r1', type=int, default=100, help='End replication index')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--n_jobs', type=int, default=5, help='Number of parallel jobs')

    # Output arguments
    parser.add_argument('--save_dir', type=str, default='../results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Determine which datasets to run
    if args.data_name is not None:
        data_names = args.data_name if isinstance(args.data_name, list) else [args.data_name]
    else:
        # Default: all datasets
        data_names = [
            'fertility', 'forest',
            'qsar_aquatic_toxicity', 'stock', 'yacht_hydrodynamics',
            'Combined_Cycle_Power_Plant',
            'winequality-red',
            'winequality-white',
            'real_estate',
            'qsar_fish_toxicity',
        ]

    r_list = range(args.r0, args.r1)

    # Run experiments
    run_experiments(
        data_name_list=data_names,
        n_obs_list=args.n_obs_list,
        r_list=r_list,
        save_dir=args.save_dir,
        epochs=args.epochs,
        n_jobs=args.n_jobs
    )

    print("\n" + "=" * 80)
    print("Experiments completed!")
    print(f"Predictions saved to: {args.save_dir}/<data_name>/other_mode_predictions_noCV/")
    print(f"Results saved to: {args.save_dir}/<data_name>/other_mode_results_noCV/")
    print("=" * 80)


if __name__ == "__main__":
    main()

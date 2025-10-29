"""
SRFnet_OOB Cross-Validation Model Selection Experiments

This script runs comprehensive experiments to evaluate SRFnet_OOB model selection
using cross-validation across multiple datasets, sample sizes, and bootstrap replications.

Key Features:
- Automated CV-based model selection across 4 smoothing modes (global, per_dim, per_tree, per_tree_dim)
- Support for 2 kernel types (normal, hyperbolic_secant)
- Parallel processing for efficiency
- Comprehensive result storage (models, predictions, smoothing parameters)
- Handles datasets with varying sample sizes
- Controlled comparison: rf10 uses same hyperparameters as best_rf (only differs in training data)

Directory Structure:
    results/
        <data_name>/
            base_rf/        # Small RF (10 trees, 80% data, CV-tuned max_depth) for SRFnet training
            rf10/           # RF (10 trees, 100% data, SAME max_depth as base_rf) for fair comparison
            rf100/          # Full RF (100 trees, 100% data, independently tuned) as strong baseline
            predictions/    # All predictions and uncertainties
            cv_results/     # CV selection results

Usage:
    python exp_SRFnet_CV_experiments.py
"""

import pandas as pd
import os
import sys
import numpy as np
import json
import gc
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from joblib import dump, load, Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('../../../')

from models.Hypsecant import HyperbolicSecant
from models.RandomForestRegressor import ExtendedRandomForest
from models.SRFnet_OOB import SRFnetOOB
from optimizer_configs import get_optimizer_scheduler
from exp_wrapper import get_bootstrap_samples


# Dataset configurations: maximum available n_obs for each dataset
DATASET_MAX_N_OBS = {
    'fertility': 100,  # Small dataset
    'forest': 500,
    'qsar_aquatic_toxicity': 500,
    'stock': 500,
    'yacht_hydrodynamics': 300 , # Medium dataset
    'real_estate': 414,
    'winequality-red': 1599,
    'winequality-white': 4898, # Large dataset
    'qsar_fish_toxicity': 908,
    'Combined_Cycle_Power_Plant': 9568,
}

def get_valid_n_obs_list(data_name, requested_n_obs_list):
    """
    Filter n_obs_list based on dataset's maximum available samples
    
    Args:
        data_name: Name of the dataset
        requested_n_obs_list: List of requested sample sizes
        
    Returns:
        Filtered list of valid sample sizes
    """
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


def split_train_valid(X, y, valid_ratio=0.2, random_state=42):
    """Split data into training and validation sets"""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=valid_ratio, random_state=random_state
    )
    return X_train, X_valid, y_train, y_valid


def get_rf_predictions_with_std(rf, X):
    """
    Get Random Forest predictions with standard deviation
    
    Args:
        rf: Trained RandomForestRegressor
        X: Test data
        
    Returns:
        pred: Mean predictions
        std: Standard deviation across trees
    """
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X) for tree in rf.estimators_])
    
    # Calculate mean and std
    pred = tree_predictions.mean(axis=0)
    std = tree_predictions.std(axis=0)
    
    return pred, std


def optimize_rf_max_depth(X, y, max_depth_range=range(3, 21), 
                          n_estimators=10, cv=5, n_jobs=5, max_features=0.8,
                          scoring='neg_mean_squared_error', **rf_kwargs):
    """Optimize Random Forest max_depth using cross-validation"""
    cv_results = []
    best_score = -np.inf
    best_max_depth = None
    
    for max_depth in max_depth_range:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=42,
            **rf_kwargs
        )
        
        scores = cross_val_score(rf, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        mean_score = scores.mean()
        std_score = scores.std()
        
        cv_results.append((max_depth, mean_score, std_score))
        
        if mean_score > best_score:
            best_score = mean_score
            best_max_depth = max_depth
    
    # Train the best model on the full training data
    best_rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=best_max_depth,
        n_jobs=n_jobs,
        random_state=42,
        max_features=max_features,
        **rf_kwargs
    )
    best_rf.fit(X, y)
    
    return {
        'best_max_depth': best_max_depth,
        'best_score': best_score,
        'best_rf': best_rf,
        'cv_results': cv_results
    }


def run_single_smoothing_mode(X_train, y_train, X_test, rf, smoothing_mode, srf_kernel, epochs=150):
    """
    Run SRFnet_OOB for a single smoothing mode
    
    Returns:
        dict with predictions, uncertainties, smoothing_params, model, calibration coefficients
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
    
    # Calibrate uncertainty components (variance scales by coef²)
    # For intra and inter variance, scaling is correct because:
    # Var(aX + b) = a² × Var(X)
    intra_var_calibrated = (coef ** 2) * intra_var
    inter_var_calibrated = (coef ** 2) * inter_var
    
    # For model variance, we need to compute it from calibrated training predictions
    # NOT by scaling, because MSE(aX + b, Y) ≠ a² × MSE(X, Y)
    train_pred_uncalibrated = smoothed_trees_train_pred.mean(axis=1)
    train_pred_calibrated = coef * train_pred_uncalibrated + intercept
    model_var_calibrated = mean_squared_error(y_train, train_pred_calibrated)
    
    # Calibrate noise-free uncertainty
    # For variance components (intra, inter), Var(aX+b) = a² × Var(X)
    # So noise_free_std² = intra_cal + inter_cal = coef² × (intra + inter)
    # Therefore noise_free_std_cal = |coef| × noise_free_std
    noise_free_uncertainty_calibrated = np.abs(coef) * noise_free_uncertainty
    
    # For total uncertainty, we need to recalculate because model_var changed
    # total_var_cal = intra_cal + inter_cal + model_var_cal
    # Note: model_var_cal is NOT coef² × model_var, it's recalculated above
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
        'model': model
    }


def SRF_model_selection_CV(X_train, y_train, X_valid, y_valid, X_test, rf, srf_kernel, epochs=100):
    """
    Run all 4 smoothing modes on validation set and select the best one
    Then evaluate on test set
    
    Returns:
        dict with CV results, selected model info, and test predictions for all modes
    """
    smoothing_modes = ['global', 'per_dim', 'per_tree', 'per_tree_dim']
    
    cv_results = {}
    test_results = {}
    
    # Run all modes on validation set
    for mode in smoothing_modes:
        try:
            # Step 1: Run on validation set (for model selection)
            result_valid = run_single_smoothing_mode(X_train, y_train, X_valid, rf, mode, srf_kernel, epochs)
            
            # Calculate validation MSE
            valid_mse = mean_squared_error(y_valid, result_valid['pred_calibrated'])
            
            cv_results[mode] = {
                'valid_mse': valid_mse,
                'smoothing_params': result_valid['smoothing_params'].tolist() if isinstance(result_valid['smoothing_params'], np.ndarray) else float(result_valid['smoothing_params']),
                'coef': result_valid['coef'].tolist() if isinstance(result_valid['coef'], np.ndarray) else float(result_valid['coef']),
                'intercept': result_valid['intercept'].tolist() if isinstance(result_valid['intercept'], np.ndarray) else float(result_valid['intercept'])
            }
            
            # Step 2: Use the same trained model to predict on test set
            trained_model = result_valid['model']
            coef = result_valid['coef'][0] if isinstance(result_valid['coef'], np.ndarray) else result_valid['coef']
            intercept = result_valid['intercept'][0] if isinstance(result_valid['intercept'], np.ndarray) else result_valid['intercept']
            
            # Get test predictions with uncertainty
            test_pred, test_total_unc, test_nf_unc = trained_model.predict(
                X_test,
                return_uncertainty=True,
                return_noise_free_uncertainty=True
            )
            
            # Get test uncertainty decomposition
            test_intra_var, test_inter_var, test_model_var = trained_model.get_detailed_uncertainty(X_test)
            
            # Calibrate test predictions
            test_pred_calibrated = coef * test_pred + intercept
            
            # Calibrate test uncertainty components
            test_intra_var_cal = (coef ** 2) * test_intra_var
            test_inter_var_cal = (coef ** 2) * test_inter_var
            
            # Recalculate calibrated model variance (same as training, it's a scalar)
            # We already computed this in result_valid
            test_model_var_cal = result_valid['model_var_calibrated']
            
            # Calibrate test uncertainties
            test_nf_unc_cal = np.abs(coef) * test_nf_unc
            test_total_var_cal = test_intra_var_cal + test_inter_var_cal + test_model_var_cal
            test_total_unc_cal = np.sqrt(test_total_var_cal)
            
            # Store test predictions and uncertainties
            test_results[mode] = {
                'pred': test_pred,
                'pred_calibrated': test_pred_calibrated,
                'total_uncertainty': test_total_unc,
                'noise_free_uncertainty': test_nf_unc,
                'total_uncertainty_calibrated': test_total_unc_cal,
                'noise_free_uncertainty_calibrated': test_nf_unc_cal,
                'intra_var': test_intra_var,
                'inter_var': test_inter_var,
                'model_var': test_model_var,
                'intra_var_calibrated': test_intra_var_cal,
                'inter_var_calibrated': test_inter_var_cal,
                'model_var_calibrated': test_model_var_cal,
                'smoothing_params': result_valid['smoothing_params']
            }
            
            # Clean up
            del result_valid
            del trained_model
            gc.collect()
            
        except Exception as e:
            print(f"Error in {mode}: {e}")
            cv_results[mode] = {'valid_mse': np.inf, 'error': str(e)}
            test_results[mode] = {
                'pred': None, 
                'pred_calibrated': None,
                'total_uncertainty': None,
                'noise_free_uncertainty': None,
                'total_uncertainty_calibrated': None,
                'noise_free_uncertainty_calibrated': None,
                'intra_var': None,
                'inter_var': None,
                'model_var': None,
                'intra_var_calibrated': None,
                'inter_var_calibrated': None,
                'model_var_calibrated': None
            }
    
    # Find best model based on validation MSE
    best_mode = min(cv_results.keys(), key=lambda k: cv_results[k].get('valid_mse', np.inf))
    
    return {
        'cv_results': cv_results,
        'best_mode': best_mode,
        'test_results': test_results
    }


def run_single_experiment(data_name, n_obs, r, save_dir, epochs=100):
    """
    Run a single experiment for one (data_name, n_obs, r) combination
    
    Args:
        data_name: Name of dataset
        n_obs: Number of observations
        r: Replication index
        save_dir: Base directory for saving results
        epochs: Number of training epochs
    """
    # Check if already completed
    pred_file = f'{save_dir}/{data_name}/predictions/{data_name}_n{n_obs}_r{r}.csv'
    if os.path.exists(pred_file):
        #print(f"Skipping {data_name} n={n_obs} r={r} (already exists)")
        return None
    
    try:
        # Load data
        X_train_all, y_train_all, X_test_all, y_test_all = real_data(data_name, n_obs, r)
        
        if X_train_all is None:
            print(f"No data for {data_name} n={n_obs} r={r}")
            return None
        
        # Split train into train/valid
        X_train, X_valid, y_train, y_valid = split_train_valid(X_train_all, y_train_all)
        
        # Optimize RF on training set (small RF for CV)
        best_rf_result = optimize_rf_max_depth(X_train, y_train, n_estimators=10)
        best_rf = best_rf_result['best_rf']
        best_max_depth = best_rf_result['best_max_depth']

        # Save base_rf
        dump(best_rf, f'{save_dir}/{data_name}/base_rf/{data_name}_n{n_obs}_r{r}_rf10.joblib', compress=3)
        
        # Base RF prediction (with 10 trees on 80% data) - compute BEFORE conversion
        base_rf_pred, base_rf_std = get_rf_predictions_with_std(best_rf, X_test_all)
        
        # Convert to ExtendedRandomForest for SRFnet
        best_rf_extended = ExtendedRandomForest.upgrade(best_rf, X_train, y_train)
        
        # Run model selection for both kernels
        results_normal = SRF_model_selection_CV(X_train, y_train, X_valid, y_valid, 
                                                X_test_all, best_rf_extended, 'normal', epochs)
        results_hypsec = SRF_model_selection_CV(X_train, y_train, X_valid, y_valid, 
                                                X_test_all, best_rf_extended, 'hyperbolic_secant', epochs)
        
        # Train RF10 on all training data (10 trees) using SAME hyperparameters as best_rf
        rf10 = optimize_rf_max_depth(X_train_all, y_train_all, n_estimators=10, cv=5)['best_rf']
        dump(rf10, f'{save_dir}/{data_name}/rf10/{data_name}_n{n_obs}_r{r}_rf10.joblib', compress=3)
        rf10_pred, rf10_std = get_rf_predictions_with_std(rf10, X_test_all)
        
        # Train full RF on all training data (100 trees)
        rf_full = optimize_rf_max_depth(X_train_all, y_train_all, n_estimators=100, cv=5)['best_rf']
        dump(rf_full, f'{save_dir}/{data_name}/rf100/{data_name}_n{n_obs}_r{r}_rf100.joblib', compress=3)
        rf_full_pred, rf_full_std = get_rf_predictions_with_std(rf_full, X_test_all)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'y_test': y_test_all,
            'base_rf_pred': base_rf_pred,
            'base_rf_std': base_rf_std,
            'rf10_pred': rf10_pred,
            'rf10_std': rf10_std,
            'rf_100_pred': rf_full_pred,
            'rf_100_std': rf_full_std
        })
        
        # Add all smoothing mode predictions and uncertainties for both kernels
        for kernel_name, results in [('normal', results_normal), ('hypsec', results_hypsec)]:
            for mode in ['global', 'per_dim', 'per_tree', 'per_tree_dim']:
                if results['test_results'][mode]['pred_calibrated'] is not None:
                    # Predictions
                    pred_df[f'srf_{kernel_name}_{mode}_pred'] = results['test_results'][mode]['pred_calibrated']
                    
                    # Uncertainties (calibrated)
                    # Total uncertainty (includes model variance)
                    pred_df[f'srf_{kernel_name}_{mode}_total_std'] = results['test_results'][mode]['total_uncertainty_calibrated']
                    # Noise-free uncertainty (excludes model variance)
                    pred_df[f'srf_{kernel_name}_{mode}_noise_free_std'] = results['test_results'][mode]['noise_free_uncertainty_calibrated']
                    
                    # Variance components (calibrated)
                    pred_df[f'srf_{kernel_name}_{mode}_intra_var'] = results['test_results'][mode]['intra_var_calibrated']
                    pred_df[f'srf_{kernel_name}_{mode}_inter_var'] = results['test_results'][mode]['inter_var_calibrated']
                    
                    # Model variance (scalar, same for all test points)
                    model_var_val = results['test_results'][mode]['model_var_calibrated']
                    if np.isscalar(model_var_val):
                        pred_df[f'srf_{kernel_name}_{mode}_model_var'] = model_var_val
                    else:
                        pred_df[f'srf_{kernel_name}_{mode}_model_var'] = model_var_val
        
        # Save predictions and uncertainties
        pred_df.to_csv(pred_file, index=False)
        
        # Save CV results and smoothing parameters
        cv_info = {
            'data_name': data_name,
            'n_obs': n_obs,
            'r': r,
            'normal': {
                'best_mode': results_normal['best_mode'],
                'cv_results': results_normal['cv_results']
            },
            'hyperbolic_secant': {
                'best_mode': results_hypsec['best_mode'],
                'cv_results': results_hypsec['cv_results']
            },
            'base_rf_max_depth': best_max_depth # Used by both best_rf and rf10
        }
        
        with open(f'{save_dir}/{data_name}/cv_results/{data_name}_n{n_obs}_r{r}.json', 'w') as f:
            json.dump(cv_info, f, indent=4)
        
        # Clean up
        gc.collect()
        
        #print(f"Completed {data_name} n={n_obs} r={r}")
        return True
        
    except Exception as e:
        print(f"Error in {data_name} n={n_obs} r={r}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiments(data_name_list, 
                   n_obs_list=[50, 100, 200,300,400,500],
                   r_list=range(100),
                   save_dir='../results',
                   epochs=100,
                   n_jobs=5):
    """
    Main function to run all experiments
    
    Args:
        data_name_list: List of dataset names
        n_obs_list: List of sample sizes to test
        r_list: List of replication indices
        save_dir: Directory to save results
        epochs: Number of training epochs for SRFnet
        n_jobs: Number of parallel jobs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create directory structure for each dataset
    for data_name in data_name_list:
        data_dir = f'{save_dir}/{data_name}'
        os.makedirs(f'{data_dir}/base_rf', exist_ok=True)
        os.makedirs(f'{data_dir}/rf10', exist_ok=True)
        os.makedirs(f'{data_dir}/rf100', exist_ok=True)
        os.makedirs(f'{data_dir}/predictions', exist_ok=True)
        os.makedirs(f'{data_dir}/cv_results', exist_ok=True)
    
    # Create task list
    tasks = []
    for data_name in data_name_list:
        # Filter n_obs_list based on dataset size
        valid_n_obs = get_valid_n_obs_list(data_name, n_obs_list)
        print(f"Processing dataset: {data_name} (sample sizes: {valid_n_obs})")
        
        for n_obs in valid_n_obs:
            for r in r_list:
                tasks.append((data_name, n_obs, r))
    
    #print(f"Total tasks: {len(tasks)}")
    
    # Run tasks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_single_experiment)(data_name, n_obs, r, save_dir, epochs)
        for data_name, n_obs, r in tasks
    )
    
    #print(f"\nCompleted {sum(1 for r in results if r is not None)} / {len(tasks)} tasks")
    

def main():
    """Main function with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SRFnet_OOB CV experiments')
    
    # Dataset arguments
    parser.add_argument('--data_name', type=str, nargs='+', default=None,
                       help='Dataset name(s) to run. Can be single or multiple. (default: run all datasets). Example: --data_name pendulum or --data_name pendulum autompg housing')
    
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
        # Dataset(s) specified (can be single or multiple)
        data_names = args.data_name if isinstance(args.data_name, list) else [args.data_name]
    else:
        # Default: all datasets
        data_names = [
            'fertility', 'forest', 
            'qsar_aquatic_toxicity',  'stock', 'yacht_hydrodynamics',
            'Combined_Cycle_Power_Plant',
            'winesquality-red',
            'winesquality-white',
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
    print(f"Results saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()


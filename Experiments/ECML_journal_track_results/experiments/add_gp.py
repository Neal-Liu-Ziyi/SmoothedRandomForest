"""
Add GP (Gaussian Process) to Existing Experiments

This script adds GP model to existing experiment results:
- Uses Matern kernel with optimized hyperparameters
- Trains on the same bootstrap samples as original experiments
- Saves predictions and uncertainties to existing prediction files (adds 2 columns)

Key Features:
- Trains GP model on the same bootstrap samples
- Saves predictions and uncertainties to existing prediction files (adds 2 columns)
  * GP: gp_pred, gp_std
- Saves GP models for future use

Directory Structure:
    results/
        <data_name>/
            predictions/                # Predictions file (append GP columns)
                                       # New columns: gp_pred, gp_std
            gp/                        # GP models (new)

Usage:
    python add_gp.py
    python add_gp.py --data_name fertility forest
"""

import pandas as pd
import os
import sys
import numpy as np
import gc
from sklearn.exceptions import ConvergenceWarning
from joblib import dump, Parallel, delayed
import warnings
warnings.filterwarnings('ignore')
# Suppress sklearn GP convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Add paths
sys.path.append('../../../')
sys.path.append('../../../../')

from models.GaussianProcess import GPmodel
from exp_wrapper import get_bootstrap_samples


# Dataset configurations
DATASET_MAX_N_OBS = {
    'fertility': 100,
    'forest': 500,
    'qsar_aquatic_toxicity': 500,
    'stock': 500,
    'yacht_hydrodynamics': 300,
    'real_estate': 414,
    'winequality-red': 1599,
    'winequality-white': 4898,
    'qsar_fish_toxicity': 908,
    'Combined_Cycle_Power_Plant': 9568,
}


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

def run_single_experiment(data_name, n_obs, r, save_dir):
    """
    Run GP for a single experiment
    
    Args:
        data_name: Name of dataset
        n_obs: Number of observations
        r: Replication index
        save_dir: Base directory for saving results
    """
    
    # Check if prediction file exists
    pred_file = f'{save_dir}/{data_name}/predictions/{data_name}_n{n_obs}_r{r}.csv'
    if not os.path.exists(pred_file):
        # print(f"Prediction file not found for {data_name} n={n_obs} r={r}")
        return None
    
    # Check if already processed
    pred_df = pd.read_csv(pred_file)
    if 'gp_pred' in pred_df.columns:
        # print(f"Already processed {data_name} n={n_obs} r={r}")
        return None
    
    try:
        # Load data (same bootstrap samples as original experiments)
        X_train, y_train, X_test, y_test = real_data(data_name, n_obs, r)
        
        if X_train is None:
            print(f"No data for {data_name} n={n_obs} r={r}")
            return None
        
        # Verify data consistency
        if not np.allclose(y_test, pred_df['y_test'].values):
            print(f"Data mismatch for {data_name} n={n_obs} r={r}")
            return None
        
        # ============================================================
        # Run GP Model
        # ============================================================
        gp = GPmodel()
        gp.fit(X_train, y_train, num_restarts=10)
        gp_preds, gp_stds = gp.predict(X_test, return_std=True)
        
        # Save GP model
        gp_model_file = f'{save_dir}/{data_name}/gp/{data_name}_n{n_obs}_r{r}_gp.joblib'
        dump(gp, gp_model_file, compress=3)
        
        # ============================================================
        # Update prediction DataFrame
        # ============================================================
        
        # Add GP columns
        pred_df['gp_pred'] = gp_preds
        pred_df['gp_std'] = gp_stds
        
        # Save updated predictions
        pred_df.to_csv(pred_file, index=False)
        
        # Clean up
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error in {data_name} n={n_obs} r={r}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiments(data_name_list,
                   n_obs_list=[50, 100, 200, 300, 400, 500],
                   r_list=range(100),
                   save_dir='../results',
                   n_jobs=5):
    """
    Main function to add GP to all experiments
    
    Args:
        data_name_list: List of dataset names
        n_obs_list: List of sample sizes to test
        r_list: List of replication indices
        save_dir: Directory to save results
        n_jobs: Number of parallel jobs
    """
    # Create directory structure for each dataset
    for data_name in data_name_list:
        data_dir = f'{save_dir}/{data_name}'
        os.makedirs(f'{data_dir}/gp', exist_ok=True)
    
    # Create task list
    tasks = []
    for data_name in data_name_list:
        # Filter n_obs_list based on dataset size
        valid_n_obs = get_valid_n_obs_list(data_name, n_obs_list)
        
        for n_obs in valid_n_obs:
            for r in r_list:
                # Only process if prediction file exists
                pred_file = f'{save_dir}/{data_name}/predictions/{data_name}_n{n_obs}_r{r}.csv'
                if os.path.exists(pred_file):
                    tasks.append((data_name, n_obs, r))
    
    # print(f"Total tasks to process: {len(tasks)}")
    
    # Run tasks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_single_experiment)(data_name, n_obs, r, save_dir)
        for data_name, n_obs, r in tasks
    )
    
    # print(f"\nCompleted {sum(1 for r in results if r is not None)} / {len(tasks)} tasks")


def main():
    """Main function with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add GP to experiments')
    
    parser.add_argument('--data_name', type=str, nargs='+', default=None,
                       help='Dataset name(s) to process')
    
    parser.add_argument('--n_obs_list', type=int, nargs='+', default=[50, 100, 200, 300, 400, 500],
                       help='List of sample sizes to test')
    
    parser.add_argument('--r0', type=int, default=0, help='Start replication index')
    parser.add_argument('--r1', type=int, default=100, help='End replication index')
    
    parser.add_argument('--n_jobs', type=int, default=5, help='Number of parallel jobs')
    
    parser.add_argument('--save_dir', type=str, default='../results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    if args.data_name is not None:
        data_names = args.data_name if isinstance(args.data_name, list) else [args.data_name]
    else:
        # Default: all datasets
        data_names = [
            'fertility',
            'forest',
            'qsar_aquatic_toxicity',
            'stock',
            'yacht_hydrodynamics',
            'real_estate',
            'winequality-red',
            'winequality-white',
            'qsar_fish_toxicity',
            'Combined_Cycle_Power_Plant'
        ]
    
    r_list = range(args.r0, args.r1)
    
    # Run experiments
    run_experiments(
        data_name_list=data_names,
        n_obs_list=args.n_obs_list,
        r_list=r_list,
        save_dir=args.save_dir,
        n_jobs=args.n_jobs
    )
    
    print("\n" + "=" * 80)
    print("GP experiments completed!")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()


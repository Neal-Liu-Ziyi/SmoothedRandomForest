"""
Add RF20 and RF50 with Optimized max_depth on 100% Data

This script trains RF20 and RF50 models with max_depth optimized on 100% training data.
Uses the same bootstrap samples as original experiments.

Usage:
    python add_rf20_rf50_optimized.py
    python add_rf20_rf50_optimized.py 
"""

import pandas as pd
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('../../../../')

from exp_wrapper import get_bootstrap_samples


# Dataset configurations: maximum available n_obs for each dataset
DATASET_MAX_N_OBS = {
    'autompg': 400,
    'breastcancer': 200,
    'fertility': 100,
    'forest': 500,
    'housing': 500,
    'pendulum': 500,
    'qsar_aquatic_toxicity': 500,
    'servo': 100,
    'stock': 500,
    'yacht_hydrodynamics': 300,
    'ENB2012_data_energy_heating': 768,
    'ENB2012_data_energy_cooling': 768,
    'real_estate': 414,
    'winequality-red': 1599,
    'winequality-white': 4898,
    'airfoil_self_noise': 1503,
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


def get_rf_predictions_with_std(rf, X):
    """Get Random Forest predictions with standard deviation"""
    tree_predictions = np.array([tree.predict(X) for tree in rf.estimators_])
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


def run_single_experiment(data_name, n_obs, r, save_dir):
    """
    Retrain RF10 with optimized max_depth for a single experiment
    
    Args:
        data_name: Name of dataset
        n_obs: Number of observations
        r: Replication index
        save_dir: Base directory for saving results
    """
    # Check if prediction file exists
    pred_file = f'{save_dir}/{data_name}/predictions/{data_name}_n{n_obs}_r{r}.csv'
    if not os.path.exists(pred_file):
        return None
    
    try:
        # Load the SAME data as original experiment
        X_train_all, y_train_all, X_test_all, y_test_all = real_data(data_name, n_obs, r)
        
        if X_train_all is None:
            print(f"No data for {data_name} n={n_obs} r={r}")
            return None
        
        # Read existing predictions
        pred_df = pd.read_csv(pred_file)
        
        # Verify data consistency
        if not np.allclose(y_test_all, pred_df['y_test'].values):
            print(f"Data mismatch for {data_name} n={n_obs} r={r}")
            return None
        
        # Train RF10 with optimized max_depth on 100% data
        rf20_result = optimize_rf_max_depth(X_train_all, y_train_all, n_estimators=20, cv=5)
        rf20 = rf20_result['best_rf']
        
        # Get predictions
        rf20_pred, rf20_std = get_rf_predictions_with_std(rf20, X_test_all)
        
        # Update prediction DataFrame
        pred_df['rf20_pred'] = rf20_pred
        pred_df['rf20_std'] = rf20_std
        
        # Save RF20 model
        dump(rf20, f'{save_dir}/{data_name}/rf20/{data_name}_n{n_obs}_r{r}_rf20.joblib', compress=3)

        rf50_result = optimize_rf_max_depth(X_train_all, y_train_all, n_estimators=50, cv=5)
        rf50 = rf50_result['best_rf']
        rf50_pred, rf50_std = get_rf_predictions_with_std(rf50, X_test_all)
        pred_df['rf50_pred'] = rf50_pred
        pred_df['rf50_std'] = rf50_std
        
        # Save updated predictions
        pred_df.to_csv(pred_file, index=False)
        
        # Save RF50 model
        dump(rf50, f'{save_dir}/{data_name}/rf50/{data_name}_n{n_obs}_r{r}_rf50.joblib', compress=3)
        
        return True
        
    except Exception as e:
        print(f"Error in {data_name} n={n_obs} r={r}: {e}")
        return None


def run_experiments(data_name_list, 
                   n_obs_list=[50, 100, 200, 300, 400, 500],
                   r_list=range(100),
                   save_dir='../results',
                   n_jobs=5):
    """
    Main function to retrain RF20 for all experiments
    
    Args:
        data_name_list: List of dataset names
        n_obs_list: List of sample sizes to test
        r_list: List of replication indices
        save_dir: Directory to save results
        n_jobs: Number of parallel jobs
    """
    # Create directory structure for each dataset
    for data_name in data_name_list:
        rf20_dir = f'{save_dir}/{data_name}/rf20'
        rf50_dir = f'{save_dir}/{data_name}/rf50'
        os.makedirs(rf20_dir, exist_ok=True)
        os.makedirs(rf50_dir, exist_ok=True)
    
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
    
    # Run tasks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_single_experiment)(data_name, n_obs, r, save_dir)
        for data_name, n_obs, r in tasks
    )
    
    # print(f"\nCompleted {sum(1 for r in results if r is not None)} / {len(tasks)} tasks")


def main():
    """Main function with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain RF20 and RF50 with optimized max_depth')
    
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
        data_names =[
                # 'autompg', # 'breastcancer', 
                'fertility', #'forest', # 'housing', 
                'pendulum',
                'qsar_aquatic_toxicity', 
                #'servo',  
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
    print("RF20 and RF50 retraining completed!")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

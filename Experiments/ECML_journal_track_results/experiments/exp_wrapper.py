import os
import pandas as pd
import numpy as np
import sys
import json
sys.path.append('../../../models')
from smoorf.RandomForestRegressor import ExtendedRandomForest

def create_experiment_folders(data_names, create_folder_path):
    """
    Create folder structure for experiments.
    
    Parameters:
    -----------
    data_names : list
        List of dataset names
    create_folder_path : str
        Base path where folders will be created
        
    Returns:
    --------
    None
    """
    # Create base folder if it doesn't exist
    if not os.path.exists(create_folder_path):
        os.makedirs(create_folder_path)
    
    # Create folders for each dataset
    for data_name in data_names:
        # Create dataset folder
        dataset_path = os.path.join(create_folder_path, data_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        # Create subfolders
        subfolders = ['lambdas', 'Pred_std', 'RF']
        for subfolder in subfolders:
            subfolder_path = os.path.join(dataset_path, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
        
        print(f"Created folder structure for {data_name}")


def convert_to_array(data_str):
    return np.array(list(map(int, data_str.split(','))))

def get_bootstrap_samples(bootstrap_df,file_name,q,r,X,y):
    try:
        bootstrap_indices = bootstrap_df[f'{file_name}_bootstrap_{q}'][r]
        oob_indices = bootstrap_df[f'{file_name}_oob_{q}'][r]
        bootstrap_indices = convert_to_array(bootstrap_indices)
        oob_indices = convert_to_array(oob_indices)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        X_oob = X[oob_indices]
        y_oob = y[oob_indices]
        return X_bootstrap, y_bootstrap, X_oob, y_oob
    except:
        return None,None,None,None





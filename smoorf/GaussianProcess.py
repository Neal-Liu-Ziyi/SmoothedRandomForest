'''
    Statistical models for computing prediction mean and uncertainty
'''
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel,RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
#from common import NullScaler

# Gaussian process regression model
class GPmodel():
    def __init__(self, kernel=Matern(nu=2.5)+WhiteKernel(), optimizeHyperparams=True,  random_state=None):
        self.optimizeHyperparams = optimizeHyperparams
        #self.X_scaler = StandardScaler() if self.standardize_x else NullScaler() 
        self.kern = kernel
        self.name = "gp"
        self.random_state = random_state
    
    def fit(self, X, y, num_restarts=10):
        #self.X_scaler.fit(X)
        #nX = self.X_scaler.transform(X)
        num_restarts = num_restarts if self.optimizeHyperparams else 0
        self.gpr = GaussianProcessRegressor(kernel=self.kern, 
                                            n_restarts_optimizer=num_restarts, 
                                            random_state=self.random_state,
                                            normalize_y=True)
        # Suppress convergence warnings during fitting
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            self.gpr.fit(X, y)
        
    def predict(self, X, return_std=True):
        #nX = self.X_scaler.transform(X)
        mean_s, std_s = self.gpr.predict(X, return_std=True)
        if return_std:
            return mean_s.ravel(), std_s.ravel() 
        else:
            return mean_s.ravel()
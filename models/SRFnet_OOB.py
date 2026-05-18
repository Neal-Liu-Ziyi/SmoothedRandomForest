"""
SRFnet_OOB: PyTorch implementation of SRF with OOB (Out-of-Bag) evaluation

This module provides a differentiable version of the SRF model that uses OOB samples
for objective evaluation, similar to the Bayesian optimization approaches but with
gradient-based parameter optimization.

Key Features:
- OOB-based objective function (no debiasing)
- Differentiable implementation using PyTorch
- Multiple smoothing modes: 'global', 'per_dim', 'per_tree', 'per_tree_dim'
- Multiple kernel functions: 'normal', 'hyperbolic_secant'
- Gradient-based optimization with early stopping
- TensorBoard logging support

The main difference from DifferentiableSRF:
1. No debiasing mechanism - uses raw SRF predictions
2. OOB-based evaluation - only uses out-of-bag samples for loss calculation
3. Objective computes individual tree OOB predictions and aggregates MSE

Usage Example:
    model = SRFnetOOB(n_estimators=100, smoothing_mode='global', srf_kernel='normal')
    model.fit(X_train, y_train, epochs=100, lr=0.01)
    predictions = model.predict(X_test)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from models.TreeInfoExtractor import SklearnTreeInfoExtractor
from models.RandomForestRegressor import ExtendedRandomForest
from models.Hypsecant import HyperbolicSecant
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter

class SRFnetOOB(nn.Module):
    def __init__(self, 
                 n_estimators=100,
                 jobs=1,
                 smoothing_mode='global',  # 'global', 'per_dim', 'per_tree', 'per_tree_dim'
                 init_smoothing=1.0,
                 srf_kernel='normal',     # 'normal' or 'hyperbolic_secant'
                 **rf_kwargs):
        super(SRFnetOOB, self).__init__()
        
        self.n_estimators = n_estimators
        self.jobs = jobs
        self.smoothing_mode = smoothing_mode
        self.srf_kernel = srf_kernel
        self.rf_params = rf_kwargs
        
        # Validate srf_kernel parameter
        if self.srf_kernel not in ['normal', 'hyperbolic_secant']:
            raise ValueError(f"srf_kernel must be either 'normal' or 'hyperbolic_secant', got '{self.srf_kernel}'")
        
        # Initialize RF, training data, and tree information
        self.rf = None
        self.X_train = None
        self.y_train = None
        self.bounds = None
        self.scaling_vect = None
        self.n_features = None
        self.n_train_samples = None
        
        # OOB mask for out-of-bag evaluation
        self.oob_mask = None
        
        # Initialize smoothing parameters
        self.init_smoothing = init_smoothing
        self.smoothing_params = None
        
    def _initialize_smoothing_params(self):
        """Initialize smoothing parameters according to smoothing_mode"""
        if self.smoothing_mode == 'global':
            # Single global parameter
            self.smoothing_params = nn.Parameter(torch.tensor(self.init_smoothing))
        elif self.smoothing_mode == 'per_dim':
            # One parameter for each feature
            self.smoothing_params = nn.Parameter(torch.full((self.n_features,), self.init_smoothing))
        elif self.smoothing_mode == 'per_tree':
            # One parameter for each tree
            self.smoothing_params = nn.Parameter(torch.full((self.n_estimators,), self.init_smoothing))
        elif self.smoothing_mode == 'per_tree_dim':
            # One parameter for each tree and each feature
            self.smoothing_params = nn.Parameter(torch.full((self.n_estimators, self.n_features), self.init_smoothing))
        else:
            raise ValueError(f"Unknown smoothing_mode: {self.smoothing_mode}")
    
    def _extract_tree_info_all(self, tree_idx):
        """Extract information for a single tree"""
        extractor = SklearnTreeInfoExtractor(rf=self.rf, tree_idx=tree_idx, 
                                           X=self.X_train, y=self.y_train)
        info = extractor.get_sample_info()
        scaling_vect = extractor.get_scaling_vects()  # shape: (n_train_samples,)
        bounds = extractor.get_sample_boundary()      # shape: (n_train_samples, n_features, 2)
        return info, scaling_vect, bounds, extractor
    
    def _create_oob_mask(self):
        """Create OOB mask from random forest"""
        self.oob_mask = np.zeros((len(self.rf.estimators_), self.X_train.shape[0]), dtype=bool)
        for i, idx_list in enumerate(self.rf.oob_samples_indices):
            self.oob_mask[i, idx_list] = True
    
    def _get_smoothing_tensor(self, n_trees):
        """
        Convert raw trainable parameters to positive smoothing parameters with correct shape
        for broadcasting in probability calculations.
        
        Returns:
            torch.Tensor: Positive smoothing parameters with appropriate shape for broadcasting
        """
        # Transform raw parameters to positive values using softplus
        clamped_params = torch.clamp(self.smoothing_params, min=-10.0, max=10.0)
        positive_params = F.softplus(clamped_params) + 1e-6
        
        # Reshape for broadcasting compatibility
        if self.smoothing_mode == 'global':
            return positive_params.view(1, 1, 1, 1)
        elif self.smoothing_mode == 'per_dim':
            return positive_params.view(1, 1, 1, -1)
        elif self.smoothing_mode == 'per_tree':
            return positive_params.view(-1, 1, 1, 1)
        elif self.smoothing_mode == 'per_tree_dim':
            return positive_params.view(n_trees, 1, 1, -1)
        else:
            raise ValueError(f"Unknown smoothing_mode: {self.smoothing_mode}")
    
    def _compute_probabilities(self, X):
        """Compute probability matrix for all trees"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        n_trees = len(self.rf.estimators_)
        
        # Get smoothing parameters tensor
        scale = self._get_smoothing_tensor(n_trees)
        
        # Convert bounds to tensor
        bounds_tensor = torch.tensor(self.bounds, dtype=torch.float32)  
        lower_bounds = bounds_tensor[..., 0]  # (n_trees, n_train, n_features)
        upper_bounds = bounds_tensor[..., 1]  # (n_trees, n_train, n_features)
        
        # Expand dimensions for broadcasting
        lower_bounds = lower_bounds[:, None, :, :]  # (n_trees, 1, n_train, n_features)
        upper_bounds = upper_bounds[:, None, :, :]  # (n_trees, 1, n_train, n_features)
        X_expanded = X[None, :, None, :]  # (1, batch_size, 1, n_features)
        
        # Compute CDF using selected kernel
        if self.srf_kernel == 'normal':
            dist = Normal(X_expanded, scale)
            lower_cdfs = dist.cdf(lower_bounds)
            upper_cdfs = dist.cdf(upper_bounds)
        elif self.srf_kernel == 'hyperbolic_secant':
            lower_cdfs = HyperbolicSecant.torch_cdf(lower_bounds, X_expanded, scale)
            upper_cdfs = HyperbolicSecant.torch_cdf(upper_bounds, X_expanded, scale)
        else:
            raise ValueError(f"Unsupported srf_kernel: {self.srf_kernel}")
        
        # Compute probabilities
        feature_probs = upper_cdfs - lower_cdfs
        feature_probs = torch.clamp(feature_probs, min=1e-12, max=1.0)
        
        # Handle invalid values
        if torch.any(torch.isnan(feature_probs)) or torch.any(torch.isinf(feature_probs)):
            print("Warning: Invalid values detected in feature probabilities.")
            feature_probs = torch.where(torch.isnan(feature_probs) | torch.isinf(feature_probs), 
                                      torch.tensor(1e-12), feature_probs)
        
        joint_probs = torch.prod(feature_probs, dim=3)  # (n_trees, batch_size, n_train)
        
        # Apply scaling vectors
        scaling_tensor = torch.tensor(self.scaling_vect, dtype=torch.float32)
        scaling_tensor = scaling_tensor[:, None, :]  # (n_trees, 1, n_train)
        
        scaled_probs = joint_probs * scaling_tensor
        
        # Normalization
        sum_probs = scaled_probs.sum(dim=2, keepdim=True)
        normalized_probs = scaled_probs / (sum_probs + 1e-12)
        
        return normalized_probs  # (n_trees, batch_size, n_train)
    
    def forward(self, X):
        """Forward propagation - returns raw ensemble predictions (no debiasing)"""
        # Compute probability matrix
        prob_matrix = self._compute_probabilities(X)  # (n_trees, batch_size, n_train)
        
        # Convert y_train to tensor
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        
        # Compute prediction of each tree
        tree_predictions = torch.matmul(prob_matrix, y_train_tensor)  # (n_trees, batch_size)
        
        # Return average prediction across trees (no debiasing)
        mean_prediction = torch.mean(tree_predictions, dim=0)  # (batch_size,)
        
        return mean_prediction
    
    def _compute_individual_tree_predictions(self, X):
        """Compute individual tree predictions for OOB evaluation"""
        prob_matrix = self._compute_probabilities(X)  # (n_trees, batch_size, n_train)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        tree_predictions = torch.matmul(prob_matrix, y_train_tensor)  # (n_trees, batch_size)
        return tree_predictions
    
    def _compute_oob_loss_differentiable(self):
        """
        Compute OOB loss by aggregating individual tree OOB predictions
        
        Key implementation details:
        1. Computes individual tree predictions on training data
        2. For each tree, extracts only its OOB samples using boolean masking
        3. Calculates MSE for each tree's OOB predictions
        4. Returns weighted average MSE across all OOB samples from all trees
        
        This maintains gradient flow for backpropagation while only using OOB samples.
        """
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        tree_predictions = self._compute_individual_tree_predictions(X_tensor)  # (n_trees, n_samples)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        
        # Create OOB mask tensor
        oob_mask_tensor = torch.tensor(self.oob_mask, dtype=torch.bool)  # (n_trees, n_samples)
        
        # Vectorized extraction of all OOB predictions and targets
        # tree_predictions: (n_trees, n_samples)
        # oob_mask_tensor: (n_trees, n_samples), bool
        
        # Check if we have any OOB samples at all
        if not torch.any(oob_mask_tensor):
            return torch.tensor(float('inf'), requires_grad=True)
        
        # Vectorized selection: get all OOB predictions in one operation
        stacked_oob_preds = tree_predictions[oob_mask_tensor]  # (total_oob_samples,)
        
        # Expand y_train to match tree_predictions shape, then select OOB targets
        y_expanded = y_train_tensor.unsqueeze(0).expand_as(tree_predictions)  # (n_trees, n_samples)
        stacked_oob_targets = y_expanded[oob_mask_tensor]  # (total_oob_samples,)
        
        # Compute MSE between the stacked vectors (same as GP algorithm)
        return F.mse_loss(stacked_oob_preds, stacked_oob_targets)
    
    def fit(self, X, y, rf=None, epochs=100, lr=0.01, weight_decay=1e-4, 
            optimizer=None, scheduler=None,
            verbose=False, tensorboard_log=None):
        """
        Train the model using OOB-based objective
        
        Args:
            X: Training features
            y: Training targets  
            rf: Pre-fitted RandomForest model (optional)
            epochs: Number of training epochs
            lr: Learning rate (used if optimizer is None)
            weight_decay: Weight decay (used if optimizer is None)
            optimizer: Can be one of:
                      - None: use default AdamW(lr=lr, weight_decay=weight_decay)
                      - Callable: function that takes model.parameters() and returns optimizer
                                 Example: lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)
                      - Optimizer instance: must be created AFTER first fit() call
            scheduler: Can be one of:
                      - None: use default ReduceLROnPlateau(patience=10, factor=0.5)
                      - Callable: function that takes optimizer and returns scheduler
                                 Example: lambda opt: torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=100, steps_per_epoch=1)
                      - Scheduler instance: must be created AFTER optimizer
            verbose: Whether to print training progress
            tensorboard_log: Directory for TensorBoard logs
            
        Examples:
            # Example 1: Default usage
            model.fit(X, y, epochs=100, lr=0.01)
            
            # Example 2: Use callable (recommended for first fit)
            model.fit(X, y, 
                      optimizer=lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9))
            
            # Example 3: Define outside and pass in (for re-training)
            model.fit(X, y, epochs=50)  # First fit to initialize parameters
            
            # Now you can create optimizer with initialized parameters
            my_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            my_scheduler = torch.optim.lr_scheduler.OneCycleLR(my_optimizer, max_lr=0.5, epochs=100, steps_per_epoch=1)
            model.fit(X, y, optimizer=my_optimizer, scheduler=my_scheduler, epochs=100)
            
            # Example 4: Disable scheduler
            model.fit(X, y, scheduler=lambda opt: None)
        """
        # Data preprocessing
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.X_train = X
        self.y_train = y.ravel()
        self.n_features = X.shape[1]
        self.n_train_samples = X.shape[0]
        
        if verbose:
            print(f"Training data shape: {X.shape}")
            print(f"Number of features: {self.n_features}")
            print(f"Number of training samples: {self.n_train_samples}")
        
        # Train or use existing RF
        if rf is None:
            self.rf = ExtendedRandomForest(n_estimators=self.n_estimators, **self.rf_params)
            self.rf.fit(X, y)
        else:
            self.rf = rf
        
        # Update n_estimators to actual number of trees
        self.n_estimators = len(self.rf.estimators_)
        
        if verbose:
            print(f"Using {self.n_estimators} trees from the random forest")
        
        # Create OOB mask
        self._create_oob_mask()
        
        if verbose:
            oob_counts = np.sum(self.oob_mask, axis=1)
            print(f"OOB samples per tree: min={oob_counts.min()}, max={oob_counts.max()}, mean={oob_counts.mean():.1f}")
        
        # Extract tree information
        if verbose:
            print("Extracting tree information...")
        
        results = Parallel(n_jobs=self.jobs)(
            delayed(self._extract_tree_info_all)(idx) 
            for idx in range(len(self.rf.estimators_))
        )
        
        tree_info, scaling_vect_list, bounds_list, extractors = zip(*results)
        
        # Convert to numpy arrays
        self.bounds = np.array(bounds_list)
        self.scaling_vect = np.array(scaling_vect_list)
        
        if verbose:
            print(f"Bounds shape: {self.bounds.shape}")
            print(f"Scaling vector shape: {self.scaling_vect.shape}")
        
        # Handle infinite bounds and zero scaling values
        finite_bounds = np.copy(self.bounds)
        finite_bounds[np.isneginf(finite_bounds)] = -1e6
        finite_bounds[np.isposinf(finite_bounds)] = 1e6
        self.bounds = finite_bounds
        
        finite_scaling = np.copy(self.scaling_vect)
        finite_scaling[finite_scaling == 0] = 1e-12
        self.scaling_vect = finite_scaling
        
        # Initialize smoothing parameters
        self._initialize_smoothing_params()
        
        if verbose:
            try:
                if self.smoothing_mode == 'global':
                    param_value = F.softplus(self.smoothing_params).item()
                    print(f"Initial smoothing parameter: {param_value:.6f}")
                else:
                    param_values = F.softplus(self.smoothing_params).detach().numpy()
                    print(f"Initial smoothing parameters shape: {self.smoothing_params.shape}")
                    print(f"Initial smoothing parameters range: [{param_values.min():.6f}, {param_values.max():.6f}]")
            except Exception as e:
                print(f"Warning: Could not display smoothing parameters: {e}")
        
        # Create optimizer: support None (default), callable (factory), or instance (pre-defined)
        if optimizer is None:
            # Default optimizer
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            if verbose:
                print(f"Using default optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")
        elif callable(optimizer):
            # Factory function
            optimizer = optimizer(self.parameters())
            if verbose:
                print(f"Using custom optimizer: {type(optimizer).__name__}")
        else:
            # Pre-instantiated optimizer (directly use it)
            if verbose:
                print(f"Using provided optimizer: {type(optimizer).__name__}")
        
        # Create scheduler: support None (default), callable (factory), or instance (pre-defined)
        if scheduler is None:
            # Default scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            if verbose:
                print("Using default scheduler: ReduceLROnPlateau(patience=10, factor=0.5)")
        elif callable(scheduler):
            # Factory function
            scheduler = scheduler(optimizer)
            if verbose:
                if scheduler is None:
                    print("Scheduler: None (disabled)")
                else:
                    print(f"Using custom scheduler: {type(scheduler).__name__}")
        else:
            # Pre-instantiated scheduler (directly use it)
            if verbose:
                print(f"Using provided scheduler: {type(scheduler).__name__}")
        
        # Initialize TensorBoard writer
        writer = None
        if tensorboard_log is not None:
            writer = SummaryWriter(log_dir=tensorboard_log)
            if verbose:
                print(f"TensorBoard logging to: {tensorboard_log}")
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        # Check if optimizer is LBFGS (requires closure)
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        
        for epoch in range(epochs):
            try:
                # LBFGS requires a closure function
                if is_lbfgs:
                    def closure():
                        optimizer.zero_grad()
                        loss = self._compute_oob_loss_differentiable()
                        if torch.isnan(loss) or torch.isinf(loss):
                            return loss
                        loss.backward()
                        return loss
                    
                    oob_loss = optimizer.step(closure)
                    
                    # Check for invalid loss
                    if torch.isnan(oob_loss) or torch.isinf(oob_loss):
                        print(f"Invalid loss ({oob_loss.item()}) at epoch {epoch}, skipping.")
                        continue
                        
                else:
                    # Standard optimizer (AdamW, SGD, etc.)
                    optimizer.zero_grad()
                    
                    # Compute OOB-based loss
                    oob_loss = self._compute_oob_loss_differentiable()
                    
                    # Check for invalid loss
                    if torch.isnan(oob_loss) or torch.isinf(oob_loss):
                        print(f"Invalid loss ({oob_loss.item()}) at epoch {epoch}, skipping.")
                        continue
                    
                    # Backward pass
                    oob_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                # Learning rate scheduling
                if scheduler is not None:
                    # Handle different scheduler types
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(oob_loss.item())
                    else:
                        # For other schedulers (OneCycleLR, CosineAnnealing, etc.)
                        scheduler.step()
                
                # Early stopping
                if oob_loss.item() < best_loss:
                    best_loss = oob_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                
                # TensorBoard logging
                if writer is not None:
                    writer.add_scalar('Loss/OOB_MSE', oob_loss.item(), epoch)
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                    
                    # Log smoothing parameters
                    if self.smoothing_mode == 'global':
                        writer.add_scalar('Smoothing/Global', F.softplus(self.smoothing_params).item(), epoch)
                    else:
                        smoothing_vals = F.softplus(self.smoothing_params)
                        writer.add_scalar('Smoothing/Mean', smoothing_vals.mean().item(), epoch)
                        writer.add_scalar('Smoothing/Std', smoothing_vals.std().item(), epoch)
                        writer.add_histogram('Smoothing/Distribution', smoothing_vals, epoch)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}, OOB Loss: {oob_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                    
            except Exception as e:
                print(f"Error at epoch {epoch}: {e}")
                continue
        
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
        
        return self
    
    def _compute_model_variance(self):
        """
        Compute model variance (training set MSE)
        
        This represents the irreducible error and model bias.
        Calculated as the MSE on training set predictions.
        
        Returns:
            float: Model variance (scalar, same for all test points)
        """
        with torch.no_grad():
            X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
            y_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            
            # Get training predictions
            train_pred = self.forward(X_tensor)
            
            # Compute MSE
            model_var = F.mse_loss(train_pred, y_tensor).item()
            
            return model_var
    
    def predict(self, X, return_uncertainty=False, return_noise_free_uncertainty=False):
        """
        Predict using the trained model
        
        Args:
            X: Input data for prediction
            return_uncertainty: If True, return total uncertainty (includes model variance)
            return_noise_free_uncertainty: If True, return noise-free uncertainty (excludes model variance)
            
        Returns:
            predictions: Mean predictions across all trees
            uncertainty: (optional) Total uncertainty if return_uncertainty=True
            noise_free_uncertainty: (optional) Noise-free uncertainty if return_noise_free_uncertainty=True
            
        Note:
            - Total uncertainty = √(intra_var + inter_var + model_var)
            - Noise-free uncertainty = √(intra_var + inter_var)
            - If both flags are True, returns (predictions, total_uncertainty, noise_free_uncertainty)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X
            
            if not return_uncertainty and not return_noise_free_uncertainty:
                predictions = self.forward(X_tensor)
                return predictions.numpy()
            else:
                # Get individual tree predictions for uncertainty calculation
                prob_matrix = self._compute_probabilities(X_tensor)  # (n_trees, batch_size, n_train)
                y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
                
                # Compute predictions for each tree
                tree_predictions = torch.matmul(prob_matrix, y_train_tensor)  # (n_trees, batch_size)
                
                # Mean prediction
                mean_prediction = torch.mean(tree_predictions, dim=0)  # (batch_size,)
                
                # Uncertainty components
                # 1. Inter-tree variance (variance across trees)
                inter_tree_var = torch.var(tree_predictions, dim=0, unbiased=False)  # (batch_size,)
                
                # 2. Intra-tree variance (average variance within each tree)
                y_squared = y_train_tensor ** 2
                # E[y²] for each tree
                tree_y_squared = torch.matmul(prob_matrix, y_squared)  # (n_trees, batch_size)
                # Var(y) = E[y²] - E[y]²
                tree_variances = tree_y_squared - tree_predictions ** 2  # (n_trees, batch_size)
                intra_tree_var = torch.mean(tree_variances, dim=0)  # (batch_size,)
                
                # 3. Model variance (training set MSE, constant for all test points)
                model_var = self._compute_model_variance()
                
                # Noise-free uncertainty (excludes model variance)
                noise_free_variance = inter_tree_var + intra_tree_var
                noise_free_variance = torch.clamp(noise_free_variance, min=1e-12)
                noise_free_uncertainty = torch.sqrt(noise_free_variance)
                
                # Total uncertainty (includes model variance)
                total_variance = inter_tree_var + intra_tree_var + model_var
                total_variance = torch.clamp(total_variance, min=1e-12)
                total_uncertainty = torch.sqrt(total_variance)
                
                # Return based on flags
                if return_uncertainty and return_noise_free_uncertainty:
                    return mean_prediction.numpy(), total_uncertainty.numpy(), noise_free_uncertainty.numpy()
                elif return_uncertainty:
                    return mean_prediction.numpy(), total_uncertainty.numpy()
                else:  # return_noise_free_uncertainty only
                    return mean_prediction.numpy(), noise_free_uncertainty.numpy()
    
    def smoothed_trees_predict(self, X):
        """
        Predict using individual trees (no averaging)
        
        Args:
            X: Input data for prediction
            
        Returns:
            np.ndarray: Individual tree predictions with shape (n_samples, n_trees)
                       Each column represents one tree's predictions
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X
            
            # Compute probability matrix for all trees
            prob_matrix = self._compute_probabilities(X_tensor)  # (n_trees, batch_size, n_train)
            
            # Convert y_train to tensor
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            
            # Compute prediction of each tree
            tree_predictions = torch.matmul(prob_matrix, y_train_tensor)  # (n_trees, batch_size)
            
            # Transpose to get (n_samples, n_trees) format
            individual_predictions = tree_predictions.transpose(0, 1)  # (batch_size, n_trees)
            
            return individual_predictions.numpy()
    
    def get_detailed_uncertainty(self, X):
        """
        Get detailed uncertainty decomposition (consistent with AdaptiveDebiasedSRF)
        
        Returns predictions and all three uncertainty components:
        - predictions: Mean predictions
        - intra_tree_variance: Average variance within each tree (smoothing-induced)
        - inter_tree_variance: Variance across different trees (ensemble diversity)
        - model_variance: Model error on training set (irreducible error + bias)
        
        This allows analyzing which component contributes more to uncertainty.
        
        Args:
            X: Input data for prediction
            
        Returns:
            tuple: (predictions, intra_tree_var, inter_tree_var, model_var)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X
            
            # Compute probability matrix for all trees
            prob_matrix = self._compute_probabilities(X_tensor)  # (n_trees, batch_size, n_train)
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            
            # Compute predictions for each tree
            tree_predictions = torch.matmul(prob_matrix, y_train_tensor)  # (n_trees, batch_size)
            
            # Mean prediction
            mean_prediction = torch.mean(tree_predictions, dim=0)  # (batch_size,)
            
            # 1. Inter-tree variance (variance across trees)
            inter_tree_var = torch.var(tree_predictions, dim=0, unbiased=False)  # (batch_size,)
            
            # 2. Intra-tree variance (average variance within each tree)
            y_squared = y_train_tensor ** 2
            # E[y²] for each tree
            tree_y_squared = torch.matmul(prob_matrix, y_squared)  # (n_trees, batch_size)
            # Var(y) = E[y²] - E[y]²
            tree_variances = tree_y_squared - tree_predictions ** 2  # (n_trees, batch_size)
            intra_tree_var = torch.mean(tree_variances, dim=0)  # (batch_size,)
            
            # 3. Model variance (training set MSE, scalar)
            model_var = self._compute_model_variance()
            
            return (intra_tree_var.numpy(), 
                    inter_tree_var.numpy(),
                    model_var)
    
    def get_smoothing_params(self):
        """Get current smoothing parameters"""
        with torch.no_grad():
            return F.softplus(self.smoothing_params).numpy()

    def get_prob_matrix(self, X):
        """Return probability matrix shape (n_trees, n_samples, n_train)"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X
            return self._compute_probabilities(X_tensor).numpy()
    
    def get_oob_score(self):
        """Get current OOB score (MSE)"""
        with torch.no_grad():
            return self._compute_oob_loss_differentiable().item()
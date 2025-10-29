"""
SRFnet_OOB Optimizer and Scheduler Configurations

Provides optimized optimizer and scheduler configuration functions for four different smoothing_modes.
These functions can be directly used in the fit() method.

Usage:
    model = SRFnetOOB(smoothing_mode='global')
    model.fit(X, y, 
              optimizer=optimizer_global,
              scheduler=scheduler_global)
"""

import torch


# ============================================================================
# Global Mode: 1 smoothing parameter
# Recommendation: LBFGS (second-order optimization, fastest convergence)
# ============================================================================

def optimizer_global(params):
    """
    Global smoothing mode optimizer
    
    LBFGS is the optimal choice for a single parameter:
    - Uses second-order information (Hessian approximation)
    - Fast convergence
    - Requires minimal hyperparameter tuning
    """
    return torch.optim.LBFGS(
        params,
        lr=1.0,              # LBFGS lr is typically set to 1.0
        max_iter=20,         # Maximum iterations per optimization step
        history_size=10,     # Number of gradient histories to keep
        line_search_fn='strong_wolfe'  # Use strong Wolfe line search
    )


def scheduler_global(optimizer):
    """
    Global smoothing mode scheduler
    
    LBFGS typically does not need a learning rate scheduler
    Returns None to disable scheduler
    """
    return None


# ============================================================================
# Per_Dim Mode: n_features smoothing parameters
# Recommendation: SGD with Momentum + Cosine Annealing
# ============================================================================

def optimizer_per_dim(params):
    """
    Per-dimension smoothing mode optimizer
    
    Number of parameters = number of features (typically 5-50)
    SGD + Momentum provides stable optimization:
    - Simple and reliable
    - Nesterov momentum accelerates convergence
    - Computationally efficient
    """
    return torch.optim.SGD(
        params,
        lr=0.01,             # Initial learning rate
        momentum=0.9,        # Momentum coefficient
        nesterov=True,       # Use Nesterov momentum
        weight_decay=1e-5    # Light weight decay
    )


def scheduler_per_dim(optimizer):
    """
    Per-dimension smoothing mode scheduler
    
    Cosine Annealing provides smooth learning rate decay:
    - Avoids sudden learning rate changes
    - Helps convergence to better local optima
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,           # Total number of epochs (adjust based on actual usage)
        eta_min=1e-6         # Minimum learning rate
    )


# ============================================================================
# Per_Tree Mode: n_estimators smoothing parameters  
# Recommendation: AdamW + ReduceLROnPlateau
# ============================================================================

def optimizer_per_tree(params):
    """
    Per-tree smoothing mode optimizer
    
    Number of parameters = number of trees (typically 50-200)
    AdamW adaptively adjusts learning rate for each parameter:
    - Adaptive learning rate for different tree parameters
    - Weight decay prevents overfitting
    - Robust to noise
    """
    return torch.optim.AdamW(
        params,
        lr=0.01,             # Initial learning rate
        betas=(0.9, 0.999),  # Adam momentum parameters
        eps=1e-8,            # Numerical stability
        weight_decay=1e-4    # Weight decay (regularization)
    )


def scheduler_per_tree(optimizer):
    """
    Per-tree smoothing mode scheduler
    
    ReduceLROnPlateau dynamically adjusts learning rate based on validation performance:
    - Reduces learning rate when OOB loss plateaus
    - Adaptive, no need to specify total epochs
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Monitor minimization metric
        factor=0.5,          # Learning rate decay factor
        patience=10,         # Number of epochs to wait before decay
        min_lr=1e-6          # Minimum learning rate
    )


# ============================================================================
# Per_Tree_Dim Mode: n_estimators × n_features smoothing parameters
# Recommendation: AdamW + OneCycleLR
# ============================================================================

def optimizer_per_tree_dim(params):
    """
    Per-tree-dimension smoothing mode optimizer
    
    Number of parameters = num_trees × num_features (potentially hundreds to thousands)
    AdamW is the best choice for high-dimensional parameter spaces:
    - Independent adaptive learning rate for each parameter
    - Memory efficient (compared to second-order methods)
    - Robust to gradient noise
    """
    return torch.optim.AdamW(
        params,
        lr=0.01,             # Initial learning rate (OneCycleLR will override)
        betas=(0.9, 0.999),  # Adam momentum parameters
        eps=1e-8,
        weight_decay=1e-3    # Stronger regularization (many parameters, prevent overfitting)
    )


def scheduler_per_tree_dim(optimizer):
    """
    Per-tree-dimension smoothing mode scheduler
    
    OneCycleLR implements super-convergence:
    - Fast convergence (suitable for many parameters)
    - Automatic warm-up and annealing
    - Often achieves better performance
    
    Note: Requires knowing the total number of epochs in advance
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,          # Peak learning rate (reached during mid-training)
        epochs=100,          # Total number of epochs (must match epochs in fit())
        steps_per_epoch=1,   # Number of steps per epoch (1 for full-batch training)
        pct_start=0.3,       # Fraction of training for warm-up phase
        anneal_strategy='cos',  # Annealing strategy ('cos' or 'linear')
        div_factor=25.0,     # initial_lr = max_lr / div_factor
        final_div_factor=1e4 # final_lr = initial_lr / final_div_factor
    )


# ============================================================================
# Convenience function: Automatically select based on smoothing_mode
# ============================================================================

def get_optimizer_scheduler(smoothing_mode, total_epochs=100):
    """
    Automatically return recommended optimizer and scheduler based on smoothing_mode
    
    Args:
        smoothing_mode: 'global', 'per_dim', 'per_tree', 'per_tree_dim'
        total_epochs: Total number of training epochs (needed for OneCycleLR in per_tree_dim)
    
    Returns:
        optimizer_fn, scheduler_fn: Two functions that can be directly passed to fit()
        
    Usage:
        opt_fn, sch_fn = get_optimizer_scheduler('global')
        model.fit(X, y, optimizer=opt_fn, scheduler=sch_fn)
    """
    configs = {
        'global': (
            optimizer_global,
            scheduler_global
        ),
        'per_dim': (
            optimizer_per_dim,
            scheduler_per_dim
        ),
        'per_tree': (
            optimizer_per_tree,
            scheduler_per_tree
        ),
        'per_tree_dim': (
            # Need to modify OneCycleLR epochs parameter
            optimizer_per_tree_dim,
            lambda opt: torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=0.1, epochs=total_epochs, steps_per_epoch=1,
                pct_start=0.3, anneal_strategy='cos'
            )
        )
    }
    
    if smoothing_mode not in configs:
        raise ValueError(f"Unknown smoothing_mode: {smoothing_mode}. "
                        f"Must be one of {list(configs.keys())}")
    
    return configs[smoothing_mode]


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    from SRFnet_OOB import SRFnetOOB
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    print("=" * 70)
    print("Usage Examples: Recommended configurations for four smoothing_modes")
    print("=" * 70)
    
    # ========== Global Mode ==========
    print("\n1. Global Mode (LBFGS)")
    model = SRFnetOOB(n_estimators=30, smoothing_mode='global')
    model.fit(
        X, y,
        epochs=20,
        optimizer=optimizer_global,
        scheduler=scheduler_global,
        verbose=True
    )
    
    # ========== Per_Dim Mode ==========
    print("\n2. Per_Dim Mode (SGD + Momentum + CosineAnnealing)")
    model = SRFnetOOB(n_estimators=30, smoothing_mode='per_dim')
    model.fit(
        X, y,
        epochs=50,
        optimizer=optimizer_per_dim,
        scheduler=scheduler_per_dim,
        verbose=True
    )
    
    # ========== Per_Tree Mode ==========
    print("\n3. Per_Tree Mode (AdamW + ReduceLROnPlateau)")
    model = SRFnetOOB(n_estimators=30, smoothing_mode='per_tree')
    model.fit(
        X, y,
        epochs=50,
        optimizer=optimizer_per_tree,
        scheduler=scheduler_per_tree,
        verbose=True
    )
    
    # ========== Per_Tree_Dim Mode ==========
    print("\n4. Per_Tree_Dim Mode (AdamW + OneCycleLR)")
    model = SRFnetOOB(n_estimators=30, smoothing_mode='per_tree_dim')
    model.fit(
        X, y,
        epochs=100,
        optimizer=optimizer_per_tree_dim,
        scheduler=scheduler_per_tree_dim,
        verbose=True
    )
    
    # ========== Using convenience function ==========
    print("\n5. Using convenience function to auto-select configurations")
    for mode in ['global', 'per_dim', 'per_tree', 'per_tree_dim']:
        print(f"\n--- {mode} mode ---")
        model = SRFnetOOB(n_estimators=20, smoothing_mode=mode)
        opt_fn, sch_fn = get_optimizer_scheduler(mode, total_epochs=50)
        model.fit(X, y, epochs=50, optimizer=opt_fn, scheduler=sch_fn, verbose=True)
    
    print("\n" + "=" * 70)
    print("Completed!")
    print("=" * 70)

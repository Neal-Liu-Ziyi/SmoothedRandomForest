"""
SRFinference.py
===============
Inference-only sibling of `SRFnet_OOB`. Pure numpy + scipy, no PyTorch.

Use case
--------
You already have:
  - a fitted RandomForest (an `ExtendedRandomForest`, or a sklearn
    `RandomForestRegressor` with `used_samples_indices` attached)
  - smoothing bandwidth(s) trained elsewhere (e.g. by `SRFnet_OOB`)
  - optionally, OOB-linear calibration coefficients `(coef, intercept)`

and you want predictions, effective-kernel matrices, and the variance
decomposition that `SRFnet_OOB.get_detailed_uncertainty` returns — without
re-running gradient descent or re-fitting the calibration.

Variance decomposition (raw — matches `SRFnet_OOB`)
---------------------------------------------------
With ``K_t`` the per-tree effective kernel and ``y`` the training targets,

    tree_pred[t, j] = K_t[j, :] @ y                          # (n_trees, n_test)
    inter_var[j]    = Var_t(tree_pred[:, j])    (ddof=0)
    intra_var[j]    = Mean_t(K_t[j, :] @ y² - tree_pred[t, j]²)
    model_var       = MSE(mean_t tree_pred_train, y)

Calibrated (only applied when ``predict(..., calibrate=True)`` and the
calibration was supplied to ``fit``):

    intra_cal  = coef² · intra_var
    inter_cal  = coef² · inter_var
    model_cal  = MSE(coef · mean_t tree_pred_train + intercept, y)
    total_std  = sqrt(intra_cal + inter_cal + model_cal)
    nf_std     = sqrt(intra_cal + inter_cal)

What is NOT here (intentionally)
--------------------------------
- Smoothing-parameter optimisation (use `SRFnet_OOB` for that).
- OOB-linear-calibration fitting (also `SRFnet_OOB` / `SRFRegressor`).
- Per-test-point RF distance reweighting (`weighted_rf` in
  `AdaptiveDebiasSRF`).
- Derivative-of-prediction utilities (the bottom block of
  `AdaptiveDebiasSRF.py`).
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

from smoorf.Hypsecant import HyperbolicSecant
from smoorf.TreeInfoExtractor import SklearnTreeInfoExtractor


class SRFInference:
    """
    Inference-only smoothed random forest.

    Parameters
    ----------
    smoothing_params : float | np.ndarray
        Positive bandwidth(s). Shape interpretation:

        - ``per_tree=False``:  scalar (STE) | shape ``(n_features,)`` (STE-PD)
        - ``per_tree=True``:   shape ``(n_estimators,)`` (EST) |
                               shape ``(n_estimators, n_features)`` (EST-PD)

    kernel : object, default ``HyperbolicSecant``
        Anything with a ``cdf(x, loc=..., scale=...)`` interface:
        ``scipy.stats.norm``, ``scipy.stats.cauchy``, or
        ``smoorf.Hypsecant.HyperbolicSecant``.

    per_tree : bool
        Whether each tree has its own bandwidth (EST / EST-PD).

    n_jobs : int
        Parallelism for tree-info extraction in ``fit``.
    """

    def __init__(self, smoothing_params, kernel=HyperbolicSecant,
                 per_tree=False, n_jobs=1):
        sp_arr = np.asarray(smoothing_params, dtype=float)
        if np.any(sp_arr <= 0):
            raise ValueError("smoothing_params must be strictly positive.")
        self.smoothing_params = sp_arr
        self.kernel = kernel
        self.per_tree = bool(per_tree)
        self.n_jobs = n_jobs

        # Populated by fit().
        self.rf = None
        self.X_train = None
        self.y_train = None
        self.tree_info = None
        self.bounds = None              # (n_trees, n_train, n_features, 2)
        self.scaling_vect = None        # (n_trees, n_train)
        self.coef = None
        self.intercept = None
        self._train_pred_mean_cache = None   # (n_train,)

    # ------------------------------------------------------------------
    # Fitting (no optimisation)
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, rf, coef=None, intercept=None):
        """
        Extract tree info from ``rf`` and (optionally) store the
        OOB-linear-calibration parameters.

        Parameters
        ----------
        X_train, y_train : training data the RF was fitted on.
        rf : ``ExtendedRandomForest`` (or any sklearn RF that has been
             upgraded to carry ``used_samples_indices``).
        coef, intercept : optional floats. If both are supplied,
             ``predict(calibrate=True)`` and ``get_detailed_uncertainty``
             will use them.

        Returns
        -------
        self
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).ravel()
        self.rf = rf

        n_trees = len(rf.estimators_)
        n_features = self.X_train.shape[1]
        self._validate_smoothing_shape(n_trees, n_features)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_one_tree)(idx) for idx in range(n_trees)
        )
        tree_info, scaling_vects, bounds, _ = zip(*results)
        self.tree_info = list(tree_info)
        self.bounds = np.asarray(bounds)
        self.scaling_vect = np.asarray(scaling_vects)

        self.coef = float(coef) if coef is not None else None
        self.intercept = float(intercept) if intercept is not None else None
        self._train_pred_mean_cache = None   # invalidate

        return self

    def _validate_smoothing_shape(self, n_trees, n_features):
        sp = self.smoothing_params
        if self.per_tree:
            ok = (
                sp.ndim == 0
                or (sp.ndim == 1 and sp.shape == (n_trees,))
                or (sp.ndim == 2 and sp.shape == (n_trees, n_features))
            )
            if not ok:
                raise ValueError(
                    f"With per_tree=True, smoothing_params must be scalar, "
                    f"({n_trees},), or ({n_trees}, {n_features}); got {sp.shape}."
                )
        else:
            ok = sp.ndim == 0 or (sp.ndim == 1 and sp.shape == (n_features,))
            if not ok:
                raise ValueError(
                    f"With per_tree=False, smoothing_params must be scalar or "
                    f"({n_features},); got {sp.shape}."
                )

    def _extract_one_tree(self, tree_idx):
        ex = SklearnTreeInfoExtractor(
            rf=self.rf, tree_idx=tree_idx,
            X=self.X_train, y=self.y_train,
        )
        return (
            ex.get_sample_info(),
            ex.get_scaling_vects(),
            ex.get_sample_boundary(),
            ex,
        )

    # ------------------------------------------------------------------
    # Effective kernel
    # ------------------------------------------------------------------
    def _apply_cdf(self, x, loc, scale):
        try:
            return self.kernel.cdf(x, loc=loc, scale=scale)
        except (AttributeError, TypeError):
            return self.kernel().cdf(x, loc=loc, scale=scale)

    def _broadcast_scale(self, n_features):
        """
        Reshape ``self.smoothing_params`` so it broadcasts against
        ``(n_trees, n_X, n_train, n_features)``.
        """
        sp = self.smoothing_params
        M = self.bounds.shape[0]
        p = n_features
        if self.per_tree:
            if sp.ndim == 0:
                return sp.reshape(1, 1, 1, 1)              # shared across all
            if sp.ndim == 1:
                return sp.reshape(M, 1, 1, 1)              # per-tree
            return sp.reshape(M, 1, 1, p)                  # per-tree, per-feature
        else:
            if sp.ndim == 0:
                return sp.reshape(1, 1, 1, 1)
            return sp.reshape(1, 1, 1, p)                  # per-feature, shared across trees

    def _continuous_features_prob(self, X):
        """
        Per-tree feature-CDF differences, product over features.
        Returns shape ``(n_trees, n_X, n_train)``.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _, p = X.shape
        scale = self._broadcast_scale(p)

        lower = self.bounds[..., 0][:, np.newaxis, :, :]   # (M, 1, n_train, p)
        upper = self.bounds[..., 1][:, np.newaxis, :, :]
        X_exp = X[np.newaxis, :, np.newaxis, :]            # (1, n_X, 1, p)

        lower_cdf = self._apply_cdf(lower, loc=X_exp, scale=scale)
        upper_cdf = self._apply_cdf(upper, loc=X_exp, scale=scale)
        return np.prod(upper_cdf - lower_cdf, axis=3)      # (M, n_X, n_train)

    def _kernel_per_tree(self, X):
        """
        Per-tree effective kernel, shape ``(n_trees, n_X, n_train)``,
        scaled by ``scaling_vect`` and row-normalised.
        """
        probs = self._continuous_features_prob(X)
        scaled = probs * self.scaling_vect[:, np.newaxis, :]
        scaled /= scaled.sum(axis=2, keepdims=True) + 1e-12
        return scaled

    def get_kernel_matrix(self, X):
        """
        Per-tree effective kernel matrix. Shape ``(n_trees, n_X, n_train)``.
        """
        self._check_fitted()
        return self._kernel_per_tree(np.asarray(X))

    def get_effective_kernel(self, X):
        """
        Mean-over-trees effective kernel. Shape ``(n_X, n_train)``.

        Row ``j`` is the (already row-normalised) weight distribution of
        test point ``j`` over training points — the convention used by
        ``EmpiricalLikelihoodPredictor``.
        """
        return self.get_kernel_matrix(X).mean(axis=0)

    # ------------------------------------------------------------------
    # Per-tree predictions
    # ------------------------------------------------------------------
    def smoothed_trees_predict(self, X):
        """
        Per-tree raw predictions. Shape ``(n_X, n_trees)`` — same layout
        as ``SRFnet_OOB.smoothed_trees_predict``.
        """
        self._check_fitted()
        K_t = self._kernel_per_tree(np.asarray(X))         # (M, n_X, n_train)
        tree_pred = K_t @ self.y_train                      # (M, n_X)
        return tree_pred.T                                  # (n_X, n_trees)

    # ------------------------------------------------------------------
    # Variance decomposition — exactly mirrors SRFnet_OOB
    # ------------------------------------------------------------------
    def get_detailed_uncertainty(self, X):
        """
        Raw uncertainty decomposition (uncalibrated), matching
        ``SRFnet_OOB.get_detailed_uncertainty``.

        Returns
        -------
        intra_var : np.ndarray, shape (n_X,)
            ``Mean_t(K_t @ y² - (K_t @ y)²)``.
        inter_var : np.ndarray, shape (n_X,)
            ``Var_t(K_t @ y)``  with ``ddof=0``.
        model_var : float
            ``MSE(mean_t K_t_train @ y, y)`` — training-set MSE of the
            uncalibrated SRF prediction.
        """
        self._check_fitted()
        K_t = self._kernel_per_tree(np.asarray(X))         # (M, n_X, n_train)
        tree_pred = K_t @ self.y_train                      # (M, n_X)
        intra_var, inter_var = self._intra_inter_from(K_t, tree_pred)
        return intra_var, inter_var, self._raw_model_var()

    def _intra_inter_from(self, K_t, tree_pred):
        inter_var = np.var(tree_pred, axis=0, ddof=0)
        y_sq = self.y_train ** 2
        tree_y_sq = K_t @ y_sq
        intra_var = (tree_y_sq - tree_pred ** 2).mean(axis=0)
        return intra_var, inter_var

    def _train_pred_mean(self):
        """Mean-over-trees SRF prediction on the training set (cached)."""
        if self._train_pred_mean_cache is None:
            K_t_train = self._kernel_per_tree(self.X_train)
            self._train_pred_mean_cache = (K_t_train @ self.y_train).mean(axis=0)
        return self._train_pred_mean_cache

    def _raw_model_var(self):
        return float(mean_squared_error(self.y_train, self._train_pred_mean()))

    def _calibrated_model_var(self):
        if self.coef is None or self.intercept is None:
            return self._raw_model_var()
        cal = self.coef * self._train_pred_mean() + self.intercept
        return float(mean_squared_error(self.y_train, cal))

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X,
                return_uncertainty=False,
                return_noise_free_uncertainty=False,
                calibrate=True):
        """
        Predict with optional uncertainty, optionally OOB-linear-calibrated.

        Parameters
        ----------
        X : array-like, shape (n_X, n_features)
        return_uncertainty : bool
            If True, also return ``total_std = √(intra + inter + model_var)``.
        return_noise_free_uncertainty : bool
            If True, also return ``noise_free_std = √(intra + inter)``.
        calibrate : bool
            When True AND both ``coef``/``intercept`` were supplied to
            ``fit()``, apply the OOB-linear calibration to predictions and
            variances; otherwise return raw values.

        Returns
        -------
        pred                                 (if no uncertainty flags)
        (pred, total_std)                    (return_uncertainty only)
        (pred, noise_free_std)               (noise-free flag only)
        (pred, total_std, noise_free_std)    (both flags set)
        """
        self._check_fitted()
        X = np.asarray(X)
        K_t = self._kernel_per_tree(X)                    # (M, n_X, n_train)
        tree_pred = K_t @ self.y_train                     # (M, n_X)
        raw_pred = tree_pred.mean(axis=0)                  # (n_X,)

        do_cal = (
            calibrate and self.coef is not None and self.intercept is not None
        )
        pred = self.coef * raw_pred + self.intercept if do_cal else raw_pred

        if not return_uncertainty and not return_noise_free_uncertainty:
            return pred

        intra_var, inter_var = self._intra_inter_from(K_t, tree_pred)
        if do_cal:
            intra_var = intra_var * self.coef ** 2
            inter_var = inter_var * self.coef ** 2
            model_var = self._calibrated_model_var()
        else:
            model_var = self._raw_model_var()

        nf_var = np.clip(intra_var + inter_var, 1e-12, None)
        nf_std = np.sqrt(nf_var)

        if return_uncertainty and return_noise_free_uncertainty:
            total_std = np.sqrt(np.clip(nf_var + model_var, 1e-12, None))
            return pred, total_std, nf_std
        if return_uncertainty:
            total_std = np.sqrt(np.clip(nf_var + model_var, 1e-12, None))
            return pred, total_std
        return pred, nf_std

    # ------------------------------------------------------------------
    def get_smoothing_params(self):
        """Return the stored (positive) bandwidth array."""
        return self.smoothing_params

    def _check_fitted(self):
        if self.rf is None:
            raise RuntimeError(
                "SRFInference is not fitted. Call .fit(X_train, y_train, rf) first."
            )

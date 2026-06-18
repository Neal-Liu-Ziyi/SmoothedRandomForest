"""
SRFRegressor: High-level wrapper for SRFnet with built-in OOB calibration.

This module provides a user-friendly interface that combines:
  - Random Forest fitting (or accepting a pre-trained RF)
  - SRFnet smoothing parameter optimization
  - OOB linear regression calibration (as used in the ECML paper experiments)
  - Calibrated predictions with full uncertainty decomposition

Default settings mirror the recommended configuration from the paper:
  - smoothing_mode : 'EST_PD'  (EST with per-dimension)
  - srf_kernel     : 'hyperbolic_secant'
  - n_estimators   : 10  (RF10)
  - epochs         : 100

Typical usage
-------------
    from models.SRFRegressor import SRFRegressor

    model = SRFRegressor()
    model.fit(X_train, y_train)

    # Point predictions (calibrated)
    y_pred = model.predict(X_test)

    # Calibrated predictions + full uncertainty
    y_pred, total_std, noise_free_std = model.predict(
        X_test,
        return_uncertainty=True,
        return_noise_free_uncertainty=True
    )

    # Full variance decomposition
    components = model.predict_with_components(X_test)
    # keys: pred, total_std, noise_free_std, intra_var, inter_var, model_var
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from models.RandomForestRegressor import ExtendedRandomForest
from models.SRFnet_OOB import SRFnetOOB
from models.optimizer_configs import get_optimizer_scheduler


class SRFRegressor:
    """
    End-to-end SRFnet predictor with OOB calibration.

    Parameters
    ----------
    smoothing_mode : str, default='EST_PD'
        Smoothing granularity (paper names; legacy names in parentheses are also accepted):
        - 'STE'     (legacy: 'global')        : one shared parameter for all trees and features
        - 'STE_PD'  (legacy: 'per_dim')       : one parameter per feature
        - 'EST'     (legacy: 'per_tree')      : one parameter per tree
        - 'EST_PD'  (legacy: 'per_tree_dim')  : one parameter per (tree, feature) — recommended by the paper
    srf_kernel : str, default='hyperbolic_secant'
        Kernel function used for smoothing. Options:
        - 'normal'             : Gaussian kernel
        - 'hyperbolic_secant'  : heavy-tailed kernel — recommended by the paper
    n_estimators : int, default=10
        Number of trees in the underlying Random Forest (RF10 in the paper).
        Ignored when a pre-trained RF is passed to ``fit()``.
    epochs : int, default=100
        Number of gradient-descent epochs for smoothing parameter optimization.
    init_smoothing : float, default=0.5
        Initial value for all smoothing parameters (before softplus transform).
    n_jobs : int, default=1
        Parallelism for tree-info extraction inside SRFnetOOB.
    verbose : bool, default=False
        Print training progress.
    **rf_kwargs
        Extra keyword arguments forwarded to ``ExtendedRandomForest`` when
        no pre-trained RF is supplied (e.g. ``max_depth``, ``min_samples_leaf``).
    """

    def __init__(
        self,
        smoothing_mode: str = 'EST_PD',
        srf_kernel: str = 'hyperbolic_secant',
        n_estimators: int = 10,
        epochs: int = 100,
        init_smoothing: float = 0.5,
        n_jobs: int = 1,
        verbose: bool = False,
        **rf_kwargs,
    ):
        self.smoothing_mode = smoothing_mode
        self.srf_kernel = srf_kernel
        self.n_estimators = n_estimators
        self.epochs = epochs
        self.init_smoothing = init_smoothing
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.rf_kwargs = rf_kwargs

        # Fitted attributes (set by fit())
        self._srf: SRFnetOOB | None = None
        self._rf: ExtendedRandomForest | None = None
        self._calibrator: LinearRegression | None = None
        self._calib_coef: float | None = None
        self._calib_intercept: float | None = None
        self._model_var_cal: float | None = None   # cached after fit()
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            rf=None) -> "SRFRegressor":
        """
        Fit SRFnet and compute the OOB calibration coefficients.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
        y_train : array-like, shape (n_samples,)
        rf : sklearn RandomForestRegressor or ExtendedRandomForest, optional
            Pre-trained RF to reuse.  If ``None``, a new ``ExtendedRandomForest``
            with ``n_estimators`` trees is fitted from scratch.
            The RF is automatically upgraded to ``ExtendedRandomForest`` if needed.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).ravel()
        self._X_train = X_train
        self._y_train = y_train

        # ── 1. Prepare RF ─────────────────────────────────────────────
        if rf is None:
            if self.verbose:
                print(f"Fitting ExtendedRandomForest with {self.n_estimators} trees …")
            self._rf = ExtendedRandomForest(
                n_estimators=self.n_estimators,
                oob_score=False,
                bootstrap=True,
                **self.rf_kwargs,
            )
            self._rf.fit(X_train, y_train)
        else:
            # Upgrade to ExtendedRandomForest if necessary
            if not isinstance(rf, ExtendedRandomForest):
                if self.verbose:
                    print("Upgrading provided RF to ExtendedRandomForest …")
                rf = ExtendedRandomForest.upgrade(rf, X_train, y_train)
            elif not hasattr(rf, 'oob_samples_indices'):
                # Already ExtendedRandomForest but OOB indices not reconstructed
                rf = ExtendedRandomForest.upgrade(rf, X_train, y_train)
            self._rf = rf

        # ── 2. Build SRFnet ───────────────────────────────────────────
        self._srf = SRFnetOOB(
            smoothing_mode=self.smoothing_mode,
            srf_kernel=self.srf_kernel,
            init_smoothing=self.init_smoothing,
            jobs=self.n_jobs,
        )

        opt_fn, sch_fn = get_optimizer_scheduler(
            self.smoothing_mode, total_epochs=self.epochs
        )

        if self.verbose:
            print(f"Fitting SRFnet  [mode={self.smoothing_mode}, "
                  f"kernel={self.srf_kernel}, epochs={self.epochs}] …")

        self._srf.fit(
            X_train, y_train,
            rf=self._rf,
            epochs=self.epochs,
            optimizer=opt_fn,
            scheduler=sch_fn,
            verbose=self.verbose,
        )

        # ── 3. OOB linear calibration ─────────────────────────────────
        if self.verbose:
            print("Computing OOB calibration …")
        self._fit_calibrator(X_train, y_train)

        return self

    def predict(
        self,
        X_test: np.ndarray,
        return_uncertainty: bool = False,
        return_noise_free_uncertainty: bool = False,
    ):
        """
        Return calibrated predictions (and optionally calibrated uncertainties).

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
        return_uncertainty : bool
            If True, also return calibrated total uncertainty (std).
        return_noise_free_uncertainty : bool
            If True, also return calibrated noise-free (epistemic) uncertainty (std).

        Returns
        -------
        pred : np.ndarray, shape (n_samples,)
            Calibrated point predictions.
        total_std : np.ndarray, shape (n_samples,)  — only if return_uncertainty=True
            Calibrated total uncertainty (√(intra + inter + model variance)).
        noise_free_std : np.ndarray, shape (n_samples,)  — only if return_noise_free_uncertainty=True
            Calibrated noise-free uncertainty (√(intra + inter variance)).
        """
        self._check_is_fitted()
        X_test = np.asarray(X_test)

        if not return_uncertainty and not return_noise_free_uncertainty:
            raw_pred = self._srf.predict(X_test)
            return self._calibrate_predictions(raw_pred)

        # Need uncertainty components
        raw_pred, raw_total_std, raw_nf_std = self._srf.predict(
            X_test,
            return_uncertainty=True,
            return_noise_free_uncertainty=True,
        )
        intra_var, inter_var, model_var = self._srf.get_detailed_uncertainty(X_test)

        pred_cal, total_std_cal, nf_std_cal = self._calibrate_with_uncertainty(
            raw_pred, intra_var, inter_var, model_var, raw_nf_std
        )

        if return_uncertainty and return_noise_free_uncertainty:
            return pred_cal, total_std_cal, nf_std_cal
        elif return_uncertainty:
            return pred_cal, total_std_cal
        else:
            return pred_cal, nf_std_cal

    def predict_with_components(self, X_test: np.ndarray) -> dict:
        """
        Return calibrated predictions and the full variance decomposition.

        Returns
        -------
        dict with keys:
            'pred'           – calibrated point predictions
            'total_std'      – calibrated total uncertainty
            'noise_free_std' – calibrated noise-free (epistemic) uncertainty
            'intra_var'      – calibrated intra-tree variance (per test point)
            'inter_var'      – calibrated inter-tree variance (per test point)
            'model_var'      – calibrated model variance (scalar, same for all points)
        """
        self._check_is_fitted()
        X_test = np.asarray(X_test)

        raw_pred, _, raw_nf_std = self._srf.predict(
            X_test,
            return_uncertainty=True,
            return_noise_free_uncertainty=True,
        )
        intra_var, inter_var, model_var = self._srf.get_detailed_uncertainty(X_test)

        coef = self._calib_coef
        pred_cal, total_std_cal, nf_std_cal = self._calibrate_with_uncertainty(
            raw_pred, intra_var, inter_var, model_var, raw_nf_std
        )

        return {
            'pred':           pred_cal,
            'total_std':      total_std_cal,
            'noise_free_std': nf_std_cal,
            'intra_var':      (coef ** 2) * intra_var,
            'inter_var':      (coef ** 2) * inter_var,
            'model_var':      self._model_var_cal,   # cached, no extra forward pass
        }

    def get_smoothing_params(self) -> np.ndarray:
        """
        Return the learned smoothing parameters after softplus transform.

        Shape depends on smoothing_mode:
        - 'STE'     → scalar
        - 'STE_PD'  → (n_features,)
        - 'EST'     → (n_estimators,)
        - 'EST_PD'  → (n_estimators, n_features)
        """
        self._check_is_fitted()
        return self._srf.get_smoothing_params()

    @property
    def calibration_coef_(self) -> float:
        """OOB linear calibration slope."""
        self._check_is_fitted()
        return self._calib_coef

    @property
    def calibration_intercept_(self) -> float:
        """OOB linear calibration intercept."""
        self._check_is_fitted()
        return self._calib_intercept

    @property
    def y_train_(self) -> np.ndarray:
        """Training targets seen by fit()."""
        self._check_is_fitted()
        return self._y_train

    def get_effective_kernel(self, X: np.ndarray) -> np.ndarray:
        """
        Effective kernel matrix averaged over trees.

        Returns
        -------
        K : np.ndarray, shape (n_samples, n_train)
            K[j, i] = mean over trees of the kernel weight assigned by the
            smoothed forest to training point i when predicting point j.
            Row j is therefore the (un-normalised) weight distribution over
            training points for test point j.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        prob_matrix = self._srf.get_prob_matrix(X)   # (n_trees, n_samples, n_train)
        return prob_matrix.mean(axis=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_calibrator(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit OOB linear regression calibrator on the training set."""
        # Per-tree smoothed predictions on training data: (n_train, n_trees)
        smoothed_trees = self._srf.smoothed_trees_predict(X_train)

        # Build OOB mask  (n_trees, n_train)
        oob_mask = np.zeros(
            (len(self._rf.oob_samples_indices), X_train.shape[0]), dtype=bool
        )
        for i, idx_list in enumerate(self._rf.oob_samples_indices):
            oob_mask[i, idx_list] = True

        # Extract OOB predictions and matching targets
        smoothed_T = smoothed_trees.T          # (n_trees, n_train)
        oob_preds = smoothed_T[oob_mask]       # (total_oob_samples,)
        oob_y = np.broadcast_to(
            y_train, smoothed_T.shape
        )[oob_mask]                             # (total_oob_samples,)

        lr = LinearRegression(fit_intercept=True, copy_X=True)
        lr.fit(oob_preds.reshape(-1, 1), oob_y.reshape(-1, 1))

        self._calibrator = lr
        self._calib_coef = float(lr.coef_.flatten()[0])
        self._calib_intercept = float(lr.intercept_.flatten()[0])

        # Cache calibrated model variance once (matches experiment pipeline:
        # use the already-computed smoothed_trees mean as training predictions)
        coef = self._calib_coef
        intercept = self._calib_intercept
        train_pred_uncal = smoothed_trees.mean(axis=1)   # (n_train,)
        train_pred_cal = coef * train_pred_uncal + intercept
        self._model_var_cal = float(mean_squared_error(y_train, train_pred_cal))

    def _calibrate_predictions(self, raw_pred: np.ndarray) -> np.ndarray:
        return self._calibrator.predict(
            raw_pred.reshape(-1, 1)
        ).flatten()

    def _calib_model_var(self) -> float:
        """Return the cached calibrated model variance (computed once during fit)."""
        return self._model_var_cal

    def _calibrate_with_uncertainty(
        self,
        raw_pred: np.ndarray,
        intra_var: np.ndarray,
        inter_var: np.ndarray,
        model_var: float,
        raw_nf_std: np.ndarray,
    ):
        """Apply linear calibration to predictions and uncertainty components."""
        coef = self._calib_coef

        # Calibrated point predictions
        pred_cal = self._calibrate_predictions(raw_pred)

        # Calibrated variance components
        intra_cal = (coef ** 2) * intra_var
        inter_cal = (coef ** 2) * inter_var
        model_var_cal = self._calib_model_var()

        # Calibrated noise-free uncertainty
        nf_std_cal = np.abs(coef) * raw_nf_std

        # Calibrated total uncertainty
        total_var_cal = intra_cal + inter_cal + model_var_cal
        total_std_cal = np.sqrt(np.maximum(total_var_cal, 1e-12))

        return pred_cal, total_std_cal, nf_std_cal

    def _check_is_fitted(self):
        if self._srf is None or self._calibrator is None:
            raise RuntimeError(
                "SRFRegressor is not fitted yet. Call fit() first."
            )

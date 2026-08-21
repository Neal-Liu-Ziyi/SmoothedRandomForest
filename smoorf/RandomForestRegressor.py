from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.special import softmax


class ExtendedRandomForest(RandomForestRegressor):
    def __init__(self, **kwargs):
        """
        Extends sklearn's RandomForestRegressor with additional methods for OOB handling.
        """
        super().__init__(**kwargs)

    @staticmethod
    def upgrade(sklearn_rf, X_train=None, y_train=None):
        """
        Convert a sklearn RandomForestRegressor to ExtendedRandomForest.
        
        Parameters:
        -----------
        sklearn_rf : RandomForestRegressor
            A fitted sklearn RandomForestRegressor instance
        X_train, y_train : array-like, optional
            Training data. ONLY needed if you want to use OOB methods:
            - get_oob_score(), get_oob_prob(), count_oob(), get_stacked_oob_mse()
            If not provided, you can still use predict() with return_std=True
            
        Returns: The same object, now as ExtendedRandomForest
        Example:
        >>> rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5)
        >>> rf = rf.fit(X_train, y_train)
        >>> rf = ExtendedRandomForest.upgrade(rf, X_train, y_train)
        >>> rf.get_oob_score()
        >>> rf.get_oob_prob()
        >>> rf.count_oob()
        >>> rf.get_stacked_oob_mse()
        >>> rf.predict(X_test)
        >>> rf.predict(X_test, return_std=True)
        """
        check_is_fitted(sklearn_rf)
        sklearn_rf.__class__ = ExtendedRandomForest
        
        # Optional: store training data for OOB methods
        if X_train is not None and y_train is not None:
            sklearn_rf.X_ = np.asarray(X_train)
            sklearn_rf.y_ = np.asarray(y_train)
            sklearn_rf.n_samples_ = sklearn_rf.X_.shape[0]
            sklearn_rf._reconstruct_bootstrap_indices()
        
        return sklearn_rf
    
    def _reconstruct_bootstrap_indices(self):
        """
        Reconstruct bootstrap sample indices for OOB calculations.
        This is an approximation since we don't have the original indices.
        """
        all_indices = np.arange(self.n_samples_)
        self.used_samples_indices = []
        self.oob_samples_indices = []
        
        for estimator in self.estimators_:
            # Generate the same bootstrap samples as during training
            sample_indices = self._generate_sample_indices(
                np.random.RandomState(estimator.random_state), 
                self.n_samples_
            )
            self.used_samples_indices.append(sample_indices)
            
            oob_indices = np.setdiff1d(all_indices, np.unique(sample_indices), assume_unique=True)
            self.oob_samples_indices.append(oob_indices)

    def get_oob_score(self):
        """
        Computes and returns the OOB (Out-Of-Bag) score using Mean Squared Error.
        """
        check_is_fitted(self)
        if not self.bootstrap:
            raise ValueError("OOB score is not available when bootstrap is False.")
        if not hasattr(self, 'X_') or not hasattr(self, 'y_'):
            raise ValueError("Training data not available. Use ExtendedRandomForest.upgrade(rf, X_train, y_train) to enable OOB methods.")
        return self._compute_oob_score()

    def _compute_oob_score(self):
        """
        Internal method to compute OOB score using MSE.
        """
        # Initialize arrays to accumulate predictions and counts
        oob_predictions = np.zeros(self.n_samples_)
        oob_counts = np.zeros(self.n_samples_)

        for tree, oob_indices in zip(self.estimators_, self._get_oob_samples()):
            if len(oob_indices) == 0:
                continue
            preds = tree.predict(self.X_[oob_indices])
            oob_predictions[oob_indices] += preds
            oob_counts[oob_indices] += 1

        # Consider only samples with at least one OOB prediction
        mask = oob_counts > 0
        if not np.any(mask):
            raise ValueError("No OOB predictions available. Increase the number of estimators or check bootstrap settings.")

        oob_pred_final = oob_predictions[mask] / oob_counts[mask]
        y_true = self.y_[mask]

        self.oob_score_ = mean_squared_error(y_true, oob_pred_final)
        return self.oob_score_

    def count_oob(self):
        """
        Counts how many times each sample was left out (OOB).
        """
        check_is_fitted(self)
        if not hasattr(self, 'X_'):
            raise ValueError("Training data not available. Use ExtendedRandomForest.upgrade(rf, X_train, y_train) to enable OOB methods.")
        all_oob_indices = np.concatenate(self._get_oob_samples())
        oob_counts = np.bincount(all_oob_indices, minlength=self.X_.shape[0])
        return oob_counts

    def get_oob_prob(self):
        """
        Computes and returns the OOB probability distribution using softmax normalization.
        """
        check_is_fitted(self)
        if not self.bootstrap:
            raise ValueError("OOB probability is not available when bootstrap is False.")
        # count_oob() will handle the training data check
        oob_counts = self.count_oob()
        return softmax(oob_counts)

    def get_stacked_oob_mse(self):
        """
        Computes the stacked OOB Mean Squared Error (tree level).
        """
        check_is_fitted(self)
        if not hasattr(self, 'X_') or not hasattr(self, 'y_'):
            raise ValueError("Training data not available. Use ExtendedRandomForest.upgrade(rf, X_train, y_train) to enable OOB methods.")

        all_oob_predictions = []
        all_oob_true = []

        for tree, oob_indices in zip(self.estimators_, self._get_oob_samples()):
            if len(oob_indices) == 0:
                continue
            preds = tree.predict(self.X_[oob_indices])
            all_oob_predictions.append(preds)
            all_oob_true.append(self.y_[oob_indices])

        # Ensure we have OOB predictions
        if len(all_oob_predictions) == 0:
            raise ValueError("No OOB predictions available. Increase the number of estimators or check bootstrap settings.")

        all_oob_predictions = np.concatenate(all_oob_predictions, axis=0)
        all_oob_true = np.concatenate(all_oob_true, axis=0)

        return mean_squared_error(all_oob_true, all_oob_predictions)

    def _get_oob_samples(self):
        """
        Retrieves OOB samples for each tree.
        """
        if not hasattr(self, 'X_'):
            raise ValueError("Training data not available. Use ExtendedRandomForest.upgrade(rf, X_train, y_train) to enable OOB methods.")
            
        all_indices = np.arange(self.X_.shape[0])
        oob_samples = []

        for estimator in self.estimators_:
            bootstrap_indices = self._generate_sample_indices(estimator.random_state, self.X_.shape[0])
            oob_indices = np.setdiff1d(all_indices, bootstrap_indices, assume_unique=True)
            oob_samples.append(oob_indices)

        return oob_samples

    def _generate_sample_indices(self, random_state, n_samples):
        """
        Generates bootstrap sample indices for each tree. Ensures random_state is properly set.
        """
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)  # Convert int to RandomState object
        return random_state.choice(np.arange(n_samples), size=n_samples, replace=True)

    def fit(self, X, y):
        """
        Overrides the fit method to store dataset information and track bootstrap/OOB sample indices.
        """
        super().fit(X, y)
        self.X_ = X
        self.y_ = y
        self.n_samples_ = X.shape[0]

        all_indices = np.arange(self.n_samples_)

        # Track which samples were used for each tree and which were OOB
        self.used_samples_indices = []
        self.oob_samples_indices = []

        for estimator in self.estimators_:
            sample_indices = self._generate_sample_indices(np.random.RandomState(estimator.random_state), self.n_samples_)
            self.used_samples_indices.append(sample_indices)
            
            oob_indices = np.setdiff1d(all_indices, np.unique(sample_indices), assume_unique=True)
            self.oob_samples_indices.append(oob_indices)

        return self

    def predict(self, X, return_std=False):
        """
        Predict using the random forest model. Optionally returns standard deviation for uncertainty estimation.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
        - return_std: bool, whether to return standard deviation of predictions

        Returns:
        - mean_predictions: array, shape (n_samples,)
        - (optional) std_predictions: array, shape (n_samples,)
        """
        check_is_fitted(self)
        X = np.asarray(X)

        # Get predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        mean_predictions = np.mean(predictions, axis=0)

        if return_std:
            std_predictions = np.std(predictions, axis=0)
            return mean_predictions, std_predictions
        else:
            return mean_predictions
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array


class SklearnTreeInfoExtractor:
    def __init__(self, rf, tree_idx, X, y):
        """
        Initialize the SklearnTreeInfoExtractor with a trained RandomForestRegressor or RandomForestClassifier instance.

        Args:
            rf (RandomForestRegressor): A trained random forest object.
            tree_idx (int): Index of the tree in the forest.
            X (np.ndarray): X fit into the random forest.
            y (np.ndarray): y fit into the random forest.
        """
        self.rf = rf
        self.tree_idx = tree_idx
        self.X = X
        self.y = y

        # Get the indices of the samples used in this tree
        self.tree_used_idx = rf.used_samples_indices[tree_idx]
        self.tree_OOB_idx = rf.oob_samples_indices[tree_idx]

        # Extract the tree
        self.tree = rf.estimators_[tree_idx]

        # Initialize the leaf information data
        self.leaf_info_data = []
        self.sample_info_data = []

        # Check if the tree has been fitted
        if not hasattr(self.tree, "tree_"):
            raise ValueError("Input must be a fitted sklearn tree object (with tree_ attribute).")

        # Extract the leaf structure
        self.extract_leaf_structure()
        self.get_sample_info_()

    def extract_leaf_structure(self):
        """
        Extract the leaf structure of the tree.

        Returns:
            None
        """
        tree_ = self.tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold
        children_left = tree_.children_left
        children_right = tree_.children_right
        n_features = tree_.n_features

        # Get the leaf nodes indices
        leaf_nodes = np.where(children_left == -1)[0]

        for leaf_idx, leaf in enumerate(leaf_nodes):
            # Get the path to the leaf node
            path = []
            node = leaf
            while node != 0:
                parent_candidates = np.where((children_left == node) | (children_right == node))[0]
                if len(parent_candidates) != 1:
                    raise ValueError(f"Node {node} has {len(parent_candidates)} parents. Expected 1.")
                parent = parent_candidates[0]
                go_left = (children_left[parent] == node)
                path.append((feature[parent], threshold[parent], go_left))
                node = parent
            path.reverse()

            # Get the bounds for each feature
            bounds = [(-np.inf, np.inf) for _ in range(n_features)]
            for feat_idx, thresh, go_left in path:
                low, high = bounds[feat_idx]
                if go_left:
                    bounds[feat_idx] = (low, min(high, thresh))
                else:
                    bounds[feat_idx] = (max(low, thresh), high)

            # Get the samples in the leaf
            samples_in_leaf = self._get_samples_in_leaf(leaf)

            # Get the samples in-bag and out-of-bag
            obs_idx_inbag = [idx for idx in self.tree_used_idx if idx in samples_in_leaf]
            obs_idx_oob = [idx for idx in self.tree_OOB_idx if idx in samples_in_leaf] # out-of-bag samples in the leaf

            # Get the unique in-bag samples
            unique_obs_idx_inbag = np.unique(obs_idx_inbag).tolist()

            # Calculate the mean y value of the in-bag samples
            if len(unique_obs_idx_inbag) > 0:
                mean_y = np.mean(self.y[unique_obs_idx_inbag])
                std_y = np.std(self.y[unique_obs_idx_inbag])
            else:
                mean_y = None  # No in-bag samples
                std_y = None   # No standard deviation

            # Append the leaf information
            self.leaf_info_data.append({
                'leaf_idx': leaf_idx,
                'bounds': bounds,
                'obs_idx_inbag': obs_idx_inbag,  # Retain duplicates
                'unique_obs_idx_inbag': unique_obs_idx_inbag,  # New key for unique indices
                'obs_idx_oob': obs_idx_oob,
                'mean_y': mean_y,
                'std_y': std_y,  # New field for standard deviation
                'original_node_idx': leaf
            })

    def _get_samples_in_leaf(self, leaf_node):
        """
        Get the samples in a leaf node.

        Args:
            leaf_node (int): Index of the leaf node.

        Returns:
            np.ndarray: Indices of the samples in the leaf node.
        """
        # Get the leaf assignments for all samples by using the apply method
        leaf_assignments = self.tree.apply(self.X)
        return np.where(leaf_assignments == leaf_node)[0]

    def get_sample_info_(self):
        """
        Get the information of each sample in the tree.

        Returns:
            list: A list containing dictionaries with information about each sample.
        """
        # Get the leaf assignments for all samples
        leaf_assignments = self.tree.apply(self.X)

        # Create a mapping from the original node index to the leaf information
        leaf_mapping = {leaf['original_node_idx']: leaf for leaf in self.leaf_info_data}

        sample_info_list = []
        for idx in range(len(self.X)):
            original_node_idx = leaf_assignments[idx]
            leaf_info = leaf_mapping.get(original_node_idx, None)
            if leaf_info is None:
                raise ValueError(f"Sample index {idx} allocated node index {original_node_idx} not found in leaf info.")

            # Check if the sample is used in the tree
            if idx in self.tree_used_idx:
                used_obs = 1  # used in-bag
            elif idx in self.tree_OOB_idx:
                used_obs = 0  # used out-of-bag
            else:
                used_obs = 0  # default to 0

            sample_info = {
                'sample_idx': idx,
                'Xi': self.X[idx],
                'original_node_idx': original_node_idx,
                'leaf_bounds': leaf_info['bounds'],
                'y_value': self.y[idx],
                'used_obs': used_obs,
                'leaf_sample_count': len(leaf_info['unique_obs_idx_inbag'])  # Use unique count
            }

            sample_info_list.append(sample_info)
            self.sample_info_data = sample_info_list

    def get_sample_info(self):
        """
        Get the information of each sample in the tree.

        Returns:
            list: A list containing dictionaries with information about each sample.
        """
        return self.sample_info_data


    def leaf_info(self):
        """
        Get the leaf information of the tree.

        Returns:
            list: A list containing dictionaries with information about each leaf.
        """
        return self.leaf_info_data

    def get_leaf_bounds(self):
        """
        Get the bounds of each leaf node in the tree.
        
        Returns:
            list: A list containing the bounds of each leaf node.
        """
        return np.array([leaf['bounds'] for leaf in self.leaf_info_data])

    def get_y_mean_array(self):
        """
        Get an array of mean y values in leaf nodes, sorted by sequential idx.
        
        Returns:
            np.ndarray: An (l, 1) array of mean y values, ordered by idx from 0 to the last.
        """
        y_means = [leaf['mean_y'] for leaf in self.leaf_info_data]
        return np.array(y_means).reshape(-1, 1)

    def get_y_scaling(self):
        """
        Scaling y by samples in leaf and if used in tree
        """
        sample_info = self.sample_info_data

        leaf_sample_count_vect = np.array([info['leaf_sample_count'] for info in sample_info])
        used_indicator_vect = np.array([info['used_obs'] for info in sample_info])

        # Ensure no division by zero
        leaf_sample_count_vect = np.where(leaf_sample_count_vect == 0, 1e-6, leaf_sample_count_vect)

        # Vectorized scaling
        y_scaled = self.y * used_indicator_vect / leaf_sample_count_vect

        return y_scaled.reshape(-1, 1)
    
    def get_scaling_vects(self):
        sample_info = self.sample_info_data
        leaf_sample_count_vect = np.array([info['leaf_sample_count'] for info in sample_info])
        used_indicator_vect = np.array([info['used_obs'] for info in sample_info])

        # Debug: Check for invalid data that shouldn't exist
        if np.any(leaf_sample_count_vect <= 0):
            zero_indices = np.where(leaf_sample_count_vect <= 0)[0]
            print(f"Warning: Found {len(zero_indices)} samples with leaf_sample_count <= 0")
            print(f"Indices: {zero_indices}")
            print(f"Values: {leaf_sample_count_vect[zero_indices]}")
            # This indicates a bug in tree info extraction, not a normal case to handle
            raise ValueError("leaf_sample_count should never be <= 0. This indicates a bug in tree processing.")

        return (1.0 / leaf_sample_count_vect) * used_indicator_vect


    def get_sample_boundary(self):
        """
        Get the boundaries of each sample based on the leaf it belongs to.
        
        Returns:
            np.ndarray: An array containing the bounds for each sample.
        """
        sample_info = self.sample_info_data
        return np.array([info['leaf_bounds'] for info in sample_info])
    








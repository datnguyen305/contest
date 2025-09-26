import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

class RandomSubspaceRF(BaseEstimator, ClassifierMixin):
    """
    Random Subspace Random Forest classifier
    
    This ensemble method trains multiple Random Forest classifiers 
    on random subsets of features, combining feature bagging with 
    additional feature subspace randomization
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_features=0.8,
                 rf_n_estimators=100,
                 rf_max_depth=None,
                 rf_min_samples_split=2,
                 rf_min_samples_leaf=1,
                 rf_max_features='sqrt',
                 rf_bootstrap=True,
                 rf_oob_score=False,
                 random_state=None,
                 n_jobs=None):
        """
        Initialize Random Subspace Random Forest classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of Random Forest estimators
        max_features : int or float, default=0.8
            Number/fraction of features for each RF estimator
        rf_n_estimators : int, default=100
            Number of trees in each Random Forest
        rf_max_depth : int, default=None
            Maximum depth of trees in Random Forest
        rf_min_samples_split : int, default=2
            Minimum samples required to split
        rf_min_samples_leaf : int, default=1
            Minimum samples required in leaf nodes
        rf_max_features : str, int or float, default='sqrt'
            Number of features to consider for best split in each tree
        rf_bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees
        rf_oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate generalization accuracy
        random_state : int, default=None
            Random state for reproducibility
        n_jobs : int, default=None
            Number of jobs for parallel processing
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_max_features = rf_max_features
        self.rf_bootstrap = rf_bootstrap
        self.rf_oob_score = rf_oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def _get_n_features(self, n_total_features):
        """Calculate number of features to use"""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total_features))
        else:
            raise ValueError("max_features must be int or float")
            
    def fit(self, X, y):
        """
        Fit the Random Subspace Random Forest classifier
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes and feature info
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Calculate number of features per estimator
        n_features_per_estimator = self._get_n_features(self.n_features_in_)
        
        # Initialize estimators and feature subsets
        self.estimators_ = []
        self.feature_subsets_ = []
        
        # Train estimators on random feature subsets
        for i in range(self.n_estimators):
            # Select random feature subset
            feature_indices = np.random.choice(
                self.n_features_in_, 
                size=n_features_per_estimator, 
                replace=False
            )
            
            # Create Random Forest estimator
            rf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                max_features=self.rf_max_features,
                bootstrap=self.rf_bootstrap,
                oob_score=self.rf_oob_score,
                random_state=None if self.random_state is None else self.random_state + i,
                n_jobs=self.n_jobs
            )
            
            # Train on feature subset
            X_subset = X[:, feature_indices]
            rf.fit(X_subset, y)
            
            # Store estimator and feature subset
            self.estimators_.append(rf)
            self.feature_subsets_.append(feature_indices)
            
        return self
        
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Get probability predictions and use majority voting
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
        
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        
        # Collect probability predictions from all estimators
        all_probas = np.zeros((n_samples, self.n_classes_))
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_indices]
            probas = estimator.predict_proba(X_subset)
            
            # Align probabilities with global class order
            estimator_classes = estimator.classes_
            for i, cls in enumerate(estimator_classes):
                class_idx = np.where(self.classes_ == cls)[0][0]
                all_probas[:, class_idx] += probas[:, i]
                
        # Average probabilities
        all_probas /= self.n_estimators
        
        return all_probas
        
    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Predicted class log-probabilities
        """
        probas = self.predict_proba(X)
        return np.log(probas + 1e-15)  # Add small epsilon to avoid log(0)
        
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
            
        Returns:
        --------
        score : float
            Mean accuracy
        """
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    @property
    def feature_importances_(self):
        """
        Get average feature importances from all Random Forest estimators
        
        Returns:
        --------
        importances : ndarray of shape (n_features,)
            Average feature importances
        """
        check_is_fitted(self)
        
        # Initialize importance array
        importances = np.zeros(self.n_features_in_)
        feature_counts = np.zeros(self.n_features_in_)
        
        # Accumulate importances from all estimators
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            rf_importances = estimator.feature_importances_
            importances[feature_indices] += rf_importances
            feature_counts[feature_indices] += 1
            
        # Average by number of times each feature was used
        feature_counts[feature_counts == 0] = 1  # Avoid division by zero
        importances /= feature_counts
        
        # Normalize to sum to 1
        if importances.sum() > 0:
            importances = importances / importances.sum()
            
        return importances
        
    @property
    def oob_score_(self):
        """
        Get average OOB score from all Random Forest estimators (if available)
        
        Returns:
        --------
        oob_score : float
            Average out-of-bag score
        """
        check_is_fitted(self)
        
        if not self.rf_oob_score:
            raise AttributeError("OOB score not available. Set rf_oob_score=True")
            
        oob_scores = []
        for estimator in self.estimators_:
            if hasattr(estimator, 'oob_score_'):
                oob_scores.append(estimator.oob_score_)
                
        if oob_scores:
            return np.mean(oob_scores)
        else:
            raise AttributeError("No OOB scores available from estimators")
            
    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        X_leaves : list of arrays
            Leaf indices for each estimator
        """
        check_is_fitted(self)
        X = check_array(X)
        
        all_leaves = []
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_indices]
            leaves = estimator.apply(X_subset)
            all_leaves.append(leaves)
            
        return all_leaves
        
    def decision_path(self, X):
        """
        Return decision path in the forest
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        paths : list of sparse matrices
            Decision paths for each estimator
        """
        check_is_fitted(self)
        X = check_array(X)
        
        all_paths = []
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_indices]
            paths = estimator.decision_path(X_subset)
            all_paths.append(paths)
            
        return all_paths
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': self.rf_max_depth,
            'rf_min_samples_split': self.rf_min_samples_split,
            'rf_min_samples_leaf': self.rf_min_samples_leaf,
            'rf_max_features': self.rf_max_features,
            'rf_bootstrap': self.rf_bootstrap,
            'rf_oob_score': self.rf_oob_score,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self
        
    def get_estimator_info(self):
        """Get information about the trained estimators"""
        check_is_fitted(self)
        
        feature_usage = np.zeros(self.n_features_in_)
        for feature_indices in self.feature_subsets_:
            feature_usage[feature_indices] += 1
            
        return {
            'n_estimators_': len(self.estimators_),
            'feature_subsets_': self.feature_subsets_,
            'feature_usage_count_': feature_usage,
            'feature_usage_frequency_': feature_usage / self.n_estimators,
            'feature_importances_': self.feature_importances_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_
        }


# Convenience function
def create_rs_rf(n_estimators=10, max_features=0.8, rf_n_estimators=100, random_state=None):
    """Create a Random Subspace Random Forest classifier with sensible defaults"""
    return RandomSubspaceRF(
        n_estimators=n_estimators,
        max_features=max_features,
        rf_n_estimators=rf_n_estimators,
        random_state=random_state
    )
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

class RandomSubspaceKNN(BaseEstimator, ClassifierMixin):
    """
    Random Subspace K-Nearest Neighbors classifier
    
    This ensemble method trains multiple KNN classifiers on random subsets
    of features to improve generalization and reduce overfitting
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_features=0.8,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 random_state=None,
                 n_jobs=None):
        """
        Initialize Random Subspace KNN classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of base KNN estimators
        max_features : int or float, default=0.8
            Number/fraction of features to consider for each estimator
        n_neighbors : int, default=5
            Number of neighbors for KNN
        weights : str or callable, default='uniform'
            Weight function ('uniform', 'distance')
        algorithm : str, default='auto'
            Algorithm to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
        leaf_size : int, default=30
            Leaf size for BallTree or KDTree
        p : int, default=2
            Parameter for Minkowski metric
        metric : str or callable, default='minkowski'
            Distance metric
        random_state : int, default=None
            Random state for reproducibility
        n_jobs : int, default=None
            Number of jobs for parallel processing
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
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
        Fit the Random Subspace KNN classifier
        
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
            
            # Create KNN estimator
            knn = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                n_jobs=self.n_jobs
            )
            
            # Train on feature subset
            X_subset = X[:, feature_indices]
            knn.fit(X_subset, y)
            
            # Store estimator and feature subset
            self.estimators_.append(knn)
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
        
        # Collect predictions from all estimators
        predictions = []
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_indices]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
            
        # Majority voting
        predictions = np.array(predictions).T  # Shape: (n_samples, n_estimators)
        
        final_predictions = []
        for sample_preds in predictions:
            # Count votes for each class
            unique_classes, counts = np.unique(sample_preds, return_counts=True)
            # Predict class with most votes
            most_voted_class = unique_classes[np.argmax(counts)]
            final_predictions.append(most_voted_class)
            
        return np.array(final_predictions)
        
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
        
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """
        Find K-neighbors of a point (using first estimator as representative)
        
        Parameters:
        -----------
        X : array-like of shape (n_queries, n_features), default=None
            Query points. If None, uses training data
        n_neighbors : int, default=None
            Number of neighbors. If None, uses self.n_neighbors
        return_distance : bool, default=True
            Whether to return distances
            
        Returns:
        --------
        distances : ndarray of shape (n_queries, n_neighbors)
            Distances to neighbors (if return_distance=True)
        indices : ndarray of shape (n_queries, n_neighbors)
            Indices of neighbors
        """
        check_is_fitted(self)
        
        if len(self.estimators_) == 0:
            raise ValueError("No estimators fitted")
            
        # Use first estimator as representative
        estimator = self.estimators_[0]
        feature_indices = self.feature_subsets_[0]
        
        if X is not None:
            X = check_array(X)
            X_subset = X[:, feature_indices]
        else:
            X_subset = None
            
        return estimator.kneighbors(
            X=X_subset, 
            n_neighbors=n_neighbors, 
            return_distance=return_distance
        )
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'p': self.p,
            'metric': self.metric,
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
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_
        }


# Convenience function
def create_rs_knn(n_estimators=10, max_features=0.8, n_neighbors=5, random_state=None):
    """Create a Random Subspace KNN classifier with sensible defaults"""
    return RandomSubspaceKNN(
        n_estimators=n_estimators,
        max_features=max_features,
        n_neighbors=n_neighbors,
        random_state=random_state
    )
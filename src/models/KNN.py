import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

class KNN(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors Classifier with additional features
    
    This is a wrapper around sklearn's KNeighborsClassifier with
    additional functionality and customization options
    """
    
    def __init__(self, 
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 normalize=False,
                 n_jobs=None):
        """
        Initialize KNN classifier
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use
        weights : str or callable, default='uniform'
            Weight function ('uniform', 'distance', or callable)
        algorithm : str, default='auto'
            Algorithm to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
        leaf_size : int, default=30
            Leaf size for BallTree or KDTree
        p : int, default=2
            Parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
        metric : str or callable, default='minkowski'
            Distance metric to use
        metric_params : dict, default=None
            Additional keyword arguments for metric function
        normalize : bool, default=False
            Whether to normalize features before training
        n_jobs : int, default=None
            Number of parallel jobs (-1 for all processors)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.normalize = normalize
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        """
        Fit the KNN classifier
        
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
        
        # Initialize scaler if normalization is requested
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_scaled = X
            
        # Create and fit KNN classifier
        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.knn_.fit(X_scaled, y)
        
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
        
        # Apply scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.knn_.predict(X_scaled)
        
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
        
        # Apply scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.knn_.predict_proba(X_scaled)
        
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """
        Find K-neighbors of a point
        
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
        
        if X is not None:
            X = check_array(X)
            # Apply scaling if used during training
            if self.scaler_ is not None:
                X_scaled = self.scaler_.transform(X)
            else:
                X_scaled = X
        else:
            X_scaled = None
            
        return self.knn_.kneighbors(
            X=X_scaled, 
            n_neighbors=n_neighbors, 
            return_distance=return_distance
        )
        
    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        """
        Compute k-neighbors graph
        
        Parameters:
        -----------
        X : array-like of shape (n_queries, n_features), default=None
            Query points. If None, uses training data
        n_neighbors : int, default=None
            Number of neighbors. If None, uses self.n_neighbors
        mode : str, default='connectivity'
            Type of returned matrix ('connectivity' or 'distance')
            
        Returns:
        --------
        A : sparse matrix of shape (n_queries, n_samples_fit)
            Adjacency matrix
        """
        check_is_fitted(self)
        
        if X is not None:
            X = check_array(X)
            # Apply scaling if used during training
            if self.scaler_ is not None:
                X_scaled = self.scaler_.transform(X)
            else:
                X_scaled = X
        else:
            X_scaled = None
            
        return self.knn_.kneighbors_graph(
            X=X_scaled,
            n_neighbors=n_neighbors,
            mode=mode
        )
        
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
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'p': self.p,
            'metric': self.metric,
            'metric_params': self.metric_params,
            'normalize': self.normalize,
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
        """Get information about the trained estimator"""
        check_is_fitted(self)
        
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'p': self.p,
            'normalize': self.normalize,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'effective_metric_': self.knn_.effective_metric_,
            'effective_metric_params_': self.knn_.effective_metric_params_
        }


class KNNRegressor(BaseEstimator, RegressorMixin):
    """
    K-Nearest Neighbors Regressor with additional features
    
    This is a wrapper around sklearn's KNeighborsRegressor with
    additional functionality and customization options
    """
    
    def __init__(self, 
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 normalize=False,
                 n_jobs=None):
        """
        Initialize KNN regressor
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use
        weights : str or callable, default='uniform'
            Weight function ('uniform', 'distance', or callable)
        algorithm : str, default='auto'
            Algorithm to compute nearest neighbors
        leaf_size : int, default=30
            Leaf size for BallTree or KDTree
        p : int, default=2
            Parameter for Minkowski metric
        metric : str or callable, default='minkowski'
            Distance metric to use
        metric_params : dict, default=None
            Additional keyword arguments for metric function
        normalize : bool, default=False
            Whether to normalize features before training
        n_jobs : int, default=None
            Number of parallel jobs
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.normalize = normalize
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        """
        Fit the KNN regressor
        
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
        
        self.n_features_in_ = X.shape[1]
        
        # Initialize scaler if normalization is requested
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_scaled = X
            
        # Create and fit KNN regressor
        self.knn_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
        
        self.knn_.fit(X_scaled, y)
        
        return self
        
    def predict(self, X):
        """
        Predict target values for samples in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.knn_.predict(X_scaled)
        
    def score(self, X, y):
        """
        Return R² score
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True target values
            
        Returns:
        --------
        score : float
            R² score
        """
        check_is_fitted(self)
        return self.knn_.score(
            self.scaler_.transform(X) if self.scaler_ else X, 
            y
        )
        
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find K-neighbors of a point"""
        check_is_fitted(self)
        
        if X is not None:
            X = check_array(X)
            if self.scaler_ is not None:
                X_scaled = self.scaler_.transform(X)
            else:
                X_scaled = X
        else:
            X_scaled = None
            
        return self.knn_.kneighbors(X_scaled, n_neighbors, return_distance)


# Convenience functions
def create_knn_classifier(n_neighbors=5, weights='uniform', normalize=False):
    """Create a KNN classifier with sensible defaults"""
    return KNN(
        n_neighbors=n_neighbors,
        weights=weights,
        normalize=normalize
    )

def create_knn_regressor(n_neighbors=5, weights='uniform', normalize=False):
    """Create a KNN regressor with sensible defaults"""
    return KNNRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        normalize=normalize
    )

def create_distance_weighted_knn(n_neighbors=5, normalize=True):
    """Create a distance-weighted KNN classifier"""
    return KNN(
        n_neighbors=n_neighbors,
        weights='distance',
        normalize=normalize
    )

def create_manhattan_knn(n_neighbors=5, normalize=False):
    """Create a KNN classifier using Manhattan distance"""
    return KNN(
        n_neighbors=n_neighbors,
        p=1,  # Manhattan distance
        normalize=normalize
    )
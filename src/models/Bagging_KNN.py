import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

class BaggingKNN(BaseEstimator, ClassifierMixin):
    """
    Bagging classifier using K-Nearest Neighbors as base estimator
    
    This combines bootstrap aggregating with KNN to reduce variance
    and improve generalization performance
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 knn_n_neighbors=5,
                 knn_weights='uniform',
                 knn_algorithm='auto',
                 knn_metric='minkowski',
                 knn_p=2,
                 normalize=True,
                 random_state=None):
        """
        Initialize Bagging KNN classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of base estimators in the ensemble
        max_samples : int or float, default=1.0
            Number of samples to draw from X to train each base estimator
        max_features : int or float, default=1.0
            Number of features to draw from X to train each base estimator
        bootstrap : bool, default=True
            Whether samples are drawn with replacement
        bootstrap_features : bool, default=False
            Whether features are drawn with replacement
        oob_score : bool, default=False
            Whether to use out-of-bag samples for score estimation
        warm_start : bool, default=False
            Whether to reuse solution of previous call to fit
        n_jobs : int, default=None
            Number of jobs to run in parallel
        knn_n_neighbors : int, default=5
            Number of neighbors for KNN
        knn_weights : str, default='uniform'
            Weight function for KNN ('uniform' or 'distance')
        knn_algorithm : str, default='auto'
            Algorithm for KNN ('auto', 'ball_tree', 'kd_tree', 'brute')
        knn_metric : str, default='minkowski'
            Distance metric for KNN
        knn_p : int, default=2
            Parameter for Minkowski metric
        normalize : bool, default=True
            Whether to normalize features
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights
        self.knn_algorithm = knn_algorithm
        self.knn_metric = knn_metric
        self.knn_p = knn_p
        self.normalize = normalize
        self.random_state = random_state
        
    def _create_base_estimator(self):
        """Create a KNN base estimator"""
        return KNeighborsClassifier(
            n_neighbors=self.knn_n_neighbors,
            weights=self.knn_weights,
            algorithm=self.knn_algorithm,
            metric=self.knn_metric,
            p=self.knn_p
        )
        
    def fit(self, X, y):
        """
        Fit the Bagging KNN classifier
        
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
        
        # Store classes
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Initialize scaler if needed
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
        
        # Create base estimator
        base_estimator = self._create_base_estimator()
        
        # Create Bagging classifier with KNN as base
        self.bagging_ = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.bagging_.fit(X, y)
        
        return self
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        return self.bagging_.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        return self.bagging_.predict_proba(X)
        
    def predict_log_proba(self, X):
        """Predict class log-probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        return self.bagging_.predict_log_proba(X)
        
    def decision_function(self, X):
        """Compute decision function for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        return self.bagging_.decision_function(X)
        
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    @property
    def oob_score_(self):
        """Get out-of-bag score"""
        check_is_fitted(self)
        if not self.oob_score:
            raise AttributeError("oob_score is only available if oob_score=True")
        return self.bagging_.oob_score_
        
    @property
    def oob_decision_function_(self):
        """Get out-of-bag decision function"""
        check_is_fitted(self)
        if not self.oob_score:
            raise AttributeError("oob_decision_function_ is only available if oob_score=True")
        return self.bagging_.oob_decision_function_
        
    @property
    def feature_importances_(self):
        """Get feature importances (uniform for KNN)"""
        check_is_fitted(self)
        # KNN doesn't have traditional feature importances
        return np.ones(self.n_features_in_) / self.n_features_in_
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'oob_score': self.oob_score,
            'warm_start': self.warm_start,
            'n_jobs': self.n_jobs,
            'knn_n_neighbors': self.knn_n_neighbors,
            'knn_weights': self.knn_weights,
            'knn_algorithm': self.knn_algorithm,
            'knn_metric': self.knn_metric,
            'knn_p': self.knn_p,
            'normalize': self.normalize,
            'random_state': self.random_state
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
        
        info = {
            'n_estimators_': len(self.bagging_.estimators_),
            'estimators_': self.bagging_.estimators_,
            'estimators_features_': self.bagging_.estimators_features_,
            'feature_importances_': self.feature_importances_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'normalized': self.scaler_ is not None
        }
        
        # Add OOB score if available
        if self.oob_score:
            info['oob_score_'] = self.oob_score_
            
        return info


# Convenience functions
def create_bagging_knn(n_estimators=10, knn_neighbors=5, bootstrap=True, oob_score=False, random_state=None):
    """Create a Bagging KNN classifier with sensible defaults"""
    return BaggingKNN(
        n_estimators=n_estimators,
        knn_n_neighbors=knn_neighbors,
        bootstrap=bootstrap,
        oob_score=oob_score,
        normalize=True,
        random_state=random_state
    )

def create_oob_bagging_knn(n_estimators=10, knn_neighbors=5, random_state=None):
    """Create a Bagging KNN classifier with OOB scoring enabled"""
    return BaggingKNN(
        n_estimators=n_estimators,
        knn_n_neighbors=knn_neighbors,
        bootstrap=True,
        oob_score=True,
        normalize=True,
        random_state=random_state
    )

def create_feature_bagging_knn(n_estimators=10, knn_neighbors=5, max_features=0.8, random_state=None):
    """Create a Bagging KNN classifier with feature bootstrapping"""
    return BaggingKNN(
        n_estimators=n_estimators,
        knn_n_neighbors=knn_neighbors,
        max_features=max_features,
        bootstrap_features=True,
        normalize=True,
        random_state=random_state
    )
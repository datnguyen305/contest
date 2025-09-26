import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import warnings

class BaggingSVM(BaseEstimator, ClassifierMixin):
    """
    Bagging classifier using Support Vector Machine as base estimator
    
    This combines multiple SVM classifiers trained on different bootstrap samples
    to improve generalization and reduce overfitting
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_gamma='scale',
                 svm_degree=3,
                 svm_probability=True,
                 random_state=None,
                 n_jobs=None):
        """
        Initialize Bagging SVM classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of base estimators in the ensemble
        max_samples : int or float, default=1.0
            Number/fraction of samples to draw for each base estimator
        max_features : int or float, default=1.0
            Number/fraction of features to draw for each base estimator
        bootstrap : bool, default=True
            Whether samples are drawn with replacement
        bootstrap_features : bool, default=False
            Whether features are drawn with replacement
        svm_C : float, default=1.0
            Regularization parameter for SVM
        svm_kernel : str, default='rbf'
            Kernel type for SVM ('linear', 'poly', 'rbf', 'sigmoid')
        svm_gamma : str or float, default='scale'
            Kernel coefficient for SVM
        svm_degree : int, default=3
            Degree for polynomial kernel
        svm_probability : bool, default=True
            Enable probability estimates
        random_state : int, default=None
            Random state for reproducibility
        n_jobs : int, default=None
            Number of jobs for parallel processing
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma
        self.svm_degree = svm_degree
        self.svm_probability = svm_probability
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def _create_base_estimator(self):
        """Create an SVM base estimator"""
        return SVC(
            C=self.svm_C,
            kernel=self.svm_kernel,
            gamma=self.svm_gamma,
            degree=self.svm_degree,
            probability=self.svm_probability,
            random_state=self.random_state
        )
        
    def fit(self, X, y):
        """
        Fit the Bagging SVM classifier
        
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
        
        # Create base estimator
        base_estimator = self._create_base_estimator()
        
        # Create Bagging classifier with SVM as base
        self.bagging_ = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.bagging_.fit(X, y)
        
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
        
        return self.bagging_.predict(X)
        
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
        
        return self.bagging_.predict_proba(X)
        
    def decision_function(self, X):
        """
        Average of decision functions of base classifiers
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        decision : ndarray of shape (n_samples, n_classes) or (n_samples,)
            Decision function values
        """
        check_is_fitted(self)
        X = check_array(X)
        
        return self.bagging_.decision_function(X)
        
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
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'svm_C': self.svm_C,
            'svm_kernel': self.svm_kernel,
            'svm_gamma': self.svm_gamma,
            'svm_degree': self.svm_degree,
            'svm_probability': self.svm_probability,
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
        
        return {
            'n_estimators_': len(self.bagging_.estimators_),
            'estimators_samples_': self.bagging_.estimators_samples_,
            'estimators_features_': self.bagging_.estimators_features_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_
        }


# Convenience function
def create_bagging_svm(n_estimators=10, svm_C=1.0, svm_kernel='rbf', random_state=None):
    """Create a Bagging SVM classifier with sensible defaults"""
    return BaggingSVM(
        n_estimators=n_estimators,
        svm_C=svm_C,
        svm_kernel=svm_kernel,
        random_state=random_state
    )
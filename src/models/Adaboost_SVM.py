import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import warnings

class AdaBoostSVM(BaseEstimator, ClassifierMixin):
    """
    AdaBoost classifier using Support Vector Machine as base estimator
    
    This combines the sequential learning capability of AdaBoost with
    the powerful classification ability of SVM
    """
    
    def __init__(self, 
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME',
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_gamma='scale',
                 svm_degree=3,
                 random_state=None):
        """
        Initialize AdaBoost SVM classifier
        
        Parameters:
        -----------
        n_estimators : int, default=50
            Number of boosting rounds
        learning_rate : float, default=1.0
            Learning rate shrinks contribution of each classifier
        algorithm : str, default='SAMME'
            Boosting algorithm ('SAMME' or 'SAMME.R')
        svm_C : float, default=1.0
            Regularization parameter for SVM
        svm_kernel : str, default='rbf'
            Kernel type for SVM
        svm_gamma : str or float, default='scale'
            Kernel coefficient for SVM
        svm_degree : int, default=3
            Degree for polynomial kernel
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma
        self.svm_degree = svm_degree
        self.random_state = random_state
        
    def _create_base_estimator(self):
        """Create an SVM base estimator"""
        return SVC(
            C=self.svm_C,
            kernel=self.svm_kernel,
            gamma=self.svm_gamma,
            degree=self.svm_degree,
            probability=True,  # Required for SAMME.R
            random_state=self.random_state
        )
        
    def fit(self, X, y):
        """
        Fit the AdaBoost SVM classifier
        
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
        
        # Create AdaBoost classifier with SVM as base
        self.adaboost_ = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.adaboost_.fit(X, y)
        
        return self
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        return self.adaboost_.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        return self.adaboost_.predict_proba(X)
        
    def decision_function(self, X):
        """Compute decision function for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        return self.adaboost_.decision_function(X)
        
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    def staged_predict(self, X):
        """Return staged predictions for X"""
        check_is_fitted(self)
        X = check_array(X)
        for pred in self.adaboost_.staged_predict(X):
            yield pred
            
    def staged_predict_proba(self, X):
        """Return staged class probabilities for X"""
        check_is_fitted(self)
        X = check_array(X)
        for proba in self.adaboost_.staged_predict_proba(X):
            yield proba
            
    def staged_score(self, X, y):
        """Return staged scores for X, y"""
        check_is_fitted(self)
        for pred in self.staged_predict(X):
            yield accuracy_score(y, pred)
            
    @property
    def feature_importances_(self):
        """Get feature importances (if available from base estimators)"""
        check_is_fitted(self)
        # SVM doesn't have feature_importances_, return uniform
        return np.ones(self.n_features_in_) / self.n_features_in_
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'algorithm': self.algorithm,
            'svm_C': self.svm_C,
            'svm_kernel': self.svm_kernel,
            'svm_gamma': self.svm_gamma,
            'svm_degree': self.svm_degree,
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
        
        return {
            'n_estimators_': self.adaboost_.n_estimators_,
            'estimator_weights_': self.adaboost_.estimator_weights_,
            'estimator_errors_': self.adaboost_.estimator_errors_,
            'feature_importances_': self.feature_importances_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_
        }


# Convenience function
def create_adaboost_svm(n_estimators=50, learning_rate=1.0, svm_C=1.0, svm_kernel='rbf', random_state=None):
    """Create an AdaBoost SVM classifier with sensible defaults"""
    return AdaBoostSVM(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        svm_C=svm_C,
        svm_kernel=svm_kernel,
        random_state=random_state
    )
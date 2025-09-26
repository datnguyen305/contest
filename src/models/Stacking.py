import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import warnings

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking Ensemble classifier with customizable base estimators and meta-learner
    
    This implementation provides a flexible stacking ensemble that can combine
    different types of base classifiers with various meta-learning strategies
    """
    
    def __init__(self, 
                 base_estimators=None,
                 meta_learner=None,
                 cv=5,
                 stack_method='auto',
                 n_jobs=None,
                 passthrough=False,
                 verbose=0,
                 random_state=None):
        """
        Initialize Stacking Ensemble classifier
        
        Parameters:
        -----------
        base_estimators : list of (str, estimator) tuples, default=None
            List of base estimators. If None, uses default set
        meta_learner : estimator, default=None
            Meta-learner to combine base estimator predictions. If None, uses LogisticRegression
        cv : int or cross-validation generator, default=5
            Cross-validation strategy for generating meta-features
        stack_method : str, default='auto'
            Method to use for each base estimator ('predict_proba', 'decision_function', 'predict', 'auto')
        n_jobs : int, default=None
            Number of jobs for parallel processing
        passthrough : bool, default=False
            Whether to pass original features to meta-learner
        verbose : int, default=0
            Verbosity level
        random_state : int, default=None
            Random state for reproducibility
        """
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose
        self.random_state = random_state
        
    def _get_default_base_estimators(self):
        """Get default base estimators"""
        return [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('svm', SVC(probability=True, random_state=self.random_state)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('nb', GaussianNB()),
            ('dt', DecisionTreeClassifier(random_state=self.random_state))
        ]
        
    def _get_default_meta_learner(self):
        """Get default meta-learner"""
        return LogisticRegression(random_state=self.random_state, max_iter=1000)
        
    def fit(self, X, y):
        """
        Fit the Stacking Ensemble classifier
        
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
        
        # Set default base estimators if not provided
        if self.base_estimators is None:
            base_estimators = self._get_default_base_estimators()
        else:
            base_estimators = self.base_estimators
            
        # Set default meta-learner if not provided
        if self.meta_learner is None:
            meta_learner = self._get_default_meta_learner()
        else:
            meta_learner = self.meta_learner
            
        # Create StackingClassifier
        self.stacking_ = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=self.cv,
            stack_method=self.stack_method,
            n_jobs=self.n_jobs,
            passthrough=self.passthrough,
            verbose=self.verbose
        )
        
        # Fit the model
        with warnings.catch_warnings():
            if self.verbose == 0:
                warnings.filterwarnings("ignore", category=UserWarning)
            self.stacking_.fit(X, y)
        
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
        
        return self.stacking_.predict(X)
        
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
        
        return self.stacking_.predict_proba(X)
        
    def decision_function(self, X):
        """
        Decision function for samples in X
        
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
        
        return self.stacking_.decision_function(X)
        
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
        
    def transform(self, X):
        """
        Return class labels or probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        X_transform : ndarray of shape (n_samples, n_outputs)
            Transformed samples
        """
        check_is_fitted(self)
        X = check_array(X)
        
        return self.stacking_.transform(X)
        
    def get_base_predictions(self, X):
        """
        Get predictions from all base estimators
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        base_predictions : dict
            Dictionary with base estimator names as keys and their predictions as values
        """
        check_is_fitted(self)
        X = check_array(X)
        
        base_predictions = {}
        
        for name, estimator in self.stacking_.named_estimators_.items():
            if name != 'final_estimator':
                if hasattr(estimator, 'predict_proba'):
                    base_predictions[name] = estimator.predict_proba(X)
                else:
                    base_predictions[name] = estimator.predict(X)
                    
        return base_predictions
        
    @property
    def named_estimators_(self):
        """Access base estimators by name"""
        check_is_fitted(self)
        return self.stacking_.named_estimators_
        
    @property
    def final_estimator_(self):
        """Access the fitted meta-learner"""
        check_is_fitted(self)
        return self.stacking_.final_estimator_
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'base_estimators': self.base_estimators,
            'meta_learner': self.meta_learner,
            'cv': self.cv,
            'stack_method': self.stack_method,
            'n_jobs': self.n_jobs,
            'passthrough': self.passthrough,
            'verbose': self.verbose,
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
        
        base_estimator_info = {}
        for name, estimator in self.named_estimators_.items():
            if name != 'final_estimator':
                info = {
                    'type': type(estimator).__name__,
                    'params': estimator.get_params()
                }
                if hasattr(estimator, 'feature_importances_'):
                    info['feature_importances'] = estimator.feature_importances_
                if hasattr(estimator, 'coef_'):
                    info['coef'] = estimator.coef_
                base_estimator_info[name] = info
                
        return {
            'base_estimators_': base_estimator_info,
            'final_estimator_': {
                'type': type(self.final_estimator_).__name__,
                'params': self.final_estimator_.get_params()
            },
            'stack_method': self.stack_method,
            'cv': self.cv,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_
        }


# Convenience functions for common stacking configurations
def create_basic_stacking(random_state=None):
    """Create a basic stacking ensemble with common classifiers"""
    return StackingEnsemble(random_state=random_state)

def create_stacking_with_svm_meta(random_state=None):
    """Create stacking ensemble with SVM as meta-learner"""
    return StackingEnsemble(
        meta_learner=SVC(probability=True, random_state=random_state),
        random_state=random_state
    )

def create_stacking_with_rf_meta(random_state=None):
    """Create stacking ensemble with Random Forest as meta-learner"""
    return StackingEnsemble(
        meta_learner=RandomForestClassifier(n_estimators=100, random_state=random_state),
        random_state=random_state
    )

def create_custom_stacking(base_estimators, meta_learner, cv=5, random_state=None):
    """Create custom stacking ensemble"""
    return StackingEnsemble(
        base_estimators=base_estimators,
        meta_learner=meta_learner,
        cv=cv,
        random_state=random_state
    )
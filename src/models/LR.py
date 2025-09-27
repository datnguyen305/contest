import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

class LR(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression Classifier with additional features
    
    This is a wrapper around sklearn's LogisticRegression with
    additional functionality and customization options
    """
    
    def __init__(self, 
                 penalty='l2',
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='auto',
                 verbose=0,
                 warm_start=False,
                 normalize=False,
                 l1_ratio=None,
                 n_jobs=None):
        """
        Initialize Logistic Regression classifier
        
        Parameters:
        -----------
        penalty : str, default='l2'
            Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
        dual : bool, default=False
            Dual or primal formulation
        tol : float, default=1e-4
            Tolerance for stopping criteria
        C : float, default=1.0
            Inverse of regularization strength
        fit_intercept : bool, default=True
            Whether to fit intercept term
        intercept_scaling : float, default=1
            Scaling between synthetic feature and other features
        class_weight : dict or str, default=None
            Weights associated with classes ('balanced' or dict)
        random_state : int, default=None
            Random state for reproducibility
        solver : str, default='lbfgs'
            Algorithm to use ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
        max_iter : int, default=1000
            Maximum number of iterations
        multi_class : str, default='auto'
            Multi-class strategy ('auto', 'ovr', 'multinomial')
        verbose : int, default=0
            Verbosity level
        warm_start : bool, default=False
            Reuse solution of previous call to fit
        normalize : bool, default=False
            Whether to normalize features before training
        l1_ratio : float, default=None
            Elastic-Net mixing parameter (for elasticnet penalty)
        n_jobs : int, default=None
            Number of parallel jobs
        """
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.normalize = normalize
        self.l1_ratio = l1_ratio
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        """
        Fit the Logistic Regression classifier
        
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
            
        # Create and fit Logistic Regression
        self.lr_ = LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start=self.warm_start,
            l1_ratio=self.l1_ratio,
            n_jobs=self.n_jobs
        )
        
        with warnings.catch_warnings():
            if self.verbose == 0:
                warnings.filterwarnings("ignore", category=UserWarning)
            self.lr_.fit(X_scaled, y)
        
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
            
        return self.lr_.predict(X_scaled)
        
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
            
        return self.lr_.predict_proba(X_scaled)
        
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
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.lr_.predict_log_proba(X_scaled)
        
    def decision_function(self, X):
        """
        Predict confidence scores for samples in X
        
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
        
        # Apply scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.lr_.decision_function(X_scaled)
        
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
    def coef_(self):
        """
        Get coefficients of the features
        
        Returns:
        --------
        coef : ndarray of shape (n_classes, n_features) or (n_features,)
            Coefficients of the features
        """
        check_is_fitted(self)
        return self.lr_.coef_
        
    @property
    def intercept_(self):
        """
        Get intercept terms
        
        Returns:
        --------
        intercept : ndarray of shape (n_classes,) or scalar
            Intercept terms
        """
        check_is_fitted(self)
        return self.lr_.intercept_
        
    @property
    def n_iter_(self):
        """
        Get actual number of iterations for all classes
        
        Returns:
        --------
        n_iter : ndarray of shape (n_classes,) or scalar
            Number of iterations
        """
        check_is_fitted(self)
        return self.lr_.n_iter_
        
    def get_feature_importance(self):
        """
        Get feature importance based on absolute coefficient values
        
        Returns:
        --------
        importance : ndarray of shape (n_features,)
            Feature importance scores
        """
        check_is_fitted(self)
        
        coef = self.coef_
        
        # Handle multi-class case
        if coef.ndim > 1:
            # Average absolute coefficients across classes
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
            
        # Normalize to sum to 1
        if importance.sum() > 0:
            importance = importance / importance.sum()
            
        return importance
        
    def get_regularization_path(self, X, y, Cs=10, cv=None, scoring=None):
        """
        Get regularization path for different C values
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        Cs : int or array-like, default=10
            C values to test
        cv : int or cross-validation generator, default=None
            Cross-validation strategy
        scoring : str, default=None
            Scoring metric
            
        Returns:
        --------
        path_info : dict
            Dictionary containing path information
        """
        from sklearn.linear_model import LogisticRegressionCV
        
        # Apply scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        lr_cv = LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            scoring=scoring,
            solver=self.solver,
            penalty=self.penalty,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter
        )
        
        lr_cv.fit(X_scaled, y)
        
        return {
            'Cs_': lr_cv.Cs_,
            'scores_': lr_cv.scores_,
            'C_': lr_cv.C_,
            'coefs_paths_': lr_cv.coefs_paths_
        }
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'penalty': self.penalty,
            'dual': self.dual,
            'tol': self.tol,
            'C': self.C,
            'fit_intercept': self.fit_intercept,
            'intercept_scaling': self.intercept_scaling,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'multi_class': self.multi_class,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'normalize': self.normalize,
            'l1_ratio': self.l1_ratio,
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
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'multi_class': self.multi_class,
            'normalize': self.normalize,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'n_iter_': self.n_iter_,
            'feature_importance_': self.get_feature_importance()
        }


# Convenience functions
def create_logistic_regression(C=1.0, penalty='l2', normalize=False, random_state=None):
    """Create a Logistic Regression classifier with sensible defaults"""
    return LR(
        C=C,
        penalty=penalty,
        normalize=normalize,
        random_state=random_state
    )

def create_l1_logistic_regression(C=1.0, normalize=True, random_state=None):
    """Create a L1-regularized Logistic Regression classifier"""
    return LR(
        C=C,
        penalty='l1',
        solver='liblinear',  # liblinear supports L1
        normalize=normalize,
        random_state=random_state
    )

def create_elastic_net_logistic_regression(C=1.0, l1_ratio=0.5, normalize=True, random_state=None):
    """Create an Elastic Net Logistic Regression classifier"""
    return LR(
        C=C,
        penalty='elasticnet',
        l1_ratio=l1_ratio,
        solver='saga',  # saga supports elasticnet
        normalize=normalize,
        random_state=random_state
    )

def create_balanced_logistic_regression(C=1.0, penalty='l2', normalize=False, random_state=None):
    """Create a balanced Logistic Regression classifier"""
    return LR(
        C=C,
        penalty=penalty,
        class_weight='balanced',
        normalize=normalize,
        random_state=random_state
    )
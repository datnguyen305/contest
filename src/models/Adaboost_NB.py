import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

class AdaBoostNB(BaseEstimator, ClassifierMixin):
    """
    AdaBoost classifier using Naive Bayes as base estimator
    
    This combines the sequential learning capability of AdaBoost with
    the probabilistic classification of Naive Bayes
    """
    
    def __init__(self, 
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME.R',
                 nb_type='gaussian',
                 alpha=1.0,
                 fit_prior=True,
                 normalize=False,
                 random_state=None):
        """
        Initialize AdaBoost Naive Bayes classifier
        
        Parameters:
        -----------
        n_estimators : int, default=50
            Number of boosting rounds
        learning_rate : float, default=1.0
            Learning rate shrinks contribution of each classifier
        algorithm : str, default='SAMME.R'
            Boosting algorithm ('SAMME' or 'SAMME.R')
        nb_type : str, default='gaussian'
            Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
        alpha : float, default=1.0
            Smoothing parameter for Multinomial and Bernoulli NB
        fit_prior : bool, default=True
            Whether to learn class prior probabilities
        normalize : bool, default=False
            Whether to normalize features (useful for Gaussian NB)
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.nb_type = nb_type.lower()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.normalize = normalize
        self.random_state = random_state
        
    def _create_base_estimator(self):
        """Create a Naive Bayes base estimator"""
        if self.nb_type == 'gaussian':
            return GaussianNB(priors=None if self.fit_prior else None)
        elif self.nb_type == 'multinomial':
            return MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
        elif self.nb_type == 'bernoulli':
            return BernoulliNB(alpha=self.alpha, fit_prior=self.fit_prior)
        else:
            raise ValueError(f"Unknown nb_type: {self.nb_type}. Use 'gaussian', 'multinomial', or 'bernoulli'")
        
    def fit(self, X, y):
        """
        Fit the AdaBoost Naive Bayes classifier
        
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
        
        # Handle negative values for MultinomialNB
        if self.nb_type == 'multinomial' and np.any(X < 0):
            if not self.normalize:
                warnings.warn("MultinomialNB requires non-negative features. Consider setting normalize=True")
                # Shift to make all values non-negative
                X = X - X.min() + 1e-8
            
        # Create base estimator
        base_estimator = self._create_base_estimator()
        
        # Create AdaBoost classifier with Naive Bayes as base
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
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        # Handle negative values for MultinomialNB
        if self.nb_type == 'multinomial' and np.any(X < 0):
            if self.scaler_ is None:
                X = X - X.min() + 1e-8
                
        return self.adaboost_.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        # Handle negative values for MultinomialNB
        if self.nb_type == 'multinomial' and np.any(X < 0):
            if self.scaler_ is None:
                X = X - X.min() + 1e-8
                
        return self.adaboost_.predict_proba(X)
        
    def predict_log_proba(self, X):
        """Predict class log-probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        # Handle negative values for MultinomialNB
        if self.nb_type == 'multinomial' and np.any(X < 0):
            if self.scaler_ is None:
                X = X - X.min() + 1e-8
                
        return self.adaboost_.predict_log_proba(X)
        
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    def staged_predict(self, X):
        """Return staged predictions for X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        # Handle negative values for MultinomialNB
        if self.nb_type == 'multinomial' and np.any(X < 0):
            if self.scaler_ is None:
                X = X - X.min() + 1e-8
                
        for pred in self.adaboost_.staged_predict(X):
            yield pred
            
    def staged_predict_proba(self, X):
        """Return staged class probabilities for X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        # Handle negative values for MultinomialNB
        if self.nb_type == 'multinomial' and np.any(X < 0):
            if self.scaler_ is None:
                X = X - X.min() + 1e-8
                
        for proba in self.adaboost_.staged_predict_proba(X):
            yield proba
            
    def staged_score(self, X, y):
        """Return staged scores for X, y"""
        check_is_fitted(self)
        for pred in self.staged_predict(X):
            yield accuracy_score(y, pred)
            
    @property
    def feature_importances_(self):
        """Get feature importances (uniform for Naive Bayes)"""
        check_is_fitted(self)
        # Naive Bayes doesn't have traditional feature importances
        return np.ones(self.n_features_in_) / self.n_features_in_
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'algorithm': self.algorithm,
            'nb_type': self.nb_type,
            'alpha': self.alpha,
            'fit_prior': self.fit_prior,
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
        
        return {
            'n_estimators_': self.adaboost_.n_estimators_,
            'estimator_weights_': self.adaboost_.estimator_weights_,
            'estimator_errors_': self.adaboost_.estimator_errors_,
            'feature_importances_': self.feature_importances_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'nb_type': self.nb_type,
            'normalized': self.scaler_ is not None
        }


# Convenience functions
def create_adaboost_gaussian_nb(n_estimators=50, learning_rate=1.0, normalize=True, random_state=None):
    """Create an AdaBoost Gaussian Naive Bayes classifier"""
    return AdaBoostNB(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        nb_type='gaussian',
        normalize=normalize,
        random_state=random_state
    )

def create_adaboost_multinomial_nb(n_estimators=50, learning_rate=1.0, alpha=1.0, random_state=None):
    """Create an AdaBoost Multinomial Naive Bayes classifier"""
    return AdaBoostNB(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        nb_type='multinomial',
        alpha=alpha,
        normalize=True,  # Usually needed for multinomial
        random_state=random_state
    )

def create_adaboost_bernoulli_nb(n_estimators=50, learning_rate=1.0, alpha=1.0, random_state=None):
    """Create an AdaBoost Bernoulli Naive Bayes classifier"""
    return AdaBoostNB(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        nb_type='bernoulli',
        alpha=alpha,
        random_state=random_state
    )
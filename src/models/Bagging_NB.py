import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import warnings

class BaggingNB(BaseEstimator, ClassifierMixin):
    """
    Bagging classifier using Naive Bayes as base estimator
    
    This combines multiple Naive Bayes classifiers trained on different bootstrap samples
    to improve stability and reduce variance
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 nb_type='gaussian',
                 nb_alpha=1.0,
                 nb_var_smoothing=1e-9,
                 random_state=None,
                 n_jobs=None):
        """
        Initialize Bagging Naive Bayes classifier
        
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
        nb_type : str, default='gaussian'
            Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
        nb_alpha : float, default=1.0
            Additive smoothing parameter (for Multinomial/Bernoulli NB)
        nb_var_smoothing : float, default=1e-9
            Portion of largest variance added to variances (for Gaussian NB)
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
        self.nb_type = nb_type
        self.nb_alpha = nb_alpha
        self.nb_var_smoothing = nb_var_smoothing
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def _create_base_estimator(self):
        """Create a Naive Bayes base estimator"""
        if self.nb_type == 'gaussian':
            return GaussianNB(var_smoothing=self.nb_var_smoothing)
        elif self.nb_type == 'multinomial':
            return MultinomialNB(alpha=self.nb_alpha)
        elif self.nb_type == 'bernoulli':
            return BernoulliNB(alpha=self.nb_alpha)
        else:
            raise ValueError("nb_type must be 'gaussian', 'multinomial', or 'bernoulli'")
        
    def fit(self, X, y):
        """
        Fit the Bagging Naive Bayes classifier
        
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
        
        # For MultinomialNB, ensure non-negative features
        if self.nb_type == 'multinomial' and np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")
            
        # Store classes
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Create base estimator
        base_estimator = self._create_base_estimator()
        
        # Create Bagging classifier with Naive Bayes as base
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
        
        # Average log probabilities from all estimators
        log_probas = []
        for estimator in self.bagging_.estimators_:
            log_probas.append(estimator.predict_log_proba(X))
        
        return np.mean(log_probas, axis=0)
        
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
    def feature_log_prob_(self):
        """
        Get average feature log probabilities from all estimators
        (Only for MultinomialNB and BernoulliNB)
        """
        check_is_fitted(self)
        
        if self.nb_type == 'gaussian':
            raise AttributeError("feature_log_prob_ not available for GaussianNB")
            
        feature_log_probs = []
        for estimator in self.bagging_.estimators_:
            if hasattr(estimator, 'feature_log_prob_'):
                feature_log_probs.append(estimator.feature_log_prob_)
                
        if feature_log_probs:
            return np.mean(feature_log_probs, axis=0)
        else:
            return None
            
    @property
    def class_log_prior_(self):
        """Get average class log priors from all estimators"""
        check_is_fitted(self)
        
        class_log_priors = []
        for estimator in self.bagging_.estimators_:
            if hasattr(estimator, 'class_log_prior_'):
                class_log_priors.append(estimator.class_log_prior_)
                
        if class_log_priors:
            return np.mean(class_log_priors, axis=0)
        else:
            return None
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'nb_type': self.nb_type,
            'nb_alpha': self.nb_alpha,
            'nb_var_smoothing': self.nb_var_smoothing,
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
            'nb_type': self.nb_type,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_
        }


# Convenience functions
def create_bagging_gaussian_nb(n_estimators=10, var_smoothing=1e-9, random_state=None):
    """Create a Bagging Gaussian Naive Bayes classifier"""
    return BaggingNB(
        n_estimators=n_estimators,
        nb_type='gaussian',
        nb_var_smoothing=var_smoothing,
        random_state=random_state
    )

def create_bagging_multinomial_nb(n_estimators=10, alpha=1.0, random_state=None):
    """Create a Bagging Multinomial Naive Bayes classifier"""
    return BaggingNB(
        n_estimators=n_estimators,
        nb_type='multinomial',
        nb_alpha=alpha,
        random_state=random_state
    )

def create_bagging_bernoulli_nb(n_estimators=10, alpha=1.0, random_state=None):
    """Create a Bagging Bernoulli Naive Bayes classifier"""
    return BaggingNB(
        n_estimators=n_estimators,
        nb_type='bernoulli',
        nb_alpha=alpha,
        random_state=random_state
    )
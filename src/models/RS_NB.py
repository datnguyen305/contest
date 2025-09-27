import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats

class RandomSubspaceNB(BaseEstimator, ClassifierMixin):
    """
    Random Subspace Method using Naive Bayes as base estimator
    
    This method trains multiple Naive Bayes classifiers on different random 
    subsets of features to reduce overfitting and improve generalization
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_features=0.8,
                 nb_type='gaussian',
                 alpha=1.0,
                 fit_prior=True,
                 normalize=False,
                 voting='soft',
                 n_jobs=None,
                 random_state=None):
        """
        Initialize Random Subspace Naive Bayes classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of Naive Bayes estimators in the ensemble
        max_features : int or float, default=0.8
            Number of features to draw for each base estimator
        nb_type : str, default='gaussian'
            Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
        alpha : float, default=1.0
            Smoothing parameter for Multinomial and Bernoulli NB
        fit_prior : bool, default=True
            Whether to learn class prior probabilities
        normalize : bool, default=False
            Whether to normalize features (useful for Gaussian NB)
        voting : str, default='soft'
            Voting method ('hard' or 'soft')
        n_jobs : int, default=None
            Number of jobs to run in parallel
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.nb_type = nb_type.lower()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.normalize = normalize
        self.voting = voting
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def _get_n_features_to_select(self, n_features_total):
        """Calculate number of features to select for each estimator"""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features_total)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features_total))
        else:
            raise ValueError("max_features must be int or float")
            
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
        Fit the Random Subspace Naive Bayes classifier
        
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
                self.feature_shift_ = X.min(axis=0)
                X = X - self.feature_shift_ + 1e-8
            else:
                self.feature_shift_ = None
        else:
            self.feature_shift_ = None
        
        # Calculate number of features to select
        n_features_to_select = self._get_n_features_to_select(self.n_features_in_)
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize storage for estimators and feature subsets
        self.estimators_ = []
        self.feature_subsets_ = []
        
        # Train estimators on random feature subsets
        for i in range(self.n_estimators):
            # Select random feature subset
            feature_subset = rng.choice(
                self.n_features_in_, 
                size=n_features_to_select, 
                replace=False
            )
            feature_subset = np.sort(feature_subset)
            
            # Extract features for this subset
            X_subset = X[:, feature_subset]
            
            # Create and train estimator
            estimator = self._create_base_estimator()
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                estimator.fit(X_subset, y)
            
            # Store estimator and feature subset
            self.estimators_.append(estimator)
            self.feature_subsets_.append(feature_subset)
        
        return self
        
    def _preprocess_X(self, X):
        """Preprocess input data with scaling and shifting if needed"""
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        # Handle negative values for MultinomialNB
        if self.feature_shift_ is not None:
            X = X - self.feature_shift_ + 1e-8
            
        return X
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Preprocess data
        X = self._preprocess_X(X)
        
        # Get predictions from all estimators
        predictions = []
        
        for estimator, feature_subset in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_subset]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        
        # Aggregate predictions
        if self.voting == 'hard':
            # Hard voting
            final_predictions = []
            for i in range(len(X)):
                vote_counts = np.bincount(predictions[i], minlength=len(self.classes_))
                final_predictions.append(self.classes_[np.argmax(vote_counts)])
            return np.array(final_predictions)
        else:
            # Soft voting (use predict_proba)
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Preprocess data
        X = self._preprocess_X(X)
        
        # Get probability predictions from all estimators
        probas = []
        
        for estimator, feature_subset in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_subset]
            proba = estimator.predict_proba(X_subset)
            probas.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
        
    def predict_log_proba(self, X):
        """Predict class log-probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Preprocess data
        X = self._preprocess_X(X)
        
        # Get log probability predictions from all estimators
        log_probas = []
        
        for estimator, feature_subset in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_subset]
            log_proba = estimator.predict_log_proba(X_subset)
            log_probas.append(log_proba)
        
        # Average log probabilities in log space (using logsumexp for numerical stability)
        from scipy.special import logsumexp
        
        # Convert to regular probabilities, average, then back to log
        probas = [np.exp(lp) for lp in log_probas]
        avg_proba = np.mean(probas, axis=0)
        avg_log_proba = np.log(avg_proba + 1e-15)  # Add small epsilon for numerical stability
        
        return avg_log_proba
        
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    @property
    def feature_importances_(self):
        """Get feature importances based on selection frequency"""
        check_is_fitted(self)
        
        # Count how many times each feature was selected
        feature_counts = np.zeros(self.n_features_in_)
        
        for feature_subset in self.feature_subsets_:
            feature_counts[feature_subset] += 1
            
        # Normalize by number of estimators
        feature_importances = feature_counts / self.n_estimators
        
        return feature_importances
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'nb_type': self.nb_type,
            'alpha': self.alpha,
            'fit_prior': self.fit_prior,
            'normalize': self.normalize,
            'voting': self.voting,
            'n_jobs': self.n_jobs,
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
            'n_estimators_': len(self.estimators_),
            'estimators_': self.estimators_,
            'feature_subsets_': self.feature_subsets_,
            'feature_importances_': self.feature_importances_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'normalized': self.scaler_ is not None,
            'nb_type': self.nb_type,
            'voting': self.voting,
            'avg_features_per_estimator': np.mean([len(fs) for fs in self.feature_subsets_])
        }


# Convenience functions
def create_random_subspace_gaussian_nb(n_estimators=10, max_features=0.8, normalize=True, random_state=None):
    """Create a Random Subspace Gaussian Naive Bayes classifier"""
    return RandomSubspaceNB(
        n_estimators=n_estimators,
        max_features=max_features,
        nb_type='gaussian',
        normalize=normalize,
        voting='soft',
        random_state=random_state
    )

def create_random_subspace_multinomial_nb(n_estimators=10, max_features=0.8, alpha=1.0, random_state=None):
    """Create a Random Subspace Multinomial Naive Bayes classifier"""
    return RandomSubspaceNB(
        n_estimators=n_estimators,
        max_features=max_features,
        nb_type='multinomial',
        alpha=alpha,
        normalize=True,  # Usually needed for multinomial
        voting='soft',
        random_state=random_state
    )

def create_random_subspace_bernoulli_nb(n_estimators=10, max_features=0.8, alpha=1.0, random_state=None):
    """Create a Random Subspace Bernoulli Naive Bayes classifier"""
    return RandomSubspaceNB(
        n_estimators=n_estimators,
        max_features=max_features,
        nb_type='bernoulli',
        alpha=alpha,
        voting='soft',
        random_state=random_state
    )

def create_hard_voting_rs_nb(n_estimators=10, max_features=0.8, nb_type='gaussian', random_state=None):
    """Create a Random Subspace Naive Bayes classifier with hard voting"""
    return RandomSubspaceNB(
        n_estimators=n_estimators,
        max_features=max_features,
        nb_type=nb_type,
        voting='hard',
        normalize=(nb_type == 'gaussian'),
        random_state=random_state
    )
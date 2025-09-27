import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
from scipy import stats

class RandomSubspaceSVM(BaseEstimator, ClassifierMixin):
    """
    Random Subspace Method using Support Vector Machine as base estimator
    
    This method trains multiple SVM classifiers on different random subsets
    of features to reduce overfitting and improve generalization
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_features=0.8,
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_gamma='scale',
                 svm_degree=3,
                 normalize=True,
                 voting='hard',
                 n_jobs=None,
                 random_state=None):
        """
        Initialize Random Subspace SVM classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of SVM estimators in the ensemble
        max_features : int or float, default=0.8
            Number of features to draw for each base estimator
        svm_C : float, default=1.0
            Regularization parameter for SVM
        svm_kernel : str, default='rbf'
            Kernel type for SVM
        svm_gamma : str or float, default='scale'
            Kernel coefficient for SVM
        svm_degree : int, default=3
            Degree for polynomial kernel
        normalize : bool, default=True
            Whether to normalize features
        voting : str, default='hard'
            Voting method ('hard' or 'soft')
        n_jobs : int, default=None
            Number of jobs to run in parallel
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma
        self.svm_degree = svm_degree
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
        """Create an SVM base estimator"""
        probability = self.voting == 'soft'
        return SVC(
            C=self.svm_C,
            kernel=self.svm_kernel,
            gamma=self.svm_gamma,
            degree=self.svm_degree,
            probability=probability,
            random_state=self.random_state
        )
        
    def fit(self, X, y):
        """
        Fit the Random Subspace SVM classifier
        
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
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        # Get predictions from all estimators
        predictions = []
        
        for estimator, feature_subset in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_subset]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        
        # Aggregate predictions using majority voting
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
        
        if self.voting != 'soft':
            raise AttributeError("predict_proba is only available when voting='soft'")
        
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        # Get probability predictions from all estimators
        probas = []
        
        for estimator, feature_subset in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_subset]
            proba = estimator.predict_proba(X_subset)
            probas.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
        
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
            'svm_C': self.svm_C,
            'svm_kernel': self.svm_kernel,
            'svm_gamma': self.svm_gamma,
            'svm_degree': self.svm_degree,
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
            'voting': self.voting,
            'avg_features_per_estimator': np.mean([len(fs) for fs in self.feature_subsets_])
        }


# Convenience functions
def create_random_subspace_svm(n_estimators=10, max_features=0.8, svm_C=1.0, voting='hard', random_state=None):
    """Create a Random Subspace SVM classifier with sensible defaults"""
    return RandomSubspaceSVM(
        n_estimators=n_estimators,
        max_features=max_features,
        svm_C=svm_C,
        voting=voting,
        normalize=True,
        random_state=random_state
    )

def create_soft_voting_rs_svm(n_estimators=10, max_features=0.8, svm_C=1.0, random_state=None):
    """Create a Random Subspace SVM classifier with soft voting"""
    return RandomSubspaceSVM(
        n_estimators=n_estimators,
        max_features=max_features,
        svm_C=svm_C,
        voting='soft',
        normalize=True,
        random_state=random_state
    )

def create_linear_rs_svm(n_estimators=10, max_features=0.8, svm_C=1.0, random_state=None):
    """Create a Random Subspace SVM classifier with linear kernel"""
    return RandomSubspaceSVM(
        n_estimators=n_estimators,
        max_features=max_features,
        svm_C=svm_C,
        svm_kernel='linear',
        normalize=True,
        random_state=random_state
    )

def create_polynomial_rs_svm(n_estimators=10, max_features=0.8, svm_C=1.0, degree=3, random_state=None):
    """Create a Random Subspace SVM classifier with polynomial kernel"""
    return RandomSubspaceSVM(
        n_estimators=n_estimators,
        max_features=max_features,
        svm_C=svm_C,
        svm_kernel='poly',
        svm_degree=degree,
        normalize=True,
        random_state=random_state
    )
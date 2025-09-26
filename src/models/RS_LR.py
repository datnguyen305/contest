import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

class RandomSubspaceLR(BaseEstimator, ClassifierMixin):
    """
    Random Subspace Logistic Regression classifier
    
    This ensemble method trains multiple Logistic Regression classifiers 
    on random subsets of features to improve generalization
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_features=0.8,
                 penalty='l2',
                 C=1.0,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='auto',
                 class_weight=None,
                 random_state=None,
                 n_jobs=None):
        """
        Initialize Random Subspace Logistic Regression classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of base LR estimators
        max_features : int or float, default=0.8
            Number/fraction of features to consider for each estimator
        penalty : str, default='l2'
            Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
        C : float, default=1.0
            Inverse of regularization strength
        solver : str, default='lbfgs'
            Algorithm to use in optimization problem
        max_iter : int, default=1000
            Maximum number of iterations
        multi_class : str, default='auto'
            Multi-class strategy ('auto', 'ovr', 'multinomial')
        class_weight : dict or str, default=None
            Weights associated with classes
        random_state : int, default=None
            Random state for reproducibility
        n_jobs : int, default=None
            Number of jobs for parallel processing
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def _get_n_features(self, n_total_features):
        """Calculate number of features to use"""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total_features))
        else:
            raise ValueError("max_features must be int or float")
            
    def fit(self, X, y):
        """
        Fit the Random Subspace Logistic Regression classifier
        
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
        
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Calculate number of features per estimator
        n_features_per_estimator = self._get_n_features(self.n_features_in_)
        
        # Initialize estimators and feature subsets
        self.estimators_ = []
        self.feature_subsets_ = []
        
        # Train estimators on random feature subsets
        for i in range(self.n_estimators):
            # Select random feature subset
            feature_indices = np.random.choice(
                self.n_features_in_, 
                size=n_features_per_estimator, 
                replace=False
            )
            
            # Create Logistic Regression estimator
            lr = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                multi_class=self.multi_class,
                class_weight=self.class_weight,
                random_state=None if self.random_state is None else self.random_state + i,
                n_jobs=self.n_jobs
            )
            
            # Train on feature subset
            X_subset = X[:, feature_indices]
            lr.fit(X_subset, y)
            
            # Store estimator and feature subset
            self.estimators_.append(lr)
            self.feature_subsets_.append(feature_indices)
            
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
        
        # Get probability predictions and use majority voting
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
        
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
        
        n_samples = X.shape[0]
        
        # Collect probability predictions from all estimators
        all_probas = np.zeros((n_samples, self.n_classes_))
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_indices]
            probas = estimator.predict_proba(X_subset)
            
            # Align probabilities with global class order
            estimator_classes = estimator.classes_
            for i, cls in enumerate(estimator_classes):
                class_idx = np.where(self.classes_ == cls)[0][0]
                all_probas[:, class_idx] += probas[:, i]
                
        # Average probabilities
        all_probas /= self.n_estimators
        
        return all_probas
        
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
        probas = self.predict_proba(X)
        return np.log(probas + 1e-15)  # Add small epsilon to avoid log(0)
        
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
        
        n_samples = X.shape[0]
        
        # For binary classification, return 1D array
        if self.n_classes_ == 2:
            all_decisions = np.zeros(n_samples)
        else:
            all_decisions = np.zeros((n_samples, self.n_classes_))
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            X_subset = X[:, feature_indices]
            decisions = estimator.decision_function(X_subset)
            
            if self.n_classes_ == 2:
                all_decisions += decisions
            else:
                # Align decisions with global class order
                estimator_classes = estimator.classes_
                for i, cls in enumerate(estimator_classes):
                    class_idx = np.where(self.classes_ == cls)[0][0]
                    all_decisions[:, class_idx] += decisions[:, i]
                    
        # Average decisions
        all_decisions /= self.n_estimators
        
        return all_decisions
        
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
        Get average coefficients from all estimators
        
        Returns:
        --------
        coef : ndarray of shape (n_classes, n_features) or (n_features,)
            Average coefficients
        """
        check_is_fitted(self)
        
        if self.n_classes_ == 2:
            avg_coef = np.zeros(self.n_features_in_)
        else:
            avg_coef = np.zeros((self.n_classes_, self.n_features_in_))
            
        feature_counts = np.zeros(self.n_features_in_)
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            if self.n_classes_ == 2:
                avg_coef[feature_indices] += estimator.coef_[0]
            else:
                avg_coef[:, feature_indices] += estimator.coef_
            feature_counts[feature_indices] += 1
            
        # Average by number of times each feature was used
        feature_counts[feature_counts == 0] = 1  # Avoid division by zero
        
        if self.n_classes_ == 2:
            avg_coef /= feature_counts
        else:
            avg_coef /= feature_counts[np.newaxis, :]
            
        return avg_coef
        
    @property
    def intercept_(self):
        """Get average intercepts from all estimators"""
        check_is_fitted(self)
        
        intercepts = [estimator.intercept_ for estimator in self.estimators_]
        return np.mean(intercepts, axis=0)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'multi_class': self.multi_class,
            'class_weight': self.class_weight,
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
        
        feature_usage = np.zeros(self.n_features_in_)
        for feature_indices in self.feature_subsets_:
            feature_usage[feature_indices] += 1
            
        return {
            'n_estimators_': len(self.estimators_),
            'feature_subsets_': self.feature_subsets_,
            'feature_usage_count_': feature_usage,
            'feature_usage_frequency_': feature_usage / self.n_estimators,
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_
        }


# Convenience function
def create_rs_lr(n_estimators=10, max_features=0.8, C=1.0, random_state=None):
    """Create a Random Subspace Logistic Regression classifier with sensible defaults"""
    return RandomSubspaceLR(
        n_estimators=n_estimators,
        max_features=max_features,
        C=C,
        random_state=random_state
    )
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import warnings

class AdaBoostRandomForest(BaseEstimator, ClassifierMixin):
    """
    AdaBoost classifier using Random Forest as base estimator
    
    This combines the ensemble power of Random Forest with the sequential
    learning capability of AdaBoost
    """
    
    def __init__(self, 
                 n_estimators=50,
                 rf_n_estimators=10,
                 rf_max_depth=3,
                 rf_min_samples_split=2,
                 rf_min_samples_leaf=1,
                 rf_max_features='sqrt',
                 learning_rate=1.0,
                 algorithm='SAMME',
                 random_state=None):
        """
        Initialize AdaBoost with Random Forest base estimators
        
        Parameters:
        -----------
        n_estimators : int, default=50
            Number of boosting rounds (Random Forest estimators)
        rf_n_estimators : int, default=10
            Number of trees in each Random Forest
        rf_max_depth : int, default=3
            Maximum depth of trees in Random Forest
        rf_min_samples_split : int, default=2
            Minimum samples required to split in Random Forest
        rf_min_samples_leaf : int, default=1
            Minimum samples required in leaf nodes
        rf_max_features : str or int, default='sqrt'
            Number of features to consider for best split
        learning_rate : float, default=1.0
            Learning rate shrinks contribution of each classifier
        algorithm : str, default='SAMME'
            Boosting algorithm ('SAMME' or 'SAMME.R')
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_max_features = rf_max_features
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        
    def _create_base_estimator(self):
        """Create a Random Forest base estimator"""
        return RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            max_features=self.rf_max_features,
            random_state=self.random_state,
            bootstrap=True,  # Enable bootstrap for better diversity
            oob_score=False  # Disable OOB to save computation
        )
        
    def fit(self, X, y):
        """
        Fit the AdaBoost Random Forest classifier
        
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
        
        # Create AdaBoost classifier with Random Forest as base
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
        
        return self.adaboost_.predict(X)
        
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
        
        return self.adaboost_.predict_proba(X)
        
    def decision_function(self, X):
        """
        Compute decision function for samples in X
        
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
        
        return self.adaboost_.decision_function(X)
        
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
    def feature_importances_(self):
        """
        Get feature importances from the trained model
        
        Returns:
        --------
        importances : ndarray of shape (n_features,)
            Feature importances
        """
        check_is_fitted(self)
        
        # Get feature importances from all base estimators
        importances = np.zeros(self.n_features_in_)
        
        for estimator, weight in zip(self.adaboost_.estimators_, 
                                   self.adaboost_.estimator_weights_):
            if hasattr(estimator, 'feature_importances_'):
                importances += weight * estimator.feature_importances_
                
        # Normalize
        if importances.sum() > 0:
            importances = importances / importances.sum()
            
        return importances
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, return parameters for sub-estimators too
            
        Returns:
        --------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'n_estimators': self.n_estimators,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': self.rf_max_depth,
            'rf_min_samples_split': self.rf_min_samples_split,
            'rf_min_samples_leaf': self.rf_min_samples_leaf,
            'rf_max_features': self.rf_max_features,
            'learning_rate': self.learning_rate,
            'algorithm': self.algorithm,
            'random_state': self.random_state
        }
        
    def set_params(self, **params):
        """
        Set parameters for this estimator
        
        Parameters:
        -----------
        **params : dict
            Estimator parameters
            
        Returns:
        --------
        self : object
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self
        
    def staged_predict(self, X):
        """
        Return staged predictions for X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Yields:
        -------
        y_pred : generator of ndarray of shape (n_samples,)
            Predicted class labels at each stage
        """
        check_is_fitted(self)
        X = check_array(X)
        
        for pred in self.adaboost_.staged_predict(X):
            yield pred
            
    def staged_predict_proba(self, X):
        """
        Return staged class probabilities for X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Yields:
        -------
        y_proba : generator of ndarray of shape (n_samples, n_classes)
            Predicted class probabilities at each stage
        """
        check_is_fitted(self)
        X = check_array(X)
        
        for proba in self.adaboost_.staged_predict_proba(X):
            yield proba
            
    def staged_score(self, X, y):
        """
        Return staged scores for X, y
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
            
        Yields:
        -------
        score : generator of float
            Accuracy scores at each stage
        """
        check_is_fitted(self)
        
        for pred in self.staged_predict(X):
            yield accuracy_score(y, pred)
            
    def get_estimator_info(self):
        """
        Get information about the trained estimators
        
        Returns:
        --------
        info : dict
            Dictionary containing estimator information
        """
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


# Convenience function for quick usage
def create_adaboost_rf(n_estimators=50, rf_n_estimators=10, rf_max_depth=3, 
                      learning_rate=1.0, random_state=None):
    """
    Create an AdaBoost Random Forest classifier with sensible defaults
    
    Parameters:
    -----------
    n_estimators : int, default=50
        Number of boosting rounds
    rf_n_estimators : int, default=10
        Number of trees in each Random Forest
    rf_max_depth : int, default=3
        Maximum depth of trees
    learning_rate : float, default=1.0
        Learning rate
    random_state : int, default=None
        Random state
        
    Returns:
    --------
    classifier : AdaBoostRandomForest
        Configured classifier
    """
    return AdaBoostRandomForest(
        n_estimators=n_estimators,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
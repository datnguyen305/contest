import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

class RF(BaseEstimator, ClassifierMixin):
    """
    Random Forest Classifier with additional features
    
    This is a wrapper around sklearn's RandomForestClassifier with
    additional functionality and customization options
    """
    
    def __init__(self, 
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='sqrt',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 normalize=False,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 n_jobs=None):
        """
        Initialize Random Forest classifier
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        criterion : str, default='gini'
            Function to measure quality of split ('gini', 'entropy')
        max_depth : int, default=None
            Maximum depth of trees
        min_samples_split : int or float, default=2
            Minimum samples required to split internal node
        min_samples_leaf : int or float, default=1
            Minimum samples required to be at leaf node
        min_weight_fraction_leaf : float, default=0.0
            Minimum weighted fraction of sum total weights at leaf
        max_features : int, float, str, default='sqrt'
            Number of features to consider for best split
        max_leaf_nodes : int, default=None
            Maximum number of leaf nodes
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease required for split
        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees
        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate generalization accuracy
        normalize : bool, default=False
            Whether to normalize features before training
        random_state : int, default=None
            Random state for reproducibility
        verbose : int, default=0
            Verbosity level
        warm_start : bool, default=False
            Reuse solution of previous call to fit
        class_weight : dict, str, default=None
            Weights associated with classes
        ccp_alpha : float, default=0.0
            Complexity parameter for pruning
        max_samples : int or float, default=None
            Number of samples to draw for each base estimator
        n_jobs : int, default=None
            Number of parallel jobs
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.normalize = normalize
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        """
        Fit the Random Forest classifier
        
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
            
        # Create and fit Random Forest
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs
        )
        
        with warnings.catch_warnings():
            if self.verbose == 0:
                warnings.filterwarnings("ignore", category=UserWarning)
            self.rf_.fit(X_scaled, y)
        
        return self
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.rf_.predict(X_scaled)
        
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.rf_.predict_proba(X_scaled)
        
    def predict_log_proba(self, X):
        """Predict class log-probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.rf_.predict_log_proba(X_scaled)
        
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    @property
    def feature_importances_(self):
        """Get feature importances"""
        check_is_fitted(self)
        return self.rf_.feature_importances_
        
    @property
    def oob_score_(self):
        """Get out-of-bag score"""
        check_is_fitted(self)
        if not self.oob_score:
            raise AttributeError("OOB score not available. Set oob_score=True")
        return self.rf_.oob_score_
        
    @property
    def oob_decision_function_(self):
        """Get out-of-bag decision function"""
        check_is_fitted(self)
        if not self.oob_score:
            raise AttributeError("OOB decision function not available. Set oob_score=True")
        return self.rf_.oob_decision_function_
        
    @property
    def estimators_(self):
        """Get the collection of fitted sub-estimators"""
        check_is_fitted(self)
        return self.rf_.estimators_
        
    def apply(self, X):
        """Apply trees in the forest to X, return leaf indices"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.rf_.apply(X_scaled)
        
    def decision_path(self, X):
        """Return decision path in the forest"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.rf_.decision_path(X_scaled)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'normalize': self.normalize,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'class_weight': self.class_weight,
            'ccp_alpha': self.ccp_alpha,
            'max_samples': self.max_samples,
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
        
        info = {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'normalize': self.normalize,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'feature_importances_': self.feature_importances_
        }
        
        if self.oob_score:
            info['oob_score_'] = self.oob_score_
            
        return info


class RFRegressor(BaseEstimator, RegressorMixin):
    """Random Forest Regressor with additional features"""
    
    def __init__(self, 
                 n_estimators=100,
                 criterion='squared_error',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 bootstrap=True,
                 oob_score=False,
                 normalize=False,
                 random_state=None,
                 n_jobs=None):
        """Initialize Random Forest regressor"""
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.normalize = normalize
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        """Fit the Random Forest regressor"""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_scaled = X
            
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.rf_.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        """Predict target values for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
            
        return self.rf_.predict(X_scaled)
        
    @property
    def feature_importances_(self):
        """Get feature importances"""
        check_is_fitted(self)
        return self.rf_.feature_importances_


# Convenience functions
def create_random_forest(n_estimators=100, max_depth=None, normalize=False, random_state=None):
    """Create a Random Forest classifier with sensible defaults"""
    return RF(
        n_estimators=n_estimators,
        max_depth=max_depth,
        normalize=normalize,
        random_state=random_state
    )

def create_balanced_random_forest(n_estimators=100, normalize=False, random_state=None):
    """Create a balanced Random Forest classifier"""
    return RF(
        n_estimators=n_estimators,
        class_weight='balanced',
        normalize=normalize,
        random_state=random_state
    )

def create_oob_random_forest(n_estimators=100, normalize=False, random_state=None):
    """Create a Random Forest with out-of-bag scoring"""
    return RF(
        n_estimators=n_estimators,
        oob_score=True,
        normalize=normalize,
        random_state=random_state
    )

def create_rf_regressor(n_estimators=100, max_depth=None, normalize=False, random_state=None):
    """Create a Random Forest regressor"""
    return RFRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        normalize=normalize,
        random_state=random_state
    )
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

class BaggingRF(BaseEstimator, ClassifierMixin):
    """
    Bagging classifier using Random Forest as base estimator
    
    This creates an ensemble of Random Forest classifiers using bootstrap
    aggregating, effectively creating a "forest of forests"
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 rf_n_estimators=10,
                 rf_criterion='gini',
                 rf_max_depth=None,
                 rf_min_samples_split=2,
                 rf_min_samples_leaf=1,
                 rf_max_features='sqrt',
                 rf_bootstrap=True,
                 rf_oob_score=False,
                 normalize=False,
                 random_state=None):
        """
        Initialize Bagging Random Forest classifier
        
        Parameters:
        -----------
        n_estimators : int, default=10
            Number of Random Forest base estimators in the ensemble
        max_samples : int or float, default=1.0
            Number of samples to draw from X to train each base estimator
        max_features : int or float, default=1.0
            Number of features to draw from X to train each base estimator
        bootstrap : bool, default=True
            Whether samples are drawn with replacement for bagging
        bootstrap_features : bool, default=False
            Whether features are drawn with replacement for bagging
        oob_score : bool, default=False
            Whether to use out-of-bag samples for bagging score estimation
        warm_start : bool, default=False
            Whether to reuse solution of previous call to fit
        n_jobs : int, default=None
            Number of jobs to run in parallel
        rf_n_estimators : int, default=10
            Number of trees in each Random Forest
        rf_criterion : str, default='gini'
            Splitting criterion for Random Forest
        rf_max_depth : int, default=None
            Maximum depth of trees in Random Forest
        rf_min_samples_split : int, default=2
            Minimum samples required to split node in Random Forest
        rf_min_samples_leaf : int, default=1
            Minimum samples required at leaf node in Random Forest
        rf_max_features : str, default='sqrt'
            Number of features to consider for best split in Random Forest
        rf_bootstrap : bool, default=True
            Whether Random Forest uses bootstrap sampling
        rf_oob_score : bool, default=False
            Whether Random Forest computes OOB score
        normalize : bool, default=False
            Whether to normalize features (usually not needed for RF)
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.rf_n_estimators = rf_n_estimators
        self.rf_criterion = rf_criterion
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_max_features = rf_max_features
        self.rf_bootstrap = rf_bootstrap
        self.rf_oob_score = rf_oob_score
        self.normalize = normalize
        self.random_state = random_state
        
    def _create_base_estimator(self):
        """Create a Random Forest base estimator"""
        return RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            criterion=self.rf_criterion,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            max_features=self.rf_max_features,
            bootstrap=self.rf_bootstrap,
            oob_score=self.rf_oob_score,
            n_jobs=1,  # Avoid nested parallelism
            random_state=self.random_state
        )
        
    def fit(self, X, y):
        """
        Fit the Bagging Random Forest classifier
        
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
        
        # Initialize scaler if needed (usually not for RF)
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
        
        # Create base estimator
        base_estimator = self._create_base_estimator()
        
        # Create Bagging classifier with Random Forest as base
        self.bagging_ = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.bagging_.fit(X, y)
        
        return self
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        return self.bagging_.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        return self.bagging_.predict_proba(X)
        
    def predict_log_proba(self, X):
        """Predict class log-probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        
        # Apply normalization if used during training
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        return self.bagging_.predict_log_proba(X)
        
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        check_is_fitted(self)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    @property
    def oob_score_(self):
        """Get out-of-bag score for bagging"""
        check_is_fitted(self)
        if not self.oob_score:
            raise AttributeError("oob_score is only available if oob_score=True")
        return self.bagging_.oob_score_
        
    @property
    def oob_decision_function_(self):
        """Get out-of-bag decision function for bagging"""
        check_is_fitted(self)
        if not self.oob_score:
            raise AttributeError("oob_decision_function_ is only available if oob_score=True")
        return self.bagging_.oob_decision_function_
        
    @property
    def feature_importances_(self):
        """Get aggregated feature importances from all Random Forests"""
        check_is_fitted(self)
        
        # Aggregate feature importances from all Random Forest estimators
        importances = np.zeros(self.n_features_in_)
        n_estimators = 0
        
        for estimator in self.bagging_.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances += estimator.feature_importances_
                n_estimators += 1
                
        if n_estimators > 0:
            importances = importances / n_estimators
        else:
            importances = np.ones(self.n_features_in_) / self.n_features_in_
            
        return importances
        
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'oob_score': self.oob_score,
            'warm_start': self.warm_start,
            'n_jobs': self.n_jobs,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_criterion': self.rf_criterion,
            'rf_max_depth': self.rf_max_depth,
            'rf_min_samples_split': self.rf_min_samples_split,
            'rf_min_samples_leaf': self.rf_min_samples_leaf,
            'rf_max_features': self.rf_max_features,
            'rf_bootstrap': self.rf_bootstrap,
            'rf_oob_score': self.rf_oob_score,
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
        
        info = {
            'n_estimators_': len(self.bagging_.estimators_),
            'estimators_': self.bagging_.estimators_,
            'estimators_features_': self.bagging_.estimators_features_,
            'feature_importances_': self.feature_importances_,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'n_features_in_': self.n_features_in_,
            'normalized': self.scaler_ is not None,
            'total_trees': len(self.bagging_.estimators_) * self.rf_n_estimators
        }
        
        # Add OOB score if available
        if self.oob_score:
            info['oob_score_'] = self.oob_score_
            
        return info


# Convenience functions
def create_bagging_rf(n_estimators=5, rf_n_estimators=10, bootstrap=True, oob_score=False, random_state=None):
    """Create a Bagging Random Forest classifier with sensible defaults"""
    return BaggingRF(
        n_estimators=n_estimators,
        rf_n_estimators=rf_n_estimators,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state
    )

def create_oob_bagging_rf(n_estimators=5, rf_n_estimators=10, random_state=None):
    """Create a Bagging Random Forest classifier with OOB scoring enabled"""
    return BaggingRF(
        n_estimators=n_estimators,
        rf_n_estimators=rf_n_estimators,
        bootstrap=True,
        oob_score=True,
        rf_oob_score=True,  # Enable OOB for base RF too
        random_state=random_state
    )

def create_deep_bagging_rf(n_estimators=3, rf_n_estimators=20, rf_max_depth=10, random_state=None):
    """Create a Bagging Random Forest classifier with deeper trees"""
    return BaggingRF(
        n_estimators=n_estimators,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_split=5,
        rf_min_samples_leaf=2,
        random_state=random_state
    )

def create_feature_bagging_rf(n_estimators=5, rf_n_estimators=10, max_features=0.8, random_state=None):
    """Create a Bagging Random Forest classifier with feature bootstrapping"""
    return BaggingRF(
        n_estimators=n_estimators,
        rf_n_estimators=rf_n_estimators,
        max_features=max_features,
        bootstrap_features=True,
        random_state=random_state
    )
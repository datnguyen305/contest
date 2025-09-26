import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import warnings
warnings.filterwarnings('ignore')

class AdaBoostLR(BaseEstimator, ClassifierMixin):
    """
    AdaBoost classifier using Logistic Regression as base classifier
    Uses sklearn's LogisticRegression and follows sklearn's API conventions
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None,
                 max_iter=1000, solver='lbfgs'):
        """
        Initialize AdaBoost with Logistic Regression
        
        Args:
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Learning rate for boosting
            random_state (int): Random seed for reproducibility
            max_iter (int): Maximum iterations for LogisticRegression
            solver (str): Solver for LogisticRegression
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver
        
    def fit(self, X, y):
        """
        Train the AdaBoost classifier
        
        Args:
            X (array-like): Training features of shape (n_samples, n_features)
            y (array-like): Training labels of shape (n_samples,)
            
        Returns:
            self: Returns the instance itself
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes and number of features
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Check if binary classification
        if len(self.classes_) != 2:
            raise ValueError("AdaBoost with LR currently supports only binary classification")
        
        # Convert labels to -1, +1
        y_encoded = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = X.shape[0]
        
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        # Initialize storage for base classifiers and their weights
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
        for iteration in range(self.n_estimators):
            # Create base classifier
            base_classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=self.max_iter,
                solver=self.solver
            )
            
            # Train on weighted samples using sample_weight parameter
            base_classifier.fit(X, y_encoded, sample_weight=sample_weights)
            
            # Make predictions
            predictions = base_classifier.predict(X)
            
            # Calculate weighted error
            incorrect_mask = (predictions != y_encoded)
            error = np.average(incorrect_mask, weights=sample_weights)
            
            # Handle edge cases
            if error <= 0:
                # Perfect classifier
                self.estimators_.append(base_classifier)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(error)
                break
                
            if error >= 0.5:
                # Random classifier or worse
                if len(self.estimators_) == 0:
                    # If first classifier is bad, still add it with small weight
                    alpha = 0.1
                else:
                    # Stop if we can't improve
                    break
            else:
                # Calculate classifier weight (alpha)
                alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Store classifier and its weight
            self.estimators_.append(base_classifier)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            
            # Update sample weights
            sample_weights *= np.exp(-alpha * y_encoded * predictions)
            
            # Normalize weights
            sample_weights /= np.sum(sample_weights)
            
            # Check for numerical stability
            if np.sum(sample_weights) == 0:
                sample_weights = np.ones(n_samples) / n_samples
        
        return self
        
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (array-like): Features to predict of shape (n_samples, n_features)
            
        Returns:
            ndarray: Predicted class labels
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Get decision function scores
        decision_scores = self.decision_function(X)
        
        # Convert to class predictions
        predictions = np.where(decision_scores >= 0, self.classes_[1], self.classes_[0])
        
        return predictions
        
    def decision_function(self, X):
        """
        Compute the decision function of X
        
        Args:
            X (array-like): Features of shape (n_samples, n_features)
            
        Returns:
            ndarray: Decision function values
        """
        check_is_fitted(self)
        X = check_array(X)
        
        decision = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            # Get predictions from base classifier
            predictions = estimator.predict(X)
            decision += weight * predictions
            
        return decision
        
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (array-like): Features to predict of shape (n_samples, n_features)
            
        Returns:
            ndarray: Predicted probabilities of shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Get decision function scores
        decision_scores = self.decision_function(X)
        
        # Convert to probabilities using sigmoid function
        probabilities = 1 / (1 + np.exp(-2 * decision_scores))
        
        # Return probabilities for both classes
        return np.column_stack([1 - probabilities, probabilities])
        
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X (array-like): Test features
            y (array-like): True labels
            
        Returns:
            float: Mean accuracy
        """
        from sklearn.metrics import accuracy_score
        
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
        
    def staged_predict(self, X):
        """
        Return staged predictions for X (predictions at each boosting iteration)
        
        Args:
            X (array-like): Features to predict
            
        Yields:
            ndarray: Predictions at each stage
        """
        check_is_fitted(self)
        X = check_array(X)
        
        decision = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            predictions = estimator.predict(X)
            decision += weight * predictions
            
            # Convert to class predictions
            stage_predictions = np.where(decision >= 0, self.classes_[1], self.classes_[0])
            yield stage_predictions
            
    def staged_score(self, X, y):
        """
        Return staged scores for X, y (scores at each boosting iteration)
        
        Args:
            X (array-like): Test features
            y (array-like): True labels
            
        Yields:
            float: Accuracy at each stage
        """
        from sklearn.metrics import accuracy_score
        
        for predictions in self.staged_predict(X):
            yield accuracy_score(y, predictions)
            
    def get_params(self, deep=True):
        """
        Get parameters for this estimator
        
        Args:
            deep (bool): If True, return parameters for this estimator and sub-estimators
            
        Returns:
            dict: Parameter names mapped to their values
        """
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'solver': self.solver
        }
        
    def set_params(self, **params):
        """
        Set the parameters of this estimator
        
        Args:
            **params: Estimator parameters
            
        Returns:
            self: Estimator instance
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
        
    @property
    def feature_importances_(self):
        """
        Calculate feature importances based on base classifiers
        
        Returns:
            ndarray: Feature importances
        """
        check_is_fitted(self)
        
        if not hasattr(self, '_feature_importances'):
            importances = np.zeros(self.n_features_in_)
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                # For LogisticRegression, use absolute values of coefficients as importance
                coef = np.abs(estimator.coef_[0])
                importances += weight * coef
                
            # Normalize
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)
                
            self._feature_importances = importances
            
        return self._feature_importances
import numpy as np
from collections import Counter
import warnings

# Try to import sklearn components, fallback to basic implementation if not available
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.utils.validation import check_X_y, check_array
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Using basic implementation.")

if SKLEARN_AVAILABLE:
    class WeightedKNNClassifier(BaseEstimator, ClassifierMixin):
        """
        KNN Classifier that supports sample weights (wrapper around sklearn KNN)
        """
        
        def __init__(self, n_neighbors=3, metric='euclidean', weights='distance'):
            self.n_neighbors = n_neighbors
            self.metric = metric
            self.weights = weights
            self.knn = KNeighborsClassifier(
                n_neighbors=n_neighbors, 
                metric=metric, 
                weights=weights
            )
            self.sample_weights_ = None
            
        def fit(self, X, y, sample_weight=None):
            """Fit the weighted KNN classifier"""
            X, y = check_X_y(X, y)
            check_classification_targets(y)
            
            self.classes_ = np.unique(y)
            self.sample_weights_ = sample_weight
            
            if sample_weight is not None:
                # For weighted sampling, we duplicate samples based on weights
                # This is a simple approach - more sophisticated methods exist
                weights_normalized = sample_weight / np.sum(sample_weight) * len(sample_weight)
                indices = np.random.choice(
                    len(X), 
                    size=len(X), 
                    p=sample_weight/np.sum(sample_weight),
                    replace=True
                )
                X_weighted = X[indices]
                y_weighted = y[indices]
                self.knn.fit(X_weighted, y_weighted)
            else:
                self.knn.fit(X, y)
                
            return self
            
        def predict(self, X):
            """Make predictions"""
            X = check_array(X)
            return self.knn.predict(X)
            
        def predict_proba(self, X):
            """Predict class probabilities"""
            X = check_array(X)
            return self.knn.predict_proba(X)
else:
    # Basic KNN implementation if sklearn not available
    class WeightedKNNClassifier:
        """
        Simple KNN implementation with sample weight support
        """
        
        def __init__(self, n_neighbors=3, metric='euclidean', weights='distance'):
            self.n_neighbors = n_neighbors
            self.metric = metric
            self.weights = weights
            self.X_train_ = None
            self.y_train_ = None
            self.sample_weights_ = None
            self.classes_ = None
            
        def fit(self, X, y, sample_weight=None):
            """Fit the KNN classifier"""
            self.X_train_ = np.array(X)
            self.y_train_ = np.array(y)
            self.classes_ = np.unique(y)
            self.sample_weights_ = sample_weight if sample_weight is not None else np.ones(len(y))
            return self
            
        def _calculate_distance(self, x1, x2):
            """Calculate distance between two points"""
            if self.metric == 'euclidean':
                return np.sqrt(np.sum((x1 - x2) ** 2))
            elif self.metric == 'manhattan':
                return np.sum(np.abs(x1 - x2))
            else:
                return np.sqrt(np.sum((x1 - x2) ** 2))  # Default to euclidean
                
        def predict(self, X):
            """Make predictions"""
            X = np.array(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
                
            predictions = []
            
            for sample in X:
                # Calculate distances
                distances = []
                for i, train_sample in enumerate(self.X_train_):
                    dist = self._calculate_distance(sample, train_sample)
                    distances.append((dist, self.y_train_[i], self.sample_weights_[i]))
                
                # Sort and get k nearest
                distances.sort(key=lambda x: x[0])
                k_nearest = distances[:self.n_neighbors]
                
                # Voting
                if self.weights == 'distance':
                    class_weights = {}
                    for dist, label, weight in k_nearest:
                        if label not in class_weights:
                            class_weights[label] = 0
                        distance_weight = 1.0 / (dist + 1e-8)
                        class_weights[label] += weight * distance_weight
                else:
                    # Uniform weights
                    votes = [label for _, label, _ in k_nearest]
                    class_weights = Counter(votes)
                
                predicted_class = max(class_weights, key=class_weights.get)
                predictions.append(predicted_class)
                
            return np.array(predictions)


class AdaBoostKNN:
    """
    AdaBoost classifier using KNN as base classifier
    Uses sklearn's AdaBoostClassifier with KNN base estimator when available
    """
    
    def __init__(self, n_estimators=50, n_neighbors=3, metric='euclidean', 
                 learning_rate=1.0, random_state=None, algorithm='SAMME'):
        """
        Initialize AdaBoost with KNN
        
        Args:
            n_estimators (int): Number of boosting rounds
            n_neighbors (int): Number of neighbors for KNN
            metric (str): Distance metric for KNN
            learning_rate (float): Learning rate for boosting
            random_state (int): Random seed for reproducibility
            algorithm (str): AdaBoost algorithm ('SAMME' or 'SAMME.R')
        """
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.algorithm = algorithm
        
        if SKLEARN_AVAILABLE:
            # Use sklearn's AdaBoost with KNN base estimator
            base_estimator = WeightedKNNClassifier(
                n_neighbors=n_neighbors,
                metric=metric,
                weights='distance'
            )
            
            self.model = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                algorithm=algorithm,
                random_state=random_state
            )
        else:
            # Fallback to manual implementation
            self.model = None
            self.classifiers = []
            self.classifier_weights = []
            self.classes = None
            self.n_features = None
            
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
            
    def fit(self, X, y):
        """
        Train the AdaBoost classifier
        
        Args:
            X (numpy.ndarray): Training features
            y (numpy.ndarray): Training labels
        """
        X = np.array(X)
        y = np.array(y)
        
        if SKLEARN_AVAILABLE:
            # Use sklearn implementation
            self.model.fit(X, y)
        else:
            # Manual implementation
            self._fit_manual(X, y)
            
        self.is_fitted = True
        return self
        
    def _fit_manual(self, X, y):
        """Manual AdaBoost implementation when sklearn not available"""
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        
        # Encode labels for binary classification
        if len(self.classes) == 2:
            y_encoded = np.where(y == self.classes[0], -1, 1)
        else:
            y_encoded = y.copy()
            
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples
        
        self.classifiers = []
        self.classifier_weights = []
        
        for t in range(self.n_estimators):
            # Create and train KNN classifier
            knn = WeightedKNNClassifier(
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                weights='distance'
            )
            knn.fit(X, y_encoded, sample_weight=sample_weights)
            
            # Make predictions
            predictions = knn.predict(X)
            
            # Calculate weighted error
            incorrect = (predictions != y_encoded)
            error = np.sum(sample_weights * incorrect)
            
            # Avoid numerical issues
            error = max(error, 1e-10)
            error = min(error, 1 - 1e-10)
            
            # Calculate classifier weight
            if len(self.classes) == 2:
                alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            else:
                alpha = self.learning_rate * np.log((1 - error) / error) + np.log(len(self.classes) - 1)
            
            # Store classifier and its weight
            self.classifiers.append(knn)
            self.classifier_weights.append(alpha)
            
            # Update sample weights
            if len(self.classes) == 2:
                sample_weights *= np.exp(-alpha * y_encoded * predictions)
            else:
                sample_weights *= np.exp(alpha * incorrect)
            
            # Normalize weights
            sample_weights /= np.sum(sample_weights)
            
            # Early stopping
            if error < 1e-10:
                break
        
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if SKLEARN_AVAILABLE:
            return self.model.predict(X)
        else:
            return self._predict_manual(X)
            
    def _predict_manual(self, X):
        """Manual prediction when sklearn not available"""
        n_samples = X.shape[0]
        
        if len(self.classes) == 2:
            # Binary classification
            final_predictions = np.zeros(n_samples)
            
            for classifier, weight in zip(self.classifiers, self.classifier_weights):
                predictions = classifier.predict(X)
                final_predictions += weight * predictions
                
            # Convert back to original labels
            binary_predictions = np.sign(final_predictions)
            result = np.where(binary_predictions == -1, self.classes[0], self.classes[1])
            
        else:
            # Multi-class classification
            class_scores = {class_label: np.zeros(n_samples) for class_label in self.classes}
            
            for classifier, weight in zip(self.classifiers, self.classifier_weights):
                predictions = classifier.predict(X)
                
                for i, pred in enumerate(predictions):
                    if pred in class_scores:
                        class_scores[pred][i] += weight
                    
            # Get class with highest score for each sample
            result = []
            for i in range(n_samples):
                scores = {class_label: class_scores[class_label][i] for class_label in self.classes}
                predicted_class = max(scores, key=scores.get)
                result.append(predicted_class)
                
            result = np.array(result)
            
        return result
        
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if SKLEARN_AVAILABLE:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # If SAMME algorithm doesn't support predict_proba, use decision function
                decision_scores = self.model.decision_function(X)
                if decision_scores.ndim == 1:  # Binary classification
                    probabilities = 1 / (1 + np.exp(-decision_scores))
                    return np.column_stack([1 - probabilities, probabilities])
                else:  # Multi-class
                    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            return self._predict_proba_manual(X)
            
    def _predict_proba_manual(self, X):
        """Manual probability prediction when sklearn not available"""
        n_samples = X.shape[0]
        
        if len(self.classes) == 2:
            # Binary classification
            decision_scores = np.zeros(n_samples)
            
            for classifier, weight in zip(self.classifiers, self.classifier_weights):
                predictions = classifier.predict(X)
                decision_scores += weight * predictions
                
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-2 * decision_scores))
            return np.column_stack([1 - probabilities, probabilities])
            
        else:
            # Multi-class classification
            class_scores = np.zeros((n_samples, len(self.classes)))
            
            for classifier, weight in zip(self.classifiers, self.classifier_weights):
                predictions = classifier.predict(X)
                
                for i, pred in enumerate(predictions):
                    class_idx = np.where(self.classes == pred)[0]
                    if len(class_idx) > 0:
                        class_scores[i, class_idx[0]] += weight
                    
            # Apply softmax to convert scores to probabilities
            exp_scores = np.exp(class_scores - np.max(class_scores, axis=1, keepdims=True))
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            return probabilities
            
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X (numpy.ndarray): Test features
            y (numpy.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        if SKLEARN_AVAILABLE:
            return self.model.score(X, y)
        else:
            predictions = self.predict(X)
            return np.mean(predictions == y)
        
    def get_feature_importances(self):
        """
        Get feature importances
        Returns uniform importance since KNN doesn't provide direct feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if SKLEARN_AVAILABLE and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            # Return uniform importance for KNN-based models
            n_features = getattr(self, 'n_features', None)
            if n_features is None:
                raise ValueError("Number of features not available")
            return np.ones(n_features) / n_features
        
    def get_classifier_info(self):
        """
        Get information about the trained classifiers
        
        Returns:
            dict: Information about the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        info = {
            'n_estimators': self.n_estimators,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'learning_rate': self.learning_rate,
            'sklearn_available': SKLEARN_AVAILABLE
        }
        
        if SKLEARN_AVAILABLE:
            info['algorithm'] = self.algorithm
            if hasattr(self.model, 'estimators_'):
                info['actual_n_estimators'] = len(self.model.estimators_)
            if hasattr(self.model, 'estimator_weights_'):
                info['estimator_weights'] = self.model.estimator_weights_
        else:
            info['actual_n_estimators'] = len(self.classifiers) if hasattr(self, 'classifiers') else 0
            info['classifier_weights'] = getattr(self, 'classifier_weights', [])
            
        return info
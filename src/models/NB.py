import numpy as np
from collections import defaultdict
import math

class NaiveBayes:
    """
    Naive Bayes Classifier implementation
    Supports both Gaussian and Multinomial variants
    """
    
    def __init__(self, variant='gaussian'):
        """
        Initialize Naive Bayes classifier
        
        Args:
            variant (str): 'gaussian' for continuous features, 'multinomial' for discrete features
        """
        self.variant = variant
        self.classes = None
        self.class_priors = {}
        self.feature_stats = {}  # For Gaussian: mean and std, For Multinomial: feature counts
        self.n_features = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Args:
            X (numpy.ndarray): Training features of shape (n_samples, n_features)
            y (numpy.ndarray): Training labels of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        
        # Calculate class priors P(class)
        n_samples = len(y)
        for class_label in self.classes:
            self.class_priors[class_label] = np.sum(y == class_label) / n_samples
            
        # Calculate feature statistics for each class
        if self.variant == 'gaussian':
            self._fit_gaussian(X, y)
        elif self.variant == 'multinomial':
            self._fit_multinomial(X, y)
        else:
            raise ValueError("Variant must be 'gaussian' or 'multinomial'")
            
        self.is_fitted = True
        
    def _fit_gaussian(self, X, y):
        """
        Fit Gaussian Naive Bayes (for continuous features)
        Calculate mean and standard deviation for each feature in each class
        """
        self.feature_stats = {}
        
        for class_label in self.classes:
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Calculate mean and std for each feature
            means = np.mean(X_class, axis=0)
            stds = np.std(X_class, axis=0)
            
            # Add small epsilon to avoid division by zero
            stds = np.where(stds == 0, 1e-6, stds)
            
            self.feature_stats[class_label] = {
                'mean': means,
                'std': stds
            }
            
    def _fit_multinomial(self, X, y):
        """
        Fit Multinomial Naive Bayes (for discrete features)
        Calculate feature counts for each class
        """
        self.feature_stats = {}
        
        for class_label in self.classes:
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Count occurrences of each feature value
            feature_counts = np.sum(X_class, axis=0)
            
            # Add Laplace smoothing (add 1 to avoid zero probabilities)
            feature_counts = feature_counts + 1
            total_count = np.sum(feature_counts)
            
            self.feature_stats[class_label] = {
                'counts': feature_counts,
                'total': total_count
            }
            
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X (numpy.ndarray): Features to predict of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        
        for sample in X:
            class_scores = {}
            
            for class_label in self.classes:
                # Start with log of class prior
                log_prob = math.log(self.class_priors[class_label])
                
                # Add log probabilities of features
                if self.variant == 'gaussian':
                    log_prob += self._calculate_gaussian_log_likelihood(sample, class_label)
                elif self.variant == 'multinomial':
                    log_prob += self._calculate_multinomial_log_likelihood(sample, class_label)
                    
                class_scores[class_label] = log_prob
                
            # Predict class with highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
            
        return np.array(predictions)
        
    def _calculate_gaussian_log_likelihood(self, sample, class_label):
        """
        Calculate log likelihood for Gaussian Naive Bayes
        """
        mean = self.feature_stats[class_label]['mean']
        std = self.feature_stats[class_label]['std']
        
        log_likelihood = 0
        for i, (x, mu, sigma) in enumerate(zip(sample, mean, std)):
            # Gaussian probability density function (in log space)
            log_likelihood += -0.5 * math.log(2 * math.pi * sigma**2)
            log_likelihood += -0.5 * ((x - mu) / sigma) ** 2
            
        return log_likelihood
        
    def _calculate_multinomial_log_likelihood(self, sample, class_label):
        """
        Calculate log likelihood for Multinomial Naive Bayes
        """
        counts = self.feature_stats[class_label]['counts']
        total = self.feature_stats[class_label]['total']
        
        log_likelihood = 0
        for i, x in enumerate(sample):
            if x > 0:  # Only consider features that are present
                prob = counts[i] / total
                log_likelihood += x * math.log(prob)
                
        return log_likelihood
        
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (numpy.ndarray): Features to predict of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        probabilities = []
        
        for sample in X:
            class_scores = {}
            
            for class_label in self.classes:
                # Calculate log probability
                log_prob = math.log(self.class_priors[class_label])
                
                if self.variant == 'gaussian':
                    log_prob += self._calculate_gaussian_log_likelihood(sample, class_label)
                elif self.variant == 'multinomial':
                    log_prob += self._calculate_multinomial_log_likelihood(sample, class_label)
                    
                class_scores[class_label] = log_prob
                
            # Convert log probabilities to probabilities using softmax
            max_score = max(class_scores.values())
            exp_scores = {k: math.exp(v - max_score) for k, v in class_scores.items()}
            total_exp = sum(exp_scores.values())
            
            probs = [exp_scores[class_label] / total_exp for class_label in self.classes]
            probabilities.append(probs)
            
        return np.array(probabilities)
        
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X (numpy.ndarray): Test features
            y (numpy.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
        
    def get_feature_importance(self):
        """
        Get feature importance (for Gaussian variant only)
        Returns the inverse of standard deviation as importance measure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        if self.variant != 'gaussian':
            raise ValueError("Feature importance only available for Gaussian variant")
            
        importance = np.zeros(self.n_features)
        
        for class_label in self.classes:
            std = self.feature_stats[class_label]['std']
            # Inverse of std as importance (higher std = less important)
            class_importance = 1.0 / (std + 1e-6)
            importance += class_importance * self.class_priors[class_label]
            
        # Normalize
        importance = importance / np.sum(importance)
        
        return importance
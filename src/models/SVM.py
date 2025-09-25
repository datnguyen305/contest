import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


class SVM:
    """
    Support Vector Machine wrapper using scikit-learn
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, 
                 random_state=None, scale_features=True):
        """
        Initialize SVM model
        
        Args:
            C: Regularization parameter
            kernel: 'linear', 'poly', 'rbf', 'sigmoid'
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree of the polynomial kernel
            random_state: Random seed
            scale_features: Whether to scale features
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.scale_features = scale_features
        
        # Initialize models
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            random_state=self.random_state
        )
        
        self.scaler = StandardScaler() if scale_features else None
        self.fitted = False
    
    def fit(self, X, y):
        """
        Train the SVM model
        
        Args:
            X: Training features
            y: Training labels
        """
        X = np.array(X)
        y = np.array(y)
        
        # Scale features if requested
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X, y)
        self.fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Test features
            
        Returns:
            Predicted labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        # Scale features if scaler was used during training
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (requires probability=True in SVC)
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model was not trained with probability=True")
        
        X = np.array(X)
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_support_vectors(self):
        """
        Get support vectors
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.support_vectors_
    
    def get_params(self):
        """
        Get model parameters
        """
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'n_support_vectors': len(self.model.support_vectors_) if self.fitted else 0
        }
    
    def plot_confusion_matrix(self, X_test, y_test, figsize=(8, 6)):
        """
        Plot confusion matrix
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def classification_report(self, X_test, y_test):
        """
        Print classification report
        """
        y_pred = self.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    def plot_decision_boundary(self, X, y, resolution=100, figsize=(10, 8)):
        """
        Plot decision boundary for 2D data
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        if X.shape[1] != 2:
            raise ValueError("Can only plot decision boundary for 2D data")
        
        # Scale data if scaler was used
        X_plot = self.scaler.transform(X) if self.scaler else X
        
        # Create mesh
        h = 0.02
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        
        # Plot support vectors
        if hasattr(self.model, 'support_vectors_'):
            plt.scatter(self.model.support_vectors_[:, 0], 
                       self.model.support_vectors_[:, 1],
                       s=100, facecolors='none', edgecolors='black', linewidth=2)
        
        plt.colorbar(scatter)
        plt.title(f'SVM Decision Boundary (kernel={self.kernel}, C={self.C})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
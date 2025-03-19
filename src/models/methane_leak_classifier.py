import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class MethaneLeakClassifier:
    """
    Binary classification model to detect the presence or absence of methane leaks
    based on whether methane concentration > 0 (leak present) or = 0 (leak absent).
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the methane leak classifier.
        
        Parameters:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1  # Use all available cores
        )
        self.feature_importances_ = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """
        Train the methane leak classifier.
        
        Parameters:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features)
            y (np.ndarray): Binary target values (1 for leak, 0 for no leak)
            feature_names (list, optional): Names of the features
        """
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        
        print("Methane leak classifier training complete.")
        
        # Print feature importances if feature names are available
        if self.feature_names is not None:
            importances = sorted(zip(self.feature_importances_, self.feature_names), reverse=True)
            print("\nFeature importance ranking:")
            for importance, feature in importances[:10]:  # Show top 10 features
                print(f"{feature}: {importance:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict leak presence using the trained classifier.
        
        Parameters:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features)
        
        Returns:
            y_pred (np.ndarray): Binary predictions (1 for leak, 0 for no leak)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probability of leak presence.
        
        Parameters:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features)
        
        Returns:
            proba (np.ndarray): Probability of leak for each sample
        """
        # Return probability of the positive class (leak present)
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Parameters:
            X_test (np.ndarray): Test feature matrix
            y_test (np.ndarray): Test target values
        
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(conf_matrix)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Parameters:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from disk.
        
        Parameters:
            filepath (str): Path to the saved model
            
        Returns:
            MethaneLeakClassifier: Loaded model
        """
        return joblib.load(filepath)


if __name__ == "__main__":
    print("MethaneLeakClassifier module - this module provides binary classification for methane leak detection")
    print("Import and use this module in other scripts rather than running it directly.") 

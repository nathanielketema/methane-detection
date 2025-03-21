import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix,
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


class ClassificationEvaluator:
    """
    Evaluates classification models using multiple metrics and visualizations.
    Designed for methane leak detection classification tasks.
    """
    
    def __init__(self, model=None):
        """
        Initialize the classification evaluator.
        
        Parameters:
            model: Optional classifier model with predict and predict_proba methods
        """
        self.model = model
        self.metrics = {}
    
    def evaluate(self, X_test, y_true, y_pred=None, y_prob=None):
        """
        Evaluate classification performance using various metrics.
        
        Parameters:
            X_test: Test feature matrix
            y_true: True labels
            y_pred: Predicted labels (if None, will use self.model.predict)
            y_prob: Predicted probabilities (if None, will use self.model.predict_proba if available)
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Get predictions if not provided
        if y_pred is None and self.model is not None:
            y_pred = self.model.predict(X_test)
        
        if y_pred is None:
            raise ValueError("Either y_pred must be provided or a model must be set.")
        
        # Get probabilities if not provided (for ROC-AUC)
        if y_prob is None and self.model is not None and hasattr(self.model, 'predict_proba'):
            try:
                y_prob = self.model.predict_proba(X_test)
                # Handle both 1D and 2D probability arrays
                if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
                    # Standard sklearn format: [P(negative), P(positive)]
                    y_prob = y_prob[:, 1]  # Use probability of positive class
            except (IndexError, AttributeError):
                y_prob = None
        
        # Calculate metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        self.metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        self.metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        if y_prob is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return self.metrics
    
    def print_report(self):
        """
        Print a formatted report of classification metrics.
        """
        if not self.metrics:
            print("No metrics available. Run evaluate() first.")
            return
        
        print("\n===== Classification Metrics =====")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        if 'roc_auc' in self.metrics:
            print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1']:.4f}")
        print(f"MCC:       {self.metrics['mcc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(self.metrics['confusion_matrix'])
    
    def plot_confusion_matrix(self, normalize=True, save_path=None, figsize=(8, 6)):
        """
        Plot confusion matrix as a heatmap.
        
        Parameters:
            normalize: Whether to normalize the confusion matrix
            save_path: If provided, the plot will be saved to this path
            figsize: Figure size as (width, height)
            
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        if 'confusion_matrix' not in self.metrics:
            raise ValueError("Confusion matrix not available. Run evaluate() first.")
        
        cm = self.metrics['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            vmin, vmax = 0, 1
        else:
            fmt = 'd'
            vmin, vmax = None, None
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=['No Leak', 'Leak'],
                    yticklabels=['No Leak', 'Leak'],
                    vmin=vmin, vmax=vmax, ax=ax)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig, ax
    
    def plot_roc_curve(self, X_test, y_true, save_path=None, figsize=(8, 6)):
        """
        Plot ROC curve for the classifier.
        
        Parameters:
            X_test: Test feature matrix
            y_true: True labels
            save_path: If provided, the plot will be saved to this path
            figsize: Figure size as (width, height)
            
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model not set or doesn't support predict_proba.")
        
        from sklearn.metrics import roc_curve
        
        # Get prediction probabilities
        y_prob = self.model.predict_proba(X_test)
        
        # Handle both 1D and 2D probability arrays
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            # Standard sklearn format: [P(negative), P(positive)]
            y_prob = y_prob[:, 1]  # Use probability of positive class
        
        # Calculate ROC curve points
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, label=f'AUC = {self.metrics.get("roc_auc", 0):.4f}')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig, ax


class RegressionEvaluator:
    """
    Evaluates regression models using multiple metrics and visualizations.
    Designed for methane concentration prediction tasks.
    """
    
    def __init__(self, model=None):
        """
        Initialize the regression evaluator.
        
        Parameters:
            model: Optional regressor model with predict method
        """
        self.model = model
        self.metrics = {}
    
    def evaluate(self, X_test, y_true, y_pred=None):
        """
        Evaluate regression performance using various metrics.
        
        Parameters:
            X_test: Test feature matrix
            y_true: True values
            y_pred: Predicted values (if None, will use self.model.predict)
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Get predictions if not provided
        if y_pred is None and self.model is not None:
            try:
                # Handle the QGPR model which returns both predictions and uncertainty
                y_pred_result = self.model.predict(X_test)
                if isinstance(y_pred_result, tuple):
                    y_pred = y_pred_result[0]  # Get just predictions, not uncertainty
                else:
                    y_pred = y_pred_result
            except:
                raise ValueError("Error calling predict on the model")
        
        if y_pred is None:
            raise ValueError("Either y_pred must be provided or a model must be set.")
        
        # Calculate metrics
        self.metrics['r2'] = r2_score(y_true, y_pred)
        self.metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        self.metrics['mse'] = mean_squared_error(y_true, y_pred)
        self.metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Store original values for plotting
        self.y_true = y_true
        self.y_pred = y_pred
        
        return self.metrics
    
    def print_report(self):
        """
        Print a formatted report of regression metrics.
        """
        if not self.metrics:
            print("No metrics available. Run evaluate() first.")
            return
        
        print("\n===== Regression Metrics =====")
        print(f"R²:    {self.metrics['r2']:.4f}")
        print(f"RMSE:  {self.metrics['rmse']:.4f}")
        print(f"MSE:   {self.metrics['mse']:.4f}")
        print(f"MAE:   {self.metrics['mae']:.4f}")
    
    def plot_predictions(self, save_path=None, figsize=(8, 6)):
        """
        Create a scatter plot of predicted vs actual values.
        
        Parameters:
            save_path: If provided, the plot will be saved to this path
            figsize: Figure size as (width, height)
            
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        if not hasattr(self, 'y_true') or not hasattr(self, 'y_pred'):
            raise ValueError("Prediction data not available. Run evaluate() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the scatter plot
        ax.scatter(self.y_true, self.y_pred, alpha=0.6, edgecolor='k')
        
        # Add the diagonal line (perfect predictions)
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        margin = (max_val - min_val) * 0.1
        ax.plot([min_val - margin, max_val + margin], 
                [min_val - margin, max_val + margin], 
                'k--', label='Perfect Prediction')
        
        ax.set_xlabel('Actual Tracer Concentration')
        ax.set_ylabel('Predicted Tracer Concentration')
        ax.set_title('Predicted vs Actual Tracer Concentrations')
        
        # Add R² and RMSE to the plot
        if self.metrics:
            textstr = f"$R^2 = {self.metrics['r2']:.4f}$\n$RMSE = {self.metrics['rmse']:.4f}$"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig, ax
    
    def plot_residuals(self, save_path=None, figsize=(10, 6)):
        """
        Create plots of residuals (predicted - actual) to diagnose model performance.
        
        Parameters:
            save_path: If provided, the plot will be saved to this path
            figsize: Figure size as (width, height)
            
        Returns:
            fig: Matplotlib figure object
        """
        if not hasattr(self, 'y_true') or not hasattr(self, 'y_pred'):
            raise ValueError("Prediction data not available. Run evaluate() first.")
        
        residuals = self.y_pred - self.y_true
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Predicted values
        ax1.scatter(self.y_pred, residuals, alpha=0.6, edgecolor='k')
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Histogram of residuals
        sns.histplot(residuals, ax=ax2, kde=True)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig


def plot_geospatial_errors(lat, lon, true_values, predictions, uncertainties=None, 
                          figsize=(12, 10), save_path=None):
    """
    Create geospatial maps showing prediction errors or uncertainties.
    
    Parameters:
        lat: Array of latitude coordinates
        lon: Array of longitude coordinates
        true_values: Array of actual tracer concentrations
        predictions: Array of predicted tracer concentrations
        uncertainties: Array of prediction uncertainties (optional)
        figsize: Figure size as (width, height)
        save_path: If provided, the plot will be saved to this path
        
    Returns:
        fig: Matplotlib figure object
    """
    # Calculate absolute errors
    abs_errors = np.abs(predictions - true_values)
    
    fig, axes = plt.subplots(2, 1 if uncertainties is None else 2, figsize=figsize)
    
    if uncertainties is None:
        axes = axes.reshape(-1, 1)
    
    # Plot 1: True concentrations
    sc1 = axes[0, 0].scatter(lon, lat, c=true_values, cmap='viridis', 
                           s=50, edgecolor='k', alpha=0.8)
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title('Actual Tracer Concentrations')
    fig.colorbar(sc1, ax=axes[0, 0], label='Concentration')
    
    # Plot 2: Prediction errors
    sc2 = axes[1, 0].scatter(lon, lat, c=abs_errors, cmap='Reds', 
                           s=50, edgecolor='k', alpha=0.8)
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].set_title('Absolute Prediction Errors')
    fig.colorbar(sc2, ax=axes[1, 0], label='|Predicted - Actual|')
    
    # If uncertainties are provided
    if uncertainties is not None:
        # Plot 3: Predicted concentrations
        sc3 = axes[0, 1].scatter(lon, lat, c=predictions, cmap='viridis', 
                               s=50, edgecolor='k', alpha=0.8)
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        axes[0, 1].set_title('Predicted Tracer Concentrations')
        fig.colorbar(sc3, ax=axes[0, 1], label='Concentration')
        
        # Plot 4: Prediction uncertainties
        sc4 = axes[1, 1].scatter(lon, lat, c=uncertainties, cmap='plasma', 
                               s=50, edgecolor='k', alpha=0.8)
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].set_title('Prediction Uncertainties')
        fig.colorbar(sc4, ax=axes[1, 1], label='Uncertainty (std)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    return fig


if __name__ == "__main__":
    print("Model evaluation module - this module provides functions for evaluating classification and regression models")
    print("Import and use this module in other scripts rather than running it directly.") 
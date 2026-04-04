import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Visualize actual vs predicted prices

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, target_city: str):

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='blue')
    
    # Perfect prediction line (y = x)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    city_name = target_city.capitalize() if target_city else "All Cities"
    
    plt.title(f'Actual vs Predicted Prices ({city_name})')
    plt.xlabel('Actual Price [PLN]')
    plt.ylabel('Predicted Price [PLN]')
    plt.legend()
    plt.tight_layout()
    
    filename = f"actual_vs_predicted_{city_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, target_city: str):
    """
    Generates a histogram of the residuals (errors) to analyze their distribution.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color='purple')
    
    # Add a vertical line at 0 for reference
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    city_name = target_city.capitalize() if target_city else "All Cities"
    
    plt.title(f'Residuals Distribution ({city_name})')
    plt.xlabel('Error (Actual - Predicted) [PLN]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    
    filename = f"residuals_{city_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()

def plot_residuals_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, target_city: str):
    """
    Generates a scatter plot of residuals vs predicted values to check for heteroscedasticity.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color='green')
    
    # Add a horizontal line at 0 for reference
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    city_name = target_city.capitalize() if target_city else "All Cities"
    
    plt.title(f'Residuals vs Predicted Prices ({city_name})')
    plt.xlabel('Predicted Price [PLN]')
    plt.ylabel('Residuals (Actual - Predicted) [PLN]')
    plt.legend()
    plt.tight_layout()
    
    filename = f"residuals_vs_predicted_{city_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()

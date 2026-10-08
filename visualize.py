import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from xgboost import plot_importance

# Visualize actual vs predicted prices

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, target_city: str, model: str):

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
    
    filename = f"{model}/actual_vs_predicted_{city_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, target_city: str, model: str):
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
    
    filename = f"{model}/residuals_{city_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()

def plot_residuals_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, target_city: str, model: str):
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
    
    filename = f"{model}/residuals_vs_predicted_{city_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()

def feature_importance(model, target_city: str, max_features: int = 15):

    plot_importance(model, importance_type = 'gain', max_num_features = max_features)
    plt.title(f'XGBoost feature importance')
    plt.tight_layout()
    plt.savefig('XGB/feature_importance.png')
    plt.show()
    

def shap_summary(model, X_test, target_city: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist(), 
                      plot_size=(10, 8), show=False)
    plt.title(f"SHAP Values for {target_city.capitalize()}")
    plt.tight_layout()
    plt.savefig("XGB/xgb_shap_summary.png", dpi=150)
    plt.show()

def shap_dependence(model, X: pd.DataFrame, target_city: str, top_k: int = 5, samples: int = 2000):

    X_sample = X.sample(n=samples, random_state=42)
    X_sample = X_sample.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_sample)
    shap_values = np.asarray(shap_values)

    feature_names = X_sample.columns.tolist()

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names)

    top_features = importance.sort_values(ascending=False).head(top_k).index.to_list()

    for feature in top_features:
        feature_index = feature_names.index(feature)

        shap.dependence_plot(
            feature_index, 
            shap_values, 
            X_sample, 
            feature_names = feature_names,
            show=False
        )

        plt.title(f'SHAP Dependencies')
        plt.tight_layout()
        plt.savefig(f'XGB/dependence_{feature}.png')
        plt.show()
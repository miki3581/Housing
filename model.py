import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Splitting data into train and test sets
def split_data(df: pd.DataFrame, target_col: str = 'price', test_size: float = 0.2, random_state: int = 42):

    # Separate features (X) and target variable (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Perform the split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Scaling numerical features
def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):

    scaler = StandardScaler()
    
    # Identify continuous columns
    continuous_cols = [col for col in X_train.columns if X_train[col].nunique() > 2]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit on train data, transform both train and test data
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    return X_train_scaled, X_test_scaled

# Train Linear Regression model
def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
 
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model performance
def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:

    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Evaluate feature importance
def evaluate_feature_importance(model: LinearRegression, feature_names: pd.Index) -> pd.DataFrame:

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    return importance_df.sort_values(by='Coefficient', ascending=False)

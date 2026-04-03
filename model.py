import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

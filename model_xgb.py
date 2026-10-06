import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from model import save_model
from pathlib import Path

def split_data(df: pd.DataFrame, 
               target_col = 'price', 
               test_size = 0.3, 
               random_state = 42) -> tuple:
    
    X = df.drop(columns = [target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size = test_size, random_state = random_state)


def optimize_params(
        df: pd.DataFrame, 
        target_col: str = 'price', 
        param_grid: dict = None, 
        search_type: str = 'random', 
        n_splits: int = 5,
        random_state: int = 42,
        scoring: str = 'neg_mean_squared_error',
        n_jobs: int = -1,
        n_iter: int = 50,
    ) -> tuple:

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if param_grid is None:
        param_grid = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': loguniform(0.001, 0.3),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3),
            'gamma': loguniform(0.001, 1000),
            'reg_alpha': loguniform(0.001, 10),
            'reg_lambda': loguniform(0.1, 50)
        }

    xgb_model = XGBRegressor(random_state=42)

    if search_type == 'grid':
        cv_search = GridSearchCV(
            estimator = xgb_model, 
            param_grid = param_grid, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            cv=n_splits,
            verbose = 1
            )
    elif search_type == 'random':
        cv_search = RandomizedSearchCV(
            estimator=xgb_model, 
            param_distributions=param_grid, 
            n_iter=n_iter, 
            cv=n_splits, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            verbose = 1, 
            random_state=random_state)

    cv_search.fit(X, y)

    best_model = cv_search.best_estimator_
    best_params = cv_search.best_params_

    print(f"\nBest CV MAE: {-cv_search.best_score_:.3f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v:.6f}")

    return best_model, best_params, pd.DataFrame(cv_search.cv_results_)

def train_xgb_model(X: pd.DataFrame, y: pd.Series, best_params: dict, target_col: str = 'price', param_grid: dict = None) -> tuple:

    model = XGBRegressor(**best_params, random_state = 42)

    model.fit(X, y)

    return model

def load_or_train_model(X: pd.DataFrame, y: pd.Series, model_path: str = 'XGB/housing_model.joblib', retrain: bool = False) -> tuple:

    model_filename = Path(model_path)

    if model_filename.exists() and not retrain:
        print(f"Model found. Loading...")
        try:
            artifact = joblib.load(model_filename)
            model = artifact['model']
            feature_names = artifact.get('feature_names', None)

            return model, feature_names
        
        except Exception as e:
            print(f'Failed to load model')

    best_model, best_params, cv_result = optimize_params(pd.concat([X, y], axis = 1), target_col='price')
    final_model = train_xgb_model(X, y, best_params)
    model_filename = "XGB/housing_model.joblib"
    artifacts_to_save = {'model': final_model, 'feature_names': list(X)}
    save_model(artifacts_to_save, model_filename)


    return final_model, list(X.columns)
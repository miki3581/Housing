import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    print({"Best parameters:"})
    for i, j in best_params.items():
        print(f"{i}: {j}")

    return best_model, best_params, pd.DataFrame(cv_search.cv_results_)

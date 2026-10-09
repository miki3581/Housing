import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
from pathlib import Path


# Evaluate LightGBM model performance
def evaluate_model_lgbm(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Symmetric MAPE
    denom = (np.abs(y_test) + np.abs(y_pred)) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        smape_vals = np.where(denom == 0, 0, 2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))
    smape = np.mean(smape_vals)

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape}


# Hyperparameter optimization using lgbm.cv
def optimize_params_lgbm(
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int = 50,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> tuple:

    # Base parameters (fixed)
    base_params = {
        'objective': 'regression_l1',
        'metric': ['l1', 'rmse'],
        'verbose': -1,
        'random_state': random_state,
        'n_jobs': -1,
        'feature_pre_filter': False
    }

    # Search space for randomized search
    param_distributions = {
        'num_leaves': randint(15, 127),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.001, 0.3),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'reg_alpha': loguniform(0.001, 10),
        'reg_lambda': loguniform(0.1, 50),
        'min_child_samples': randint(5, 100),
    }

    dtrain = lgb.Dataset(X, label=y)

    best_mae = np.inf
    best_params = None
    best_cv_results = None
    best_iteration = 0

    print(f"\nRunning LightGBM CV optimization ({n_iter} iterations, {n_folds}-fold)...")

    for i in range(n_iter):
        # Sample random parameters
        sampled = {
            'num_leaves': param_distributions['num_leaves'].rvs(random_state=random_state + i),
            'max_depth': param_distributions['max_depth'].rvs(random_state=random_state + i + 1000),
            'learning_rate': param_distributions['learning_rate'].rvs(random_state=random_state + i + 2000),
            'subsample': param_distributions['subsample'].rvs(random_state=random_state + i + 3000),
            'colsample_bytree': param_distributions['colsample_bytree'].rvs(random_state=random_state + i + 4000),
            'reg_alpha': param_distributions['reg_alpha'].rvs(random_state=random_state + i + 5000),
            'reg_lambda': param_distributions['reg_lambda'].rvs(random_state=random_state + i + 6000),
            'min_child_samples': param_distributions['min_child_samples'].rvs(random_state=random_state + i + 7000),
        }

        params = {**base_params, **sampled}

        # Run lgbm.cv
        cv_results = lgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=n_folds,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            return_cvbooster=False,
        )

        # Extract validation MAE (l1 metric)
        cv_mae = cv_results['valid l1-mean'][-1]
        cv_rmse = cv_results['valid rmse-mean'][-1]
        iteration = len(cv_results['valid l1-mean'])

        print(f"  Iter {i+1:2d}/{n_iter} | MAE: {cv_mae:,.0f} | RMSE: {cv_rmse:,.0f} | Rounds: {iteration}")

        # Track best
        if cv_mae < best_mae:
            best_mae = cv_mae
            best_params = sampled
            best_cv_results = cv_results
            best_iteration = iteration

    print(f"\nBest CV MAE: {best_mae:,.0f}")
    print(f"Best iteration: {best_iteration}")
    print("Best parameters:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    return best_params, best_iteration, best_cv_results


# Train LightGBM model on full training set
def train_lgbm_model(X: pd.DataFrame, y: pd.Series, best_params: dict, best_iteration: int) -> lgb.Booster:

    params = {
        'objective': 'regression_l1',
        'metric': ['l1', 'rmse'],
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        **best_params,
    }

    dtrain = lgb.Dataset(X, label=y)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=best_iteration,
    )

    return model


# Load existing model or train a new one
def load_or_train_model_lgbm(X: pd.DataFrame, y: pd.Series, city: str, retrain: bool = False) -> lgb.Booster:

    model_filename = Path(f"LightGBM/{city}_model.joblib")

    if model_filename.exists() and not retrain:
        print(f"\nModel found at {model_filename}. Loading...")
        try:
            model = joblib.load(model_filename)
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Retraining...")

    # Optimize hyperparameters
    best_params, best_iteration, cv_results = optimize_params_lgbm(X, y)

    # Train final model on full training set
    print("\nTraining final LightGBM model...")
    model = train_lgbm_model(X, y, best_params, best_iteration)

    # Save model
    model_filename.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

    return model


# Save model to file
def save_model_lgbm(model, filename: str):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model successfully saved to {filename}")


# Load model from file
def load_model_lgbm(filename: str):
    model = joblib.load(filename)
    print(f"Model successfully loaded from {filename}")
    return model

# LightGBM-safe feature name cleaning
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    cleaned = []

    for col in df.columns:
        col = str(col)

        # Replace anything that is not a letter, number, or underscore
        col = re.sub(r'[^0-9A-Za-z_]+', '_', col)

        # Remove leading/trailing underscores
        col = col.strip('_')

        # If the column name became empty, give it a generic name
        cleaned.append(col if col else 'feature')

    # Make sure all column names are unique
    used = set()
    final_cols = []

    for col in cleaned:
        if col not in used:
            final = col
        else:
            i = 1
            while f'{col}_{i}' in used:
                i += 1
            final = f'{col}_{i}'

        used.add(final)
        final_cols.append(final)

    df.columns = final_cols

    return df

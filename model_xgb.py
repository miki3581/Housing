import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import randint, uniform, loguniform

param_grid = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': loguniform(0.001, 0.3),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'gamma': loguniform[0.001, 1],
    'reg_alpha': loguniform[0, 100],
    'reg_lambda': loguniform[0, 1000]
}

xgb_model = XGBRegressor(random_state=42)
import pandas as pd
import numpy as np
import pickle
from scipy.stats import describe 

from preprocess import load_data_reg, reg_model_report

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.linear_model import RidgeCV

import sys

data = load_data_reg("Heavy")

scaler = RobustScaler()
y_scaler = StandardScaler()

X = data.drop(["DR3", "Salvage", "Treatment", "TimeStep"], axis=1)


y = data['DR3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 10000, num = 100)]

criterion = ['mse', 'mae']

max_features = [int(x) for x in np.linspace(start=3, stop=X.shape[1], num=1)]

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]
oob_score = [True, False]

param_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion, 
    'max_features': max_features, 
    'max_depth': max_depth,
    'min_samples_split': min_samples_split, 
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap, 
    'oob_score': oob_score
}

search = RandomForestRegressor()

# search = RandomizedSearchCV(rf, param_grid, n_jobs=3)

search.fit(X_train, y_train)
preds = search.predict(X_test)

train_preds = search.predict(X_train)

reg_model_report(preds, train_preds, y_test, y_train)

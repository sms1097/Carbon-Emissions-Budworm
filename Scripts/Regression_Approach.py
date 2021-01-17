import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  mean_squared_error

from xgboost import XGBRegressor

import sys

data = pd.read_csv("/home/sean/ML/optimization/Exact-Rule-Learning/Data/Classifier_Inputs.csv")

salvage = data[(data['TimeStep'] == 40) & (data['Treatment'] == "NoMgmt") & (data['Salvage'] == 'Salvage')]
salvage = salvage.set_index("StandID")
salvage = salvage.fillna(salvage.median())

no_salvage = data[(data['TimeStep'] == 40) & (data['Treatment'] == "NoMgmt") & (data['Salvage'] == 'NoSalvage')]
no_salvage = no_salvage.set_index("StandID")
no_salvage = no_salvage.fillna(no_salvage.median())

scaler = RobustScaler()
y_scaler = RobustScaler()

X_salvage = salvage.drop(["NoDR", "DR1", "DR5", "DR3", "Treatment", "Salvage", "TimeStep"], axis=1)
X_salvage = scaler.fit_transform(X_salvage)

y_salvage = salvage.DR5

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_salvage, y_salvage, test_size=0.2)

parameters = {
    'booster':['dart'],
    'max_depth': [4, 6, 8, 10], 
    'subsample': [0.25, 0.5, 0.75, 1],
    'colsample_bytree': [0.25, 0.5, 0.75, 1],
    'colsample_bylevel': [0.25, 0.5, 0.75, 1],
    'colsample_bynode': [0.25, 0.5, 0.75, 1],
    'nthread': [4]
}

xgb = XGBRegressor() 
regr = GridSearchCV(xgb, parameters, n_jobs=2)
regr.fit(X_s_train, y_s_train)
preds_salvage = regr.predict(X_s_test)

rmse = mean_squared_error(y_s_test, preds_salvage, squared=False)
print(rmse)

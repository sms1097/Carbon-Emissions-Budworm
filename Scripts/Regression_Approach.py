import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import  mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import sys

data = pd.read_csv("/home/sean/ML/optimization/Exact-Rule-Learning/Data/Classifier_Inputs.csv")

salvage = data[(data['TimeStep'] == 40) & (data['Treatment'] == "NoMgmt") & (data['Salvage'] == 'Salvage')]
salvage = salvage.set_index("StandID")
salvage = salvage.fillna(salvage.median())

no_salvage = data[(data['TimeStep'] == 40) & (data['Treatment'] == "NoMgmt") & (data['Salvage'] == 'NoSalvage')]
no_salvage = no_salvage.set_index("StandID")
no_salvage = no_salvage.fillna(no_salvage.median())

data = salvage.copy()
data["dif"] = no_salvage.DR3 - data.DR3

scaler = RobustScaler()

X = data.drop(["NoDR", "DR1", "DR5", "DR3", "Treatment", "Salvage", "TimeStep", "dif"], axis=1)
X = scaler.fit_transform(X)


y = data.dif

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## This is training XGBoost 
# parameters = {
#     'booster':['dart'],
#     'max_depth': [4, 6, 8, 10], 
#     'subsample': [0.25, 0.5, 0.75, 1],
#     'colsample_bytree': [0.25, 0.5, 0.75, 1],
#     'colsample_bylevel': [0.25, 0.5, 0.75, 1],
#     'colsample_bynode': [0.25, 0.5, 0.75, 1],
#     'nthread': [4]
# }

# xgb = XGBRegressor() 
# regr = GridSearchCV(xgb, parameters, n_jobs=2)
# regr.fit(X_s_train, y_s_train)
# preds_salvage = regr.predict(X_s_test)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
}

rf = RandomForestRegressor()
search = RandomizedSearchCV(rf, random_grid, n_jobs=-1)

with open('/home/sean/ML/optimization/Exact-Rule-Learning/model/random.pickle', 'wb') as handle:
    pickle.dump(search, handle, protocol=pickle.HIGHEST_PROTOCOL)

search.fit(X_train, y_train)

preds = search.predict(X_test)

rmse = mean_squared_error(y_test, preds, squared=False)
print(rmse)


with open('/home/sean/ML/optimization/Exact-Rule-Learning/model/random.pickle', 'wb') as handle:
    pickle.dump(search, handle, protocol=pickle.HIGHEST_PROTOCOL)

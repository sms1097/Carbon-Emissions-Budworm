import pandas as pd
import numpy as np
import pickle
from preprocess import load_data_class, model_report

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
 
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from imodels import SkopeRulesClassifier

data = load_data_class('Heavy', 'DR5')
dummies = pd.get_dummies(data['Treatment'])

scaler = RobustScaler()

X = data.drop(['Voucher', 'Treatment', 'Salvage', 'TimeStep'], axis=1)
X = scaler.fit_transform(X)

y = data['Voucher']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Random Search
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 10000, num = 100)]
# Number of features to consider at every split
max_features = [int(x) for x in np.linspace(start=3, stop=X.shape[1], num=1)]
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
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
}

# Accuracy: 0.777232746955345
# Precision: 0.801490514905149
# Recall: 0.7639651275427833
# Train Accuracy: 0.8425393334461174
# {'n_estimators': 1090, 
# 'min_samples_split': 10, 
# 'min_samples_leaf': 2, 
# 'max_features': 3, 
# 'max_depth': 90, 
# 'bootstrap': True}

f = RandomForestClassifier()

rf = RandomizedSearchCV(f, param_grid, n_jobs=3)

rf.fit(X_train, y_train)
preds = rf.predict(X_test)
train_preds = rf.predict(X_train)

model_report(preds, train_preds, y_test, y_train)

with open('model/heavy_classifier', 'wb') as handle:
    pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)

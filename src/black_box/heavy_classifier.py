import pandas as pd
import numpy as np
import pickle
from preprocess import load_data_class, model_report

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
 
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


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


## RESULTS
# {'n_estimators': 6909, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 3, 'max_depth': 70, 'bootstrap': True} 
# training  99.2%
# test 78.7%

# rf = RandomForestClassifier(
    # n_estimators=6909, 
    # min_samples_split=2, 
    # min_samples_leaf=2, 
    # max_features=3, 
    # max_depth=70, 
    # bootstrap=True,
    # n_jobs=-1
# )

print(data['Voucher'].value_counts())

search = AdaBoostClassifier(
    base_estimator=base, 
    n_estimators=1000
)

# search = RandomizedSearchCV(rf, param_grid, n_jobs=3)
search.fit(X_train, y_train)
preds = search.predict(X_test)
train_preds = search.predict(X_train)

model_report(preds, train_preds, y_test, y_train)

# with open('../model/NoMgmt_classifier', 'wb') as handle:
#     pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)

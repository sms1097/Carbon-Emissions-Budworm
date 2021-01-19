import pandas as pd
import numpy as np
from preprocess import load_data, model_report

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

data = load_data()
scaler = RobustScaler()

X = data.drop(['Voucher', 'Treatment', 'Salvage', 'TimeStep'], axis=1)
X = scaler.fit_transform(X)

y = data['Voucher']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# param_grid = {
#     'n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=50)],
#     ''
# }



clf = AdaBoostClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
train_preds = clf.predict(X_train)
preds = clf.predict(X_test)

model_report(preds, train_preds, y_test, y_train)

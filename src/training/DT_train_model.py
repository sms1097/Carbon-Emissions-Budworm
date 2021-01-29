import pandas as pd
import numpy as np
import pickle

from preprocess import load_data_class, model_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV 


def train_model(management, discount):
    data = load_data_class(management, discount)

    scaler = RobustScaler()

    X = data.drop(['Voucher', 'Treatment', 'Salvage', 'TimeStep'], axis=1)
    X = scaler.fit_transform(X)

    y = data['Voucher']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    param_grid = {
        'class_weight': ['balanced', None],
        'min_samples_leaf': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['auto', None],
        'class_weight': [{0:0.1, 1:0.9}]
    }

    tree = DecisionTreeClassifier()
    f = GridSearchCV(tree, param_grid, n_jobs=3)

    f.fit(X_train, y_train)
    preds = f.predict(X_test)
    train_preds = f.predict(X_train)

    model_report(management, discount, preds, train_preds, y_test, y_train)

    with open('model/DT/' + discount + '/' + management, 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__": 
    mgmts = ['Comm-Ind', 'Heavy', 'HighGrade', 'Light', 'NoMgmt', 'Moderate']
    discounts = ['NoDR', 'DR1', 'DR3', 'DR5']
    for mgmt in mgmts:
        for discount in discounts:
            train_model(mgmt, discount)

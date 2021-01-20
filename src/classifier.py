import pickle 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class Classifier():
    def __init__(self):
        self.comm_ind = pickle.load(open('model/comm_ind_classifier', 'rb'))
        self.heavy = pickle.load(open('model/heavy_classifier', 'rb'))
        self.high_grade = pickle.load(open('model/heavy_classifier', 'rb'))
        self.light = pickle.load(open('model/light_classifier', 'rb'))
        self.moderate = pickle.load(open('model/moderate_classifier', 'rb'))
        self.no_mgmt = pickle.load(open('model/nomgmt_classifier', 'rb'))


        
    def _get_strategy(self, data, target, management):
        ## Get Optimal Strategy
        strategy = target.rename('strategy')
        salvage = pd.merge(data, strategy, on="StandID")
        salvage_strategy = salvage[
            (salvage['strategy'] == True) &
            (salvage['Salvage'] == 'NoSalvage') &
            (salvage['TimeStep'] == 40) &
            (salvage['Treatment'] == management)
        ].DR5.sum()
        
        no_salvage = pd.merge(data, strategy, on="StandID")
        no_salvage_strategy = no_salvage[
            (no_salvage['strategy'] == False) &
            (no_salvage['Salvage'] == 'Salvage') &
            (no_salvage['TimeStep'] == 40) &
            (no_salvage['Treatment'] == management)
        ].DR5.sum()
        
        outcome = (salvage_strategy + no_salvage_strategy) / target.shape[0]

        return outcome
        
    def predict(self, data, management, return_y=False):
        salvage = data[
            (data['TimeStep'] == 40) & 
            (data['Treatment'] == management) & 
            (data['Salvage'] == 'Salvage')
        ]

        salvage = salvage.fillna(salvage.mean())

        no_salvage = data[
            (data['TimeStep'] == 40) & 
            (data['Treatment'] == management) & 
            (data['Salvage'] == 'NoSalvage')
        ]

        no_salvage = no_salvage.fillna(no_salvage.mean())

        target = salvage.copy()
        target["DR5"] -= no_salvage["DR5"]

        target['Voucher'] = (target["DR5"] > 0)

        scaler = RobustScaler()
        X = target.drop(['Voucher', 'Treatment', 'NoDR', 'DR5', 'DR1', 'DR3', 'Salvage', 'TimeStep'], axis=1)
        X = scaler.fit_transform(X)
        
        y = target['Voucher']
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        _, _, _, y_no_salvage = train_test_split(X, no_salvage['DR5'], test_size=0.2, random_state=1)
        _, _, _, y_salvage = train_test_split(X, salvage['DR5'], test_size=0.2, random_state=1)
        
        if management == 'Comm-Ind':
            out = self.comm_ind.predict(X_test)
        elif management == 'Heavy':
            out = self.heavy.predict(X_test)
        elif management == 'HighGrade':
            out = self.high_grade.predict(X_test)
        elif management == 'Light':
            out = self.light.predict(X_test)
        elif management == 'Moderate':
            out = self.moderate.predict(X_test)
        elif management == 'NoMgmt':
            out = self.no_mgmt.predict(X_test)
        
        preds = pd.Series(out)
        preds.index = y_test.index.tolist()
        preds = preds.rename_axis('StandID')

        return {
            'preds':out, 
            'test': y_test, 
            'optimal_strategy': self._get_strategy(data, y_test, management), 
            'no_salvage_strategy': np.mean(y_no_salvage), 
            'salvage_strategy': np.mean(y_salvage),
            'model_strategy': self._get_strategy(data, preds, management)
        }
    
import pickle 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class Classifier():
    def __init__(self, model, discount_rate):
        base = 'model/' + model + '/' + discount_rate + '/'
        self.comm_ind = pickle.load(open(base + 'Comm-Ind', 'rb'))
        self.heavy = pickle.load(open(base + 'Heavy', 'rb'))
        self.high_grade = pickle.load(open(base + 'HighGrade', 'rb'))
        self.light = pickle.load(open(base + 'Light', 'rb'))
        self.moderate = pickle.load(open(base + 'Moderate', 'rb'))
        self.no_mgmt = pickle.load(open(base + 'NoMgmt', 'rb'))
        self.discount = discount_rate
        
    def _get_strategy(self, data, target, management):
        ## Get Optimal Strategy
        strategy = target.rename('strategy')
        salvage = pd.merge(data, strategy, on="StandID")
        salvage_strategy = salvage[
            (salvage['strategy'] == True) &
            (salvage['Salvage'] == 'NoSalvage') &
            (salvage['TimeStep'] == 40) &
            (salvage['Treatment'] == management)
        ]
        salvage_strategy = salvage_strategy[self.discount]
        
        no_salvage = pd.merge(data, strategy, on="StandID")
        no_salvage_strategy = no_salvage[
            (no_salvage['strategy'] == False) &
            (no_salvage['Salvage'] == 'Salvage') &
            (no_salvage['TimeStep'] == 40) &
            (no_salvage['Treatment'] == management)
        ]
        no_salvage_strategy = no_salvage_strategy[self.discount]
        
        # Make sure we don't duplicate
        assert target.shape[0] == salvage_strategy.shape[0] + no_salvage_strategy.shape[0]
        
        # Really make sure we don't duplicate
        a = salvage_strategy.index.tolist()
        b = no_salvage_strategy.index.tolist()
        assert len(set(a).intersection(set(b))) == 0
        
        outcome = (salvage_strategy.sum() + no_salvage_strategy.sum()) / target.shape[0]

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
        target[self.discount] -= no_salvage[self.discount]

        target['Voucher'] = (target[self.discount] > 0)

        scaler = RobustScaler()
        X = target.drop(['Voucher', 'Treatment', 'NoDR', 'DR5', 'DR1', 'DR3', 'Salvage', 'TimeStep'], axis=1)
        X = scaler.fit_transform(X)
        
        y = target['Voucher']
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        _, _, _, y_no_salvage = train_test_split(X, no_salvage[self.discount], test_size=0.2, random_state=1)
        _, _, _, y_salvage = train_test_split(X, salvage[self.discount], test_size=0.2, random_state=1)
        
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
            'preds': out, 
            'test': y_test, 
            'optimal_strategy': self._get_strategy(data, y_test, management), 
            'no_salvage_strategy': np.mean(y_no_salvage), 
            'salvage_strategy': np.mean(y_salvage),
            'model_strategy': self._get_strategy(data, preds, management)
        }
    
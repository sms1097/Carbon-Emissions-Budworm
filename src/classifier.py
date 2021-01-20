import pickle 
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


    def predict(self, data, management, return_y=False):
        salvage = data[
            (data['TimeStep'] == 40) & 
            (data['Treatment'] == management) & 
            (data['Salvage'] == 'Salvage')
        ]

        salvage = salvage.set_index("StandID")
        salvage = salvage.fillna(salvage.mean())

        no_salvage = data[
            (data['TimeStep'] == 40) & 
            (data['Treatment'] == management) & 
            (data['Salvage'] == 'NoSalvage')
        ]

        no_salvage = no_salvage.set_index("StandID")
        no_salvage = no_salvage.fillna(no_salvage.mean())

        data = salvage.copy()
        data["DR5"] -= no_salvage["DR5"]

        data['Voucher'] = (data["DR5"] > 0)

        scaler = RobustScaler()
        X = data.drop(['Voucher', 'Treatment', 'NoDR', 'DR5', 'DR1', 'DR3', 'Salvage', 'TimeStep'], axis=1)
        X = scaler.fit_transform(X)

        y = data['Voucher']

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


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

        return out, y_test

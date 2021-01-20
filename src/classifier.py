import pickle 
from sklearn.preprocessing import RobustScaler


class Classifier():
    def __init__(self):
        self.comm_ind = pickle.load(open('model/comm_ind_classifier', 'rb'))
        self.heavy = pickle.load(open('model/heavy_classifier', 'rb'))
        self.high_grade = pickle.load(open('model/heavy_classifier', 'rb'))
        self.light = pickle.load(open('model/light_classifier', 'rb'))
        self.moderate = pickle.load(open('model/moderate_classifier', 'rb'))
        self.no_mgmt = pickle.load(open('model/nomgmt_classifier', 'rb'))


    def predict(X_test, treatment):
        if treatment == 'comm_ind':
            out = self.comm_ind.predict(X_test)
        elif treatment == 'heavy':
            out = self.heavy.predict(X_test)
        elif treatment == 'high_grade':
            out = self.high_grade.predict(X_test)
        elif treatment == 'light':
            out = self.light.predict(X_test)
        elif treatment == 'moderate':
            out = self.moderate.predict(X_test)
        elif treatment == 'nomgmt':
            out = self.no_mgmt.predict(X_test)

        return out

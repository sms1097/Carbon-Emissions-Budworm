import pandas as pd 
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error


def base_load(management, discount):

    rates = ["NoDR", "DR1", "DR3", "DR5"]

    if discount not in rates:
        raise ValueError("Invalid Discount Rate!")
    
    data = pd.read_csv("../Data/Classifier_Inputs.csv")

    salvage = data[
        (data['TimeStep'] == 40) & 
        # (data['Treatment'] == management) & 
        (data['Salvage'] == 'Salvage')
    ]

    salvage = salvage.set_index("StandID")
    salvage = salvage.fillna(salvage.mean())

    no_salvage = data[
        (data['TimeStep'] == 40) & 
        # (data['Treatment'] == management) & 
        (data['Salvage'] == 'NoSalvage')
    ]

    no_salvage = no_salvage.set_index("StandID")
    no_salvage = no_salvage.fillna(no_salvage.mean())

    data = no_salvage.copy()
    data[discount] -= salvage[discount]


    return data

def load_data_class(management, discount="DR3"):
    rates = ["NoDR", "DR1", "DR3", "DR5"]

    data = base_load(management, discount)
    data['Voucher'] = (data[discount] > 0)

    data = data.drop(rates, axis=1)

    return data

def load_data_reg(management, discount="DR3"):
    data = base_load(management, discount)

    rates = ["NoDR", "DR1", "DR3", "DR5"]
    rates.remove(discount)

    data = data.drop(rates, axis=1)

    return data

def model_report(test_preds, train_preds, y_test, y_train):
    print('Accuracy: {}'.format(accuracy_score(y_test, test_preds)))
    print('Precision: {}'.format(precision_score(y_test, test_preds)))
    print('Recall: {}'.format(recall_score(y_test, test_preds)))
    print('Train Accuracy: {}'.format(accuracy_score(y_train, train_preds)))

def reg_model_report(test_preds, train_preds, y_test, y_train):
    print('RMSE Test: {}'.format(mean_squared_error(y_test, test_preds, squared=False)))
    print('RMSE Train: {}'.format(mean_squared_error(y_train, train_preds, squared=False)))

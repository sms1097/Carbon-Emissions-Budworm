import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  mean_squared_error


data = pd.read_csv("/home/sean/ML/optimization/Exact-Rule-Learning/Data/Classifier_Inputs.csv")

salvage = data[(data['TimeStep'] == 40) & (data['Treatment'] == "NoMgmt") & (data['Salvage'] == 'Salvage')]
salvage = salvage.set_index("StandID")
salvage = salvage.fillna(salvage.median())

no_salvage = data[(data['TimeStep'] == 40) & (data['Treatment'] == "NoMgmt") & (data['Salvage'] == 'NoSalvage')]
no_salvage = no_salvage.set_index("StandID")
no_salvage = no_salvage.fillna(no_salvage.median())

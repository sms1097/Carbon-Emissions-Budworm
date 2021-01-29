import pandas as pd
from classifier import Classifier


def F(mng, base):
    num = 1 - (mng['model_strategy'] - mng['optimal_strategy']) 
    dom = mng['no_salvage_strategy'] - mng['optimal_strategy'] 

    f = base.copy()
    f[-2] = 'F'
    f[-1] = num / dom

    return f


def G(mng, base):
    num = mng['no_salvage_strategy'] - mng['optimal_strategy']
    dom = mng['model_strategy'] - mng['optimal_strategy']

    g = base.copy()
    g[-2] = 'G'
    g[-1] = num / dom

    return g

def improvement(mng, base):
    strict = min(mng['salvage_strategy'], mng['no_salvage_strategy']) - mng['optimal_strategy']
    model = mng['model_strategy'] - mng['optimal_strategy']


    improv = base.copy()    
    improv[-2] = 'times'
    improv[-1] = strict / model

    return improv


if __name__ == "__main__":
    data = pd.read_csv('data/Classifier_Inputs.csv')
    data = data.set_index('StandID')

    discounts = ['NoDR', 'DR1', 'DR3', 'DR5']
    models = ['DT', 'LogReg', 'RF']
    managements = ['Heavy', 'NoMgmt', 'Moderate', 'Light', 'Comm-Ind', 'HighGrade']

    cols = ['Model', 'Managment', 'Discount', 'Metric', 'Result']
    df = pd.DataFrame(columns=cols)

    for model in models:
        for management in managements:
            for discount in discounts:
                clf = Classifier(model, discount)
                context = clf.predict(data, management)

                base = [model, management, discount, None, None]
                temp = pd.DataFrame(
                    [
                        F(context, base),
                        G(context, base),
                        improvement(context, base)
                    ],
                    columns=cols
                )

                df = df.append(temp)

    df.to_csv('data/Model_Outputs.csv')

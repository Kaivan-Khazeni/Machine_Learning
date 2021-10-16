
import pandas as pd
import numpy as np

def gradient_descent(S):
    return "hi"



if __name__ == '__main__':

    labels_value = ['unacc', 'acc', 'good', 'vgood']
    # label is where the values of each item is stored.  It is not binary,
    # it is unacceptable, acceptable, good, very good.  This is different

    # then before
    attributes = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'output']
    df = pd.read_csv('concrete/train.csv', names=attributes)
    df_test = pd.read_csv('concrete/test.csv', names=attributes)
    print(df.head(5))






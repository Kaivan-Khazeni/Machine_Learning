
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim


def take_data():
    attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    df_train = pd.read_csv('bank-note/train.csv', names=attributes)
    df_test = pd.read_csv('bank-note/test.csv', names=attributes)
    df_train.loc[df_train.label == 0, 'label'] = -1
    df_test.loc[df_test.label == 0, 'label'] = -1
    X = df_train[['variance', 'skewness', 'curtosis', 'entropy']].values
    y = df_train.label.values
    input_size = 4
    output_size = 1
    hidden_size_1 = 4
    hidden_size_2 = 4
    W1 = np.random.randn(input_size, hidden_size_1)
    W2 = np.random.randn(hidden_size_1, hidden_size_2)
    W3 = np.random.randn(hidden_size_2, output_size)

    return X, y, input_size, output_size, hidden_size_1,hidden_size_2, W1, W2, W3


def forward(X,W1,W2,W3):
    z = X.dot(W1)
    z2 = z.dot(W2)
    z3 = np.dot(z2,W3)
    output = sigmoid(z3)
    return output,z,z2,z3


def sigmoid(s,d =False):
    if d == True:
        return s*(1-s)
    else:
        return 1/(1+np.exp(-s))

def back_prop(X,y,output,W1,W2,W3,z,z2,z3):
    # y = np.asarray([[val]for val in y])
    output_error = y - output
    output_delta = output_error * sigmoid(output, d=True)

    z3_error = output_delta.dot(W3.T)
    z3_delta = z3_error * sigmoid(z3, d=True)
    z2_error = z3_delta.dot(W2.T)
    z2_delta = z2_error * sigmoid(z2, d=True)

    W1 += X.T.dot(z2_delta)
    W2 += z2.dot(z3_delta)
    W3 += z3.dot(output_delta)

    return W1, W2, W3

if __name__ == '__main__':
    X, y, input_size, output_size, hidden_size_1,hidden_size_2, W1, W2,W3 = take_data()
    output, z, z2,z3 = forward(X[0], W1, W2,W3)
    W1_B, W2_B,W3_B = back_prop(X[0], y[0], output, W1, W2,W3, z, z2,z3)


    print("output")
    print(output)
    print("Weight Vector 1")
    print(W1_B)
    print("Weight Vector 2")
    print(W2_B)
    print("Weight Vector 3")
    print(W3_B)

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def sub_grad_descent_a(T):
  attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
  df_train = pd.read_csv('bank-note-svm/train.csv', names =attributes)
  df_test = pd.read_csv('bank-note-svm/test.csv', names = attributes)
  df_train.insert(0, 'bias', 1)
  df_test.insert(0, 'bias', 1)
  df_train.loc[df_train.label == 0, 'label'] = -1
  df_test.loc[df_test.label == 0, 'label'] = -1
  hyper = [100/873,500/873,700/873]
  train_error = []
  test_error = []
  for j in range(len(hyper)):
    C = hyper[j]
    w = np.zeros(5)
    w_0 = np.zeros(4)
    y_0 = .1
    a = .001
    for t in range(1,T+1):
      y_t = (y_0/(1+((y_0/a)*t)))
      temp_train = df_train.sample(frac =1)
      X = temp_train[['bias','variance', 'skewness','curtosis', 'entropy']].values
      y = temp_train.label.values
      for i in range(len(X)):
        y_i = y[i]
        x_i = X[i]
        if y_i * w.T.dot(x_i) <= 1:
          temp = w_0
          w = w - y_t * np.insert(w_0,0,0) + y_t*C*873*(y_i*x_i)
          w_0 = temp
        else:
          w_0 = (1-y_t)*w_0
    X_test = df_test[['bias','variance', 'skewness','curtosis', 'entropy']].values
    y_test = df_test.label.values
    X_train = df_train[['bias','variance', 'skewness','curtosis', 'entropy']].values
    y_train = df_train.label.values
    err_count_train = 0
    err_count_test = 0
    for iter in range(len(df_train)):
      pred_train = np.sign(w.T.dot(X_train[iter]))
      if pred_train != y_train[iter]:
        err_count_train += 1
    for iter in range(len(df_test)):
      pred_test = np.sign(w.T.dot(X_test[iter]))
      if pred_test != y_test[iter]:
        err_count_test += 1
    train_error.append(err_count_train / len(df_train))
    test_error.append(err_count_test / len(df_test))
  return train_error,test_error

def sub_grad_descent_b(T):
  attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
  df_train = pd.read_csv('bank-note-svm/train.csv', names =attributes)
  df_test = pd.read_csv('bank-note-svm/test.csv', names = attributes)
  df_train.insert(0, 'bias', 1)
  df_test.insert(0, 'bias', 1)
  df_train.loc[df_train.label == 0, 'label'] = -1
  df_test.loc[df_test.label == 0, 'label'] = -1
  hyper = [100/873,500/873,700/873]
  train_error = []
  test_error = []
  for j in range(len(hyper)):
    C = hyper[j]
    w = np.zeros(5)
    w_0 = np.zeros(4)
    y_0 = .05
    a = .001
    for t in range(1,T+1):
      y_t = (y_0/(1+t))
      temp_train = df_train.sample(frac =1)
      X = temp_train[['bias','variance', 'skewness','curtosis', 'entropy']].values
      y = temp_train.label.values
      for i in range(len(X)):
        y_i = y[i]
        x_i = X[i]
        if y_i * w.T.dot(x_i) <= 1:
          temp = w_0
          w = w - (y_t * np.insert(w_0,0,0)) + y_t*C*873*(y_i*x_i)
          w_0 = temp
        else:
          w_0 = (1-y_t)*w_0
    X_test = df_test[['bias','variance', 'skewness','curtosis', 'entropy']].values
    y_test = df_test.label.values
    X_train = df_train[['bias','variance', 'skewness','curtosis', 'entropy']].values
    y_train = df_train.label.values
    err_count_train = 0
    err_count_test = 0
    for iter in range(len(df_train)):
      pred_train = np.sign(w.T.dot(X_train[iter]))
      if pred_train != y_train[iter]:
        err_count_train += 1
    for iter in range(len(df_test)):
      pred_test = np.sign(w.T.dot(X_test[iter]))
      if pred_test != y_test[iter]:
        err_count_test += 1
    train_error.append(err_count_train / len(df_train))
    test_error.append(err_count_test / len(df_test))
  return train_error,test_error

if __name__ == '__main__':
    train_err,test_err = sub_grad_descent_a(100)
    print("TRAINING ERROR : 2A")
    print(train_err)
    print("TESTING ERROR: 2A")
    print(test_err)

    train_err_B,test_err_B = sub_grad_descent_b(100)
    print("TRAINING ERROR : 2B")
    print(train_err_B)
    print("TESTING ERROR: 2B")
    print(test_err_B)









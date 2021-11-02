
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# perceptron will take in a training and testing data set, along with the T epochs
def perceptron(df_train, df_test, T):
    d = 4
    r = 1
    m = len(df_train)
    w = np.zeros(d+1)
    temp = df_test.copy
    error_count_arr = []
    df_test.insert(0, 'bias', 1)
    # Step 1: Iterate through 1 to T
    for i in range(1, T):
        error_count = 0
        # Step 2: Shuffle data
        temp_train = df_train.sample(frac=1)
        temp_train.insert(0, "bias", 1)
        X = temp_train[['bias','variance', 'skewness', 'curtosis', 'entropy']].values
        y = temp_train.label.values
        # Step 3 : Iterate through all of the data and update the weight vector
        #          if it passes the given condition
        for j in range(len(temp_train)):
            if y[j] * w.dot(X[j]) <= 0:
                w = w + y[j] * X[j]
        # Step 4: Find the predictions for the test data with the weight vector that was updated
        #         and find the error compared to the main testing dataframe
        pred_test = np.sign(w.T.dot(df_test[['bias','variance', 'skewness', 'curtosis', 'entropy']].values.T))

        for j in range(500):
            if pred_test[j] != df_test.label.values[j]:
                error_count += 1
        error_count_arr.append(error_count/500)
    #Step 5: Return the final weight vector after 10 epochs and also return the average prediction error
    return w, np.sum(error_count_arr)/10


# voted perceptron will use a series of weight vectors and counts of accuracy
# parameters are a train, test dataframe and T iteration count
def voted_perceptron(df_train,df_test,T):
  w = []
  w.append(np.zeros(5))
  # Store weight vectors and keep a m and C variable
  m = 0
  C = []
  C.append(0)
  error_arr = []
  df_test.insert(0, 'bias', 1)
  # Step 1: Iterate through T epochs
  for i in range(T):
    temp_train = df_train.sample(frac =1)
    temp_train.insert(0,"bias", 1)
    X = temp_train[['bias','variance', 'skewness','curtosis', 'entropy']].values
    y = temp_train.label.values
    # Step 2: Iterate through every example in this method
    for j in range(len(df_train)):
      y_i = y[j]
      X_i = X[j]
      # Step 3: If the condition is met, add a new weight vector and append to m and C.
      if y[j] * w[m].dot(X[j]) <= 0:
        w_m = w[m] + y[j]*X[j]
        w.append(w_m)
        C.append(1)
        m = m + 1
      else:
        C[m] += 1
    X_test = df_test[['bias','variance', 'skewness','curtosis', 'entropy']].values
    y_test = df_test.label.values
    pred_test = []
    error_test = 0
    # Step 4 : After iterating through all example, for each T, find error rate of the prediction using the current weight
    for j in range(len(df_test)):
      arr_ = []
      for iter in range(len(C)):
        arr_.append(C[iter] * np.sign(w[iter].T.dot(X_test[j])))
      pred_test.append(np.sign(np.sum(arr_)))
      if np.sign(np.sum(arr_)) != y_test[j]:
        error_test += 1
    error_arr.append(error_test/len(df_test))

  # returning the weight vectors, the Counts, and the average prediction
  return w[len(w)-1],np.array(w),C,np.sum(error_arr)/T


def average_perceptron(df_train, df_test, T):
    w = np.zeros(5)
    m = 0
    a = np.zeros(5)
    error_arr = []
    df_test.insert(0, 'bias', 1)
    #Step 1: Iterate through  1 to T (10)
    for i in range(1, T + 1):
        error_count = 0
        temp_train = df_train.sample(frac=1)
        temp_train.insert(0, "bias", 1)
        X = temp_train[['bias','variance', 'skewness', 'curtosis', 'entropy']].values
        y = temp_train.label.values
        # Step 2: Iterate through all examples of the shuffled array  and
        #         update the weight vector with also adding to a
        for j in range(len(temp_train)):
            if y[j] * w.dot(X[j]) <= 0:
                w = w + y[j] * X[j]
            a = a + w
        pred_test = np.sign(a.T.dot(df_test[['bias','variance', 'skewness', 'curtosis', 'entropy']].values.T))

        X_test = df_test[['bias','variance', 'skewness', 'curtosis', 'entropy']].values
        y_test = df_test.label.values
        pred_test = np.sign(a.T.dot(df_test[['bias','variance', 'skewness', 'curtosis', 'entropy']].values.T))

        for j in range(500):
            if pred_test[j] != df_test.label.values[j]:
                error_count += 1
        error_arr.append(error_count / 500)

    return w,np.sum(error_arr) / 10

if __name__ == '__main__':
    # then before
    attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    df_train = pd.read_csv('bank-note/train.csv', names = attributes)
    df_test = pd.read_csv('bank-note/test.csv', names = attributes)
    df_train.loc[df_train.label == 0, 'label'] = -1
    df_test.loc[df_test.label == 0, 'label'] = -1

    #Perceptron Method and Print
    #*first term in weight vector is resulting bias
    w,avg_pred_error = perceptron(df_train,df_test,10)
    print("Final Weight Vector for Standard Perceptron")
    print(w)
    print("Average Prediction Error for Standard Perceptron")
    print(avg_pred_error)

    attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    df_train = pd.read_csv('bank-note/train.csv', names=attributes)
    df_test = pd.read_csv('bank-note/test.csv', names=attributes)
    df_train.loc[df_train.label == 0, 'label'] = -1
    df_test.loc[df_test.label == 0, 'label'] = -1

    #Voted Method and Print with weight vector and count
    #                           *first term in weight vector is resulting bias
    w_final, w,c, avg_pred_error = voted_perceptron(df_train, df_test, 10)
    # Prints counts then the weight vector, i used this for the CSV i shared.
    for i in range(len(c)):
        print(c[i])
    for i in range(len(w)):
        print(w[i])
    print("Final Weight Vector for Voted Perceptron")
    print(w_final)
    print("Average Prediction Error for Voted Perceptron")
    print(avg_pred_error)

    attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    df_train = pd.read_csv('bank-note/train.csv', names=attributes)
    df_test = pd.read_csv('bank-note/test.csv', names=attributes)
    df_train.loc[df_train.label == 0, 'label'] = -1
    df_test.loc[df_test.label == 0, 'label'] = -1

    #Average Method and Print
    #*first term in weight vector is resulting bias
    w, avg_pred_error = average_perceptron(df_train, df_test, 10)
    print("Final Weight Vector for Average Perceptron")
    print(w)
    print("Average Prediction Error for Average Perceptron")
    print(avg_pred_error)







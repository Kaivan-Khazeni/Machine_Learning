
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def cost_function(X,y, theta):
    m = len(X)
    cost = np.sum((X.dot(theta) - y)**2)/2
    return cost


def batch_gradient_descent(S, r, iter):
    X = S[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']]
    y = S.output
    cost_history = [0] * iter
    theta = np.zeros(7)
    m = len(X)
    theta_arr = []

    converged = False
    for i in range(iter):
        pred = X.dot(theta)
        loss = pred - y
        grad = X.T.dot(loss)
        theta = theta - r * grad
        theta_arr.append(theta)
        cost = cost_function(X, y, theta)
        cost_history[i] = cost

    return converged, theta, cost_history


def stochastic_gradient_descent(S, r, iter):
    cost_history = [0] * iter
    theta = np.zeros(7)

    for i in range(iter):
        x_sample = S.sample(1, replace=True)
        X = x_sample[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']].values[0]
        y = x_sample.output
        pred = X.dot(theta)
        loss = pred - y
        for j in range(len(theta)):
            temp = theta[j] + r * (y - theta.dot(X)) * X[j]
            theta[j] = temp
        cost = cost_function(S[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']], S.output,
                             theta)
        cost_history[i] = cost

    return theta, cost_history

def print_batch(df,df_test,r,iter):

    converged, theta, cost_history = batch_gradient_descent(df, r, iter)
    print("Final Weight Vector")
    print(theta.values)
    print("Final Learning Rate")
    print(r)
    plt.plot(range(len(cost_history)), cost_history, color='red', label=r)
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    test_cost = cost_function(df_test[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']],
                              df_test.output, theta)
    print("Test Cost Function Value")
    print(test_cost)
    plt.legend()
    plt.show()

def print_stochastic(df,df_test,r,iter):

    theta, cost_history = stochastic_gradient_descent(df, r, iter)
    print("Final Weight Vector")
    print(theta)
    print("Final Learning Rate")
    print(r)
    plt.plot(range(len(cost_history)), cost_history, color='red', label=r)
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    test_cost = cost_function(df_test[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']],
                              df_test.output, theta)
    print("Test Cost Function Value")
    print(test_cost)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # then before
    attributes = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'output']
    df = pd.read_csv('concrete/train.csv', names=attributes)
    df_test = pd.read_csv('concrete/test.csv', names=attributes)


    #To Run one, please comment and uncomment one.  They will run the appropriate graident descent and print figure and
    #values

    #print_batch(df,df_test,0.005,15000)
    #print_stochastic(df,df_test,.0005,30000)

    #analytical weight vector for last problem in assignment

    x_ = df[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']].values
    y_ = df.output

    weight_vector = LA.inv(np.dot(x_.T,x_)).dot(y_.dot(x_))
    print(weight_vector)







import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def cost_function(X,y, theta):
    m = len(X)
    pred = X.dot(theta.T)
    cost = (1/2*m) * np.sum(np.square(pred-y))
    return cost

def batch_gradient_descent(S , theta, r, iter):
    y = S.output
    m = len(y)
    cost_arr = np.zeros(iter)
    theta_arr = np.zeros((iter,7))
    X = S[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']]
    converged = False
    for i in range(iter):
        pred = np.dot(X,theta.T)
        theta = theta - (1/m) * r * (X.T.dot((pred - y)))
        theta_arr[i, :] = theta.T
        cost_arr[i] = cost_function(X,y,theta)
        if i > 0:
            if LA.norm(theta_arr[i] - theta_arr[i-1]) < (1/ np.power(10,6)):
                converged = True
                return converged , theta, cost_arr, theta_arr

    return converged, theta,cost_arr,theta_arr


    """"
    iteration = 0
    w = [[]]
    m = len(S)
    cost = []
    y = S.output
    X = S[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']]
    w[0] = np.zeros(7)
    cost = []
    for iteration in range(max_iter):
        cost.append()
        w.append(np.zeros(7))
        for j in range(7):
            gradient = []
            for i in range(m):
                y_actual = y[i]
                y_pred = w[iteration].T.dot(X.iloc[iteration].values)
                gradient.append((y_pred - y_actual)*X.iloc[i])
            sum_by_m = (np.sum(gradient)) / m
            w[iteration + 1][j] = w[iteration][j] - r*sum_by_m

    return sum_by_m
    """
def stochastic_gradient_descent(X, theta, r, iterations):

    m = len(X)
    cost_arr = np.zeros(iterations)

    y = X.output
    X = X[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']]
    theta_arr = np.zeros((iterations,7))

    converged = False
    for iter in range(iterations):
        cost = 0
        for i in range(m):
            x_i = X.iloc[i].values
            y_i = y[i]
            pred = x_i.dot(theta.T)
            for j in range(7):
                theta[j] = theta[j] - (1 / m) * r *(pred - y_i) * x_i[j]
            cost += cost_function(x_i,y_i,theta)
        theta_arr[iter ] = theta
        cost_arr[iter] = cost
        #if iter > 0:
            #if cost_arr[iter]  < (1 / np.power(10, 6)):
                #converged = True
                #return converged, theta, cost_arr

    return theta,cost_arr


if __name__ == '__main__':

    labels_value = ['unacc', 'acc', 'good', 'vgood']
    # label is where the values of each item is stored.  It is not binary,
    # it is unacceptable, acceptable, good, very good.  This is different

    # then before
    attributes = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'output']
    df = pd.read_csv('concrete/train.csv', names=attributes)
    df_test = pd.read_csv('concrete/test.csv', names=attributes)

    converged = False
    r = 1
    iter = 100

    #GOAL HERE IS TO FIND CORRECT R VALUE WITH THE FINAL DESIRED WEIGHT VECTOR
    r_did_converge = False
    final_r = 0
    final_cost_history = []
    final_weight_vector = []
    while r_did_converge == False:
        theta = np.zeros(7)
        did_converge, theta, cost_history, theta_history = batch_gradient_descent(df, theta, r, iter)

        if did_converge == True:
            r_did_converge = did_converge
            final_r = r
            final_cost_history = cost_history
            final_weight_vector = theta
        else:
            r = r/2
    """
    print("Final Weight Vector")
    print(final_weight_vector.values)
    print("Final Learning Rate")
    print(r)
    plt.plot(range(iter),final_cost_history ,color='red', label= r)
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    
    test_cost = test_cost_function(df_test[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']].values, df_test.output.values,final_weight_vector)
    print("Test Cost Function Value")
    print(test_cost)
    plt.legend()
    plt.show()
    """
    """
    converged_sto = False
    r_sto = 1
    iter_sto = 10000

    # GOAL HERE IS TO FIND CORRECT R VALUE WITH THE FINAL DESIRED WEIGHT VECTOR
    r_did_converge_sto = False
    final_r_sto = 0
    final_cost_history_sto = []
    final_weight_vector_sto = []
    while r_did_converge_sto == False:
        theta_B = np.zeros(7)
        did_converge_sto,theta_SGD, cost_SGD = stochastic_gradient_descent(df, theta_B,r_sto, iter_sto)

        if did_converge_sto == True:
            r_did_converge_sto = did_converge_sto
            final_r_sto = r_sto
            final_cost_history_sto = cost_SGD
            final_weight_vector_sto = theta_SGD
        else:
            r_sto = r_sto / 2
    """
    theta_B = np.zeros(7)
    theta_SGD, cost_SGD = stochastic_gradient_descent(df, theta_B, 0.005, 5000)
    plt.plot(range(5000), cost_SGD, color='red', label=r)
    plt.show()


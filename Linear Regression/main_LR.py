
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def cost_function(X,y, theta):
    m = len(X)
    pred = X.dot(theta.T)
    cost = (1/2*m) * np.sum(np.square(pred - y))
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
def stochastic_gradient_descent(S, r, iterations):

    m = 7
    w = []
    w.append(np.zeros(m))


    y = S.output
    X = S[['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']]
    w_T = np.transpose(w)


    for iter in range(iterations):
        w.append(np.zeros(m))
        w_T = np.transpose(w[iter])

        for j in range(m):
            w[iter + 1][j] = w[iter][j] + (r * y[iter] - w_T.dot(X.iloc[iter])*X.iloc[iter][j])



    return w



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


    print("Final Weight Vector")
    print(final_weight_vector.values)
    print("Final Learning Rate")
    print(r)
    plt.plot(range(iter),final_cost_history ,color='red', label= r)
    plt.ylabel("Cost")
    plt.xlabel("Iteration")



    plt.legend()


    plt.show()




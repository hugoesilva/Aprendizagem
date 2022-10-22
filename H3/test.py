#Consider the problem of learning a regression model from 5 univariate observations
#((0.8), (1), (1.2), (1.4), (1.6)) with targets (24,20,10,13,12)

import numpy as np
from sklearn.linear_model import LinearRegression


X = np.array([[0.8], [1], [1.2], [1.4], [1.6]])
y = np.array([[24], [20], [10], [13], [12]])

#Consider the basis function, 𝜙𝑗(𝑥) = 𝑥^𝑗, for performing a 3-order polynomial regression

def phi(x):
    return np.array([x**i for i in range(4)]).T

#Use the basis function to transform the input data



X = phi(X)

#create matrix from new

X = np.matrix(X)


#get weight matrix

w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

#print("phi(X):\n" + str(X))

#print("W sem Ridge:\n" + str(w))


#Learn the Ridge regression (𝑙2 regularization) on the transformed data space using the closed
#form solution with 𝜆 = 2.

def ridge(x, y, lam):
    return np.linalg.inv(x.T.dot(x) + lam*np.identity(x.shape[1])).dot(x.T).dot(y)

w = ridge(X, y, 2)

#print("W Ridge:\n" + str(w))


#print("\n\n\n" + str(np.linalg.inv(X.T.dot(X) + 2*np.identity(X.shape[1])).dot(X.T).dot(y)))

#compute the training RMSE for the Ridge regression

def rmse(x, y, w):
    a = x.dot(w)
    b = a - y
    sum = 0
    for i in b:
        sum += i**2

    return np.sqrt(sum/len(b))


print("RMSE Ridge: " + str(rmse(X, y, w)))

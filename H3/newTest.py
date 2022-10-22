#Consider the problem of learning a regression model from 5 univariate observations
#((0.8), (1), (1.2), (1.4), (1.6)) with targets (24,20,10,13,12)
#Consider the basis function, 𝜙𝑗(𝑥) = 𝑥^𝑗, for performing a 3-order polynomial regression

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[0.8], [1], [1.2], [1.4], [1.6]])
y = np.array([24, 20, 10, 13, 12])

#Use the basis function to transform the input data

new = np.array([X**i for i in range(4)]).T
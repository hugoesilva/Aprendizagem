import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import multivariate_normal

x1 = np.array([1,2])
x2 = np.array([-1,1])
x3 = np.array([1,0])
X = np.array([x1,x2,x3])

u1 = np.array([0.750968, 1.311493])
u2 = np.array([0.034383, 0.777027])
S1 = np.array([[0.436048,0.077572],[0.077572, 0.778201]])
S2 = np.array([[0.998818,-0.215306],[-0.215306,0.467474]])
pi1 = 0.417188
pi2 = 0.582812

p1 = multivariate_normal.pdf(X, u1, S1)
p2 = multivariate_normal.pdf(X, u2, S2)



p1x = pi1*p1
p2x = pi2*p2




#print(p1)
#print(p2)

#print posteriors

#print(p1x)
#print(p2x)

#silhouette

a = np.sum(p1x)/3
b = np.sum(p2x)/3
print(a)
print(b)

s = (b-a)/max(a,b)
print("silhouette: ", s)

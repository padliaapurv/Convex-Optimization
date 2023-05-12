#imports
import numpy as np
import cvxpy as cp
from math import *

# data for censored fitting problem.
np.random.seed(15)

n = 20;  # dimension of x's
M = 25;  # number of non-censored data points
K = 100; # total number of points
c_true = np.random.randn(n,1)
X = np.random.randn(n,K)
y = np.dot(np.transpose(X),c_true) + 0.1*(np.sqrt(n))*np.random.randn(K,1)

# Reorder measurements, then censor
sort_ind = np.argsort(y.T)
y = np.sort(y.T)
y = y.T
X = X[:, sort_ind.T]
D = (y[M-1]+y[M])/2.0
y = y[list(range(M))]

X = X[:,:,0]

#Variables
c_cp = cp.Variable((n,1))
y_cp = cp.Variable((K,1))

#Constraints
constraints = []

for i in range(M):
    constraints += [y[i] == y_cp[i]]

for i in range(M,K):
    constraints += [y_cp[i] >= D]

#Objective
obj = cp.Minimize(cp.sum_squares(cp.transpose(y_cp)-cp.transpose(c_cp)@X))
obj_ls = cp.Minimize(cp.sum_squares(cp.transpose(y_cp[:M])-cp.transpose(c_cp[:M])@X[:,:M]))

#Problem
prob = cp.Problem(obj, constraints)
prob.solve()

prob_ls = cp.Problem(obj_ls, constraints)
prob_ls.solve()

#Output
print("status: ", prob.status)
print("optimal value: ", prob.value)
print("y\n", y_cp.value)
print("c_hat\n", c_cp.value)

print("ls status: ", prob_ls.status)
print("ls optimal value: ", prob_ls.value)
print("ls y\n", y_cp.value)
print("ls c_hat\n", c_cp.value)

error = 0
den = 0
for i in range(n):
    error += (c_true[i] - (c_cp.value)[i])**2
    den += (c_true[i])**2
print("Error :", sqrt(error/den))

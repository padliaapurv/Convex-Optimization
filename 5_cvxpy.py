import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#Input
np.random.seed(10)
(m, n) = (30, 10)
A = np.random.rand(m, n); A = np.asmatrix(A)
b = np.random.rand(m, 1); b = np.asmatrix(b)
c_nom = np.ones((n, 1)) + np.random.rand(n, 1); c_nom = np.asmatrix(c_nom)

#Variables
x = cp.Variable(n)

#Contraints
constraints = []
for i in range(m):
    constraints += [(A@x)[i] >= b[i]]

#Objective
obj = cp.Minimize(cp.transpose(x)@c_nom)

#Problem
prob = cp.Problem(obj, constraints)
prob.solve()

#Output
print("Nominal C  : ", prob.value)
print(len(constraints))

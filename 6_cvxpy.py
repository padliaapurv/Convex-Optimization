import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#Input
np.random.seed(10)
(m, n) = (30, 10)
A = np.random.rand(m, n); A = np.asmatrix(A)
b = np.random.rand(m, 1); b = np.asmatrix(b)
c_nom = np.ones((n, 1)) + np.random.rand(n, 1); c_nom = np.asmatrix(c_nom)

F = np.zeros((2*n + 2, n))
for i in range(n):
    F[i][i]  = 1
    F[n+i][i] = -1
    F[2*n][i] = 1
    F[2*n + 1][i] = -1

g = np.zeros((2*n + 2, 1))
c_nom_sum = 0
for i in range(n):
    g[i][0] = 1.25 * c_nom[i]
    g[n+i][0] = -0.75 * c_nom[i]
    c_nom_sum += c_nom[i]

g[2*n][0] = 1.1*c_nom_sum
g[2*n + 1][0] = -0.9*c_nom_sum

#Variables
lmda = cp.Variable(2*n+2)
x = cp.transpose([-1.84784822 , 1.98599739, -1.02036781,  0.64483889, -0.78353585, -0.58263546, 0.51293723 , 3.18932783,  2.10572291, -1.38286869])

#Contraints
constraints = []
for i in range(m):
    constraints += [(A@x)[i] >= b[i]]

for i in range(2*n + 2):
    constraints += [lmda[i] >= 0]

for i in range(n):
    constraints += [(cp.transpose(F)@lmda)[i] == x[i]]

#Objective
obj = cp.Minimize(cp.transpose(lmda)@g)

#Problem
prob = cp.Problem(obj, constraints)
prob.solve()

#Output
print("Worst Case : ", prob.value)

import numpy as np
import cvxpy as cp

#Variables
x = cp.Variable()
y = cp.Variable()

#Constraints
constraints = [x >= 0,
               y >= 0,
               2*x + y >= 1,
               x + 3*y >= 1]

#Objective
obj = cp.Minimize(cp.power(x,2) + 9*cp.power(y,2))

#Problem
prob = cp.Problem(obj, constraints)
prob.solve()

#Output
print("status: ", prob.status)
print("optimal value: ", prob.value)
print("Optimal var: ", x.value, y.value)






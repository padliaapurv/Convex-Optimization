import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#Constants
m = 200
r = []
S0 = []
call_1 = []
call_2 = []
put_1 = []
put_2 = []
collar = []

for i in range(m):
    s0 = 0.5 + i*(1.5)/(m-1)
    
    r.append(1.05)
    S0.append(s0)
    if s0 >= 1.1:
        call_1.append(s0-1.1)
    else:
        call_1.append(0)
    if s0 >= 1.2:
        call_2.append(s0-1.2)
    else:
        call_2.append(0)
    if s0 <= 0.8:
        put_1.append(0.8 - s0)
    else:
        put_1.append(0)
    if s0 <= 0.7:
        put_2.append(0.7 - s0)
    else:
        put_2.append(0)
    if s0 <= 1.15 and s0 >= 0.9:
        collar.append(s0 - 1)
    elif s0 >= 1.15:
        collar.append(0.15)
    else:
        collar.append(-0.1)

V = [r,S0,call_1, call_2, put_1, put_2, collar]

pk = [1,1,0.06, 0.03, 0.02, 0.01]
        

#Variables
p = cp.Variable(1)
y = cp.Variable(m)

#Constraints
constraints = []
for i in range(m):
    constraints += [y[i] >= 0]

for i in range(7):
    sum1 = 0
    for j in range(m):
        sum1 += V[i][j]*y[j]
        
    if i == 6:
        constraints += [sum1 == p]
    else:
        constraints += [sum1 == pk[i]]

#Objective
obj_low = cp.Minimize(p)
obj_up = cp.Minimize(-p)

#Problem
prob1 = cp.Problem(obj_low, constraints)
prob1.solve()

prob2 = cp.Problem(obj_up, constraints)
prob2.solve()

#Output
print("Lower bound : ", prob1.value)
print("Upper bound : ", -prob2.value)

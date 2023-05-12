import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from math import *

#Constants
K = 10
N = 10*K
t= []
sinkt = []
coskt = []
y = []

for i in range(1,2*N+1):
    t_temp = -pi + i*pi/N
    t.append(t_temp)
    if t_temp <= pi/2:
        y.append(1)
    else:
        y.append(0)
    
for i in range(K):
    sinkt.append([])
    coskt.append([])
    for j in t:
        sinkt[i].append(sin(j*(i+1)))
        coskt[i].append(cos(j*(i+1)))

coskt.append([])
for j in t:
        coskt[K].append(cos(j*(i+1)))
        
#Variables
a = cp.Variable(K+1)
b = cp.Variable(K)

#Constraints
constraints = []

#Objective
fcos = []
fsin = []

for j in range(2*N):
    fcos_temp = 0
    fsin_temp = 0
    for i in range(K+1):
        fcos_temp += a[i]*coskt[i][j]
    for i in range(K):
        fsin_temp += b[i]*sinkt[i][j]

    fcos.append(fcos_temp)
    fsin.append(fsin_temp)

L1 = 0
L2 = 0

for i in range(2*N):
    L1 += (pi/N)*(cp.abs(fcos[i] + fsin[i] - y[i]))
    L2 += ((fcos[i] + fsin[i] - y[i])**2)

obj_L1 = cp.Minimize(L1)
obj_L2 = cp.Minimize(L2)

#Problem
prob1 = cp.Problem(obj_L1, constraints)
prob1.solve()

aL1 = a.value
bL1 = b.value

for i in range(2*N):
    print(fcos[i].value + fsin[i].value)

prob2 = cp.Problem(obj_L2, constraints)
prob2.solve()

aL2 = a.value
bL2 = b.value

#Output
for i in range(2*N):
    print(fcos[i].value + fsin[i].value)




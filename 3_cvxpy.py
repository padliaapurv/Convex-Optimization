import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

#Input
n = 4
m = 2 
A = np.array([
[ 0.95,  0.16,  0.12,  0.01],
[-0.12,  0.98, -0.11, -0.03],
[-0.16,  0.02,  0.98,  0.03],
[-0.  ,  0.02, -0.04,  1.03],
])
B = np.array([
[ 0.8 , 0. ],
[ 0.1 , 0.2],
[ 0.  , 0.8],
[-0.2 , 0.1],
])
x_init = np.ones(n)
T = 100

T_list = []
for i in range(T):
    T_list.append(i)

#Variables
x = cp.Variable((n,T))
u = cp.Variable((m,T))

#Contraints
constraints = []
for i in range(T-1):
    constraints += [x[:,i+1] == A@(x[:,i]) + B@(u[:,i])]

constraints += [x[:,0] == x_init]
constraints += [x[:,T-1] == 0]

#Objective
objective1 = 0
objective2 = 0
objective3 = 0
objective4 = 0
for i in range(T):
    objective1 += (cp.norm(u[:,i]))**2
    objective2 += cp.norm(u[:,i])
    if cp.norm(u[:,i]) >= objective3:
        objective3 = cp.norm(u[:,i])
    objective4 += cp.norm1(u[:,i])

obj1 = cp.Minimize(objective1)
obj2 = cp.Minimize(objective2)
obj3 = cp.Minimize(objective3)
obj4 = cp.Minimize(objective4)

#Problem
prob1 = cp.Problem(obj1, constraints)
prob2 = cp.Problem(obj2, constraints)
prob3 = cp.Problem(obj3, constraints)
prob4 = cp.Problem(obj4, constraints)

#Problem 1
prob1.solve()

u_norm_list = []
for i in range(T):
    u_norm_list.append(cp.norm((u.value)[:,i]))

plt.figure(1)
plt.plot(T_list, (u.value)[0,:])
plt.plot(T_list, (u.value)[1,:])
plt.plot(T_list, u_norm_list)
plt.title("Part a")
plt.legend("u1","u2","||u||2")
plt.show()

#Problem 2
prob2.solve()

u_norm_list = []
for i in range(T):
    u_norm_list.append(cp.norm((u.value)[:,i]))
    
plt.figure(2)
plt.plot(T_list, (u.value)[0])
plt.plot(T_list, (u.value)[1])
plt.plot(T_list, u_norm_list)
plt.title("Part b")
plt.legend("u1","u2","||u||2")
plt.show()

#Problem 3
prob3.solve()

u_norm_list = []
for i in range(T):
    u_norm_list.append(cp.norm((u.value)[:,i]))

plt.figure(3)
plt.plot(T_list, (u.value)[0])
plt.plot(T_list, (u.value)[1])
plt.plot(T_list, u_norm_list)
plt.title("Part c")
plt.legend("u1","u2","||u||2")
plt.show()

#Problem 4
prob4.solve()

u_norm_list = []
for i in range(T):
    u_norm_list.append(cp.norm((u.value)[:,i]))
    
plt.figure(4)
plt.plot(T_list, (u.value)[0])
plt.plot(T_list, (u.value)[1])
plt.plot(T_list, u_norm_list)
plt.title("Part d")
plt.legend("u1","u2","||u||2")
plt.show()


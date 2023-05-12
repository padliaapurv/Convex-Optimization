import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#Problem
np.random.seed(0)
(m, n) = (300, 100)
A = np.random.rand(m, n); A = np.asmatrix(A)
b = A.dot(np.ones((n, 1)))/2; b = np.asmatrix(b)
c = -np.random.rand(n, 1); c = np.asmatrix(c)

#Variables
x = cp.Variable(n)

#Constraints
constraints = [(A@x)[0] <= b[0],x[0] >= 0,x[0] <= 1]
for i in range(1,n):
    constraints += [(A@x)[i] <= b[i]]
    constraints += [x[i] >= 0]
    constraints += [x[i] <= 1]

#Objective
obj = cp.Minimize((c.T)@x)

#Problem
prob = cp.Problem(obj, constraints)
prob.solve()

#relaxed to binary
max_violation = []
objective = []
t_list = []

for i in range(1,100,1):

    t = i/100
    t_list.append(t)

    x_bin = []

    for j in range(n):
        if x.value[j] >= t:
            x_bin.append(1)
        else:
            x_bin.append(0)

    max_vio = float((A@x_bin)[0,0] - (b)[0,0])
          
    for j in range(1,n):        
        if float((A@x_bin)[0,j] - (b)[j,0]) >= max_vio:
            max_vio = float((A@x_bin)[0,j] - (b)[j,0])

    max_violation.append(max_vio)

    objective.append(((c.T)@x_bin)[0,0])

for i in range(n):
    if max_violation[i] <= 0:
        index = i
        t_bound = i/100
        break

objective_upper = objective[index]

#Output
print("Lower bound : " + str(prob.value))
print("Upper bound : " + str(objective_upper))
print("value of t  : " + str(t_bound))

#Plot
plt.figure(1)
plt.plot(t_list, max_violation)
plt.plot([t_bound,1],[0,0])
plt.title("max_violation")
plt.legend(["max_violation","feasible set"])
plt.show()

plt.figure(2)
plt.plot(t_list, objective)
plt.plot([t_bound,1],[objective_upper,objective_upper])
plt.plot([t_bound,1],[prob.value,prob.value])
plt.title("objective")
plt.legend(["objective", "upper bound","lower bound"])
plt.show()

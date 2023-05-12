import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

#storage_trade_data
np.random.seed(1)

T = 96
t = np.linspace(1, T, num=T).reshape(T,1)
p = np.exp(-np.cos((t-15)*2*np.pi/T)+0.01*np.random.randn(T,1))
u = 2*np.exp(-0.6*np.cos((t+40)*np.pi/T) - 0.7*np.cos(t*4*np.pi/T)+0.01*np.random.randn(T,1))


q_list = []
price_optimal_3 = []
price_optimal_1 = []

for i in range(36):
    q_list.append(i)

for ii in range(2):
    for jj in range(36):

        Q = jj
        CD = 3 - 2*ii

        print(CD)

        #Variables
        q = cp.Variable((T,1))
        c = cp.Variable((T,1))

        #Constraints
        constraints = [q >= 0,
                       q <= Q,
                       c >= -CD,
                       c <= CD]

        constraints += [q[0] == q[T-1] + c[T-1]]
        constraints += [u[T-1] + c[T-1] >= 0]

        for i in range(T-1):
            constraints += [q[i+1] == q[i] + c[i]]
            constraints += [u[i] + c[i] >= 0]

        #Objective
        obj = cp.Minimize(cp.scalar_product(p,(u + c)))

        #Problem
        prob = cp.Problem(obj, constraints)
        prob.solve()

        #Storage
        if ii == 0:
            price_optimal_3.append(prob.value)
        else:
            price_optimal_1.append(prob.value)
        

        
#Plot
plt.figure(1)
plt.plot(q_list, price_optimal_3)
plt.plot(q_list, price_optimal_1)
plt.title("A20.9c")
plt.legend(["C = D = 3","C = D = 1"])
plt.show()

                       

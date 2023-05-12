# Jonathan Richter
# 3/1/2023
# EE364 Convex Optimization
# HW6 Python Problems

import cvxpy as cp
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


#matplotlib gives a very weird warning that is super annoying... from online it looks like a bug
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def nabla2F(x):
    return np.diag(np.divide(1, (np.power(x, 2))).flatten())


def nabla2Finv(x):
    return np.diag(np.power(x, 2).flatten())


def nablaF(x, c):
    return c - (1 / x)


def rprimalL(x, A, b):
    return np.matmul(A, x) - b


def rdualL(x, v, A, c):
    return A.T @ v + nablaF(x, c)


def prob_A10_4(A, b, c, x0, alpha, beta):
    (m, n) = A.shape
    err = 1*10**-6
    residual = 1
    residual_plot = []
    v0 = 0.1*np.ones((m, 1)) #argmin||A^T*nu - NablaF(x0)||_2^2
    v_iter = v0
    x_iter = x0

    while residual - err >= 0:
        # Declaring t as 1 for the beginning of each loop
        t = 1

        # Calculating the residuals (primal and dual)
        rprimal = rprimalL(x_iter, A, b)
        rdual = rdualL(x_iter, v_iter, A, c)

        # Calculating the Hessian and Inverse Hessian
        H = nabla2F(x_iter)
        Hinv = nabla2Finv(x_iter)

        # Page 546 of textbook: Solving Block Elimination

        HinvA = Hinv @ A.T
        Hinvrdual = Hinv @ rdual
        S = -A @ HinvA
        dv = np.linalg.solve(S, A @ Hinvrdual - rprimal)
        dx = np.linalg.solve(H, -1*A.T @ dv - rdual)


        while np.min(x_iter + t*dx) <= 0:
            #print(np.min(x_iter))
            t = beta*t

        #print("Tvalue Before: ", t)
            #Saving data prior to entering while loop


        while np.linalg.norm(np.concatenate((rprimalL(x_iter + t * dx, A, b).flatten(), rdualL(x_iter + t * dx, v_iter + t * dv, A, c).flatten()), axis=None), ord=2) > (1-alpha*t)*np.linalg.norm(np.concatenate((rprimalL(x_iter, A, b).flatten(), rdualL(x_iter, v_iter, A, c).flatten()), axis=None), ord=2):
            t = beta * t
            #print("First Term: ", np.linalg.norm(np.concatenate((rprimalL(x_iter + t * dx, A, b).flatten(), rdualL(x_iter + t * dx, v_iter + t * dv, A, c).flatten()), axis=None), ord=2))
            #print("2nd Term no alph: ", np.linalg.norm(np.concatenate((rprimalL(x_iter, A, b).flatten(), rdualL(x_iter, v_iter, A, c).flatten()), axis=None), ord=2))
            #print("Second Term: ", (1-alpha*t)*np.linalg.norm(np.concatenate((rprimalL(x_iter, A, b).flatten(), rdualL(x_iter, v_iter, A, c).flatten()), axis=None), ord=2))
            #print("Tvalue: ", t)
            #print(" ")




        #print("Tvalue: ", t)
        residual = np.linalg.norm(np.concatenate((rprimalL(x_iter + t * dx, A, b).flatten(), rdualL(x_iter + t * dx, v_iter + t * dv, A, c).flatten()), axis=None), ord=2)
        residual_plot.append(residual)
        x_iter = x_iter + t * dx
        v_iter = v_iter + t * dv
        #print(residual)


    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(0, len(residual_plot)), np.log(residual_plot), 'b*--', linewidth=2)
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel(r"$log[||Residual||_2]$")
    ax.set_title("Problem A10.4:")
    ax.grid(True, which='both')
    fig.savefig("A10_4.png", dpi=200)
    plt.show()

def prob_A6_13():
    np.random.seed(15)
    n = 20;  # dimension of x's
    M = 25;  # number of non-censored data points
    K = 100;  # total number of points
    c_true = np.random.randn(n, 1)
    X = np.random.randn(n, K)
    y = np.dot(np.transpose(X), c_true) + 0.1 * (np.sqrt(n)) * np.random.randn(K, 1)

    # Reorder measurements, then censor
    sort_ind = np.argsort(y.T)
    y = np.sort(y.T)
    y = y.T

    X = X[:, sort_ind.T]
    X = np.squeeze(X)
    D = (y[M - 1] + y[M]) / 2.0
    y = y[list(range(M))]
    y = y.reshape(len(y))

    c = cp.Variable((n, 1))
    z = cp.Variable((K, 1))

    constraints =[]
    for i in range(M):
        constraints.append(z[i] == y[i])

    for j in range(M, K):
        constraints.append(z[j] >= D)

    obj = cp.Minimize(cp.sum_squares(z - cp.transpose((cp.transpose(c) @ X))))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    cvala = c.value

    c_ls = cp.Variable(n)
    obj_ls = cp.Minimize(cp.sum_squares(y.T - cp.transpose(c_ls) @ X[:, 0:25]))
    prob_ls = cp.Problem(obj_ls)
    prob_ls.solve()
    cvalb = c_ls.value

    chate = np.linalg.norm(c_true - cvala.reshape(len(cvala), 1))/np.linalg.norm(c_true)
    c_lse = np.linalg.norm(c_true - cvalb.reshape(len(cvalb), 1))/np.linalg.norm(c_true)

    print("Problem A6.13: ")
    print("c_hat: ")
    print(cvala.T)
    print("c_ls: ")
    print(cvalb)
    print("Error From True Value: ")
    print("c_hat error: ", chate)
    print("c_ls error: ", c_lse)




def main():
    #Problem A10.4
    #Honghao and I were comparing and we used the same starting conditions
    m = 100
    n = 200
    np.random.seed(12)
    # generate a random matrix with dimensions m x n
    A = np.random.rand(m, n)

    # check the rank of the matrix
    rank_A = np.linalg.matrix_rank(A)

    # if rank_A < min(m,n), then the matrix is not full rank
    while rank_A < min(m, n):
        A = np.random.rand(m, n)
        rank_A = np.linalg.matrix_rank(A)


    p = np.random.rand(n, 1)
    b = A @ p
    c = -np.ones((n, 1))

    x0 = 0.5*np.ones((A.shape[1], 1))
    alpha = 0.25 #0.4
    beta = 0.8 #0.9


    prob_A10_4(A, b, c, x0, alpha, beta)
    #prob_A6_13()


if __name__ == "__main__":
    main()

import numpy as np

def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        # Diagonal entry
        sum_sq = np.dot(L[i, :i], L[i, :i])
        L[i, i] = np.sqrt(A[i, i] - sum_sq)

        # Off‐diagonal entries in column i
        for j in range(i+1, n):
            sum_products = np.dot(L[j, :i], L[i, :i])
            L[j, i] = (A[j, i] - sum_products) / L[i, i]

    return L

# Define matrices
A1 = np.array([[ 2, -1,  0,  0],
               [-1,  2, -1,  0],
               [ 0, -1,  2, -1],
               [ 0,  0, -1,  2]], dtype=float)

A2 = np.array([[1,      1/2,   1/3,    1/4],
               [1/2,    1/3,   1/4,    1/5],
               [1/3,    1/4,   1/5,    1/6],
               [1/4,    1/5,   1/6,    1/7]], dtype=float)

L1 = cholesky(A1)
L2 = cholesky(A2)

print("L1 =\n", L1)
print()
print("L2 =\n", L2)


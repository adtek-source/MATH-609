import math
import numpy as np
import matplotlib.pyplot as plt

# Conjugate Gradient (CG) method
def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    n = len(b)
    x = np.zeros(n)  # Initial guess: x(0) = 0
    r = b - A.dot(x)
    v = r.copy()
    c = np.dot(r, r)
    
    for _ in range(max_iter):
        Av = A.dot(v)
        t = c / np.dot(v, Av)
        x = x + t * v
        r = r - t * Av
        d = np.dot(r, r)
        if d < tol:
            break
        v = r + (d / c) * v
        c = d
    
    return x

plt.figure(figsize=(10, 8))

for n in range(1, 15):
    N = 2**n - 1
    h = 2**(-n)

    b = np.array([0 for i in range(N)])
    b[0] = 1
    b[N - 1] = math.e

    A = np.zeros((N,N))
    d = np.full(N, 0.0)
    for i in range(N):
        d[i] = 2 + (h**2)*(4*((i+1)*h)**2 + 2)
    np.fill_diagonal(A, d)
    np.fill_diagonal(A[:-1,1:], -1)
    np.fill_diagonal(A[1:, :-1], -1)

    cg_sol = conjugate_gradient(A, b)
    h_steps = np.array([i*h for i in range(1, N+1)])

    # Exact solution: e^(x^2)
    exact_solution_cg = np.exp(h_steps**2)
    
    # Compute the error as the absolute difference from the exact solution
    error_cg = np.abs(exact_solution_cg - cg_sol)

    # Compute the maximum error
    max_error_cg = np.max(error_cg)
    
    # Compute the error norms
    norm_cg = max_error_cg
    norm_h_cg = norm_cg / (2**(-n))
    norm_h2_cg = norm_cg / (2**(-2 *n))
    norm_h3_cg = norm_cg / (2**(-3 * n))

    print(f'h = {2**(-n)}')
    print(f'Norm: {norm_cg}')
    print(f'Norm divided by h: {norm_h_cg}')
    print(f'Norm divided by h^2: {norm_h2_cg}')
    print(f'Norm divided by h^3: {norm_h3_cg}')
    print()

    plt.plot(h_steps, cg_sol, label=f'CG n={n}')

plt.xlabel('x')
plt.ylabel('$\phi(x)$')
plt.title("Solution of $-\phi'' + (4x^2 + 2)\phi = 0$")
plt.legend()
plt.grid(True)
plt.show()



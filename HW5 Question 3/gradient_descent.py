import math
import numpy as np
import matplotlib.pyplot as plt

# Gradient Descent (GD) method
def gradient_descent(A, b, max_iter=1000, tol=1e-16):
    n = len(b)
    x = np.zeros(n)  # Initial guess: x(0) = 0
    for _ in range(max_iter):
        v = b - A.dot(x)
        t = v.dot(v) / v.dot(A.T.dot(v)) # Gradient of the cost function
        x += t*v # Update x
        if np.dot(b - A.dot(x), b - A.dot(x)) < tol:
            break
        
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

    gd_sol = gradient_descent(A, b)
    h_steps = np.array([i*h for i in range(1, N+1)])

    print(f'h = {2**(-n)}')

    # Exact solution: e^(x^2)
    exact_solution_gd = np.exp(h_steps**2)
    
    # Compute the error as the absolute difference from the exact solution
    error_gd = np.abs(exact_solution_gd - gd_sol)

    # Compute the maximum error
    max_error_gd = np.max(error_gd)
    
    # Compute the error norms
    norm_gd = max_error_gd
    norm_h_gd = norm_gd / (2**(-n))  # Error divided by h
    norm_h2_gd = norm_gd / (2**(-2 * n))  # Error divided by h^2
    norm_h3_gd = norm_gd / (2**(-3 * n))  # Error divided by h^3


    print(f'Norm: {norm_gd}')
    print(f'Norm divided by h: {norm_h_gd}')
    print(f'Norm divided by h^2: {norm_h2_gd}')
    print(f'Norm divided by h^3: {norm_h3_gd}')
    print()

    plt.plot(h_steps, gd_sol, label=f'GD n={n}')

plt.xlabel('x')
plt.ylabel('$\phi(x)$')
plt.title("Solution of $-\phi'' + (4x^2 + 2)\phi = 0$")
plt.legend()
plt.grid(True)
plt.show()



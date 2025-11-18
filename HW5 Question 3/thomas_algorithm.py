import math
import numpy as np
import matplotlib.pyplot as plt

def tridiagonal_solver(n, epsilon=1e-3):
    N = 2**n - 1
    h = 2**(-n)

    b = np.array([0.0 for i in range(N)])
    b[0] = 1
    b[N - 1] = math.e

    a = np.full(N-1, -1.0)
    d = np.full(N, 0.0)
    for i in range(N):
        d[i] = 2 + (h**2)*(4*((i+1)*h)**2 + 2)
    c = np.full(N-1, -1.0)

    for i in range(1, N):
        factor = a[i-1]/ d[i-1]
        d[i] -= factor * c[i-1]
        b[i] -= factor * b[i-1]

    y = np.zeros(N)
    y[-1] = b[-1] / d[-1]
    for i in range(N-2, -1, -1):
        y[i] = (b[i] - c[i] * y[i+1]) / d[i]

    return y, np.array([i*h for i in range(1, N+1)])

plt.figure(figsize=(10, 8))

for n in range(1, 15):
    u, x = tridiagonal_solver(n)
    plt.plot(x, u, label=f'n={n}')

    print(f'h: {2**(-n)}')

    # Exact solution: e^(x^2)
    exact_solution = np.exp(x**2)
    
    # Compute the error as the absolute difference from the exact solution
    error = np.abs(exact_solution - u)

    # Compute the maximum error
    max_error = np.max(error)
    
    # Compute the error norms
    norm = max_error
    norm_h = norm / (2**(-n))  # Error divided by h
    norm_h2 = norm / (2**(-2 * n))  # Error divided by h^2
    norm_h3 = norm / (2**(-3 * n))  # Error divided by h^3

    print(f'Norm: {norm}')
    print(f'Norm divided by h: {norm_h}')
    print(f'Norm divided by h^2: {norm_h2}')
    print(f'Norm divided by h^3: {norm_h3}')
    print()
    
plt.xlabel('x')
plt.ylabel('$\phi(x)$')
plt.title("Solution of $-\phi'' + (4x^2 + 2)\phi = 0$")
plt.legend()
plt.grid(True)
plt.show()

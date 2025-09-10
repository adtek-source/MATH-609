import numpy as np
import matplotlib.pyplot as plt

def tridiagonal_solver(n, epsilon=1e-3):
    N = 2**n - 1
    h = 2**(-n)
    
    f = np.array([2*i*h + 1 for i in range(1, N+1)])
    
    a = np.full(N-1, -epsilon/(h**2)) # subdiagonal
    b = np.full(N, 2*epsilon/(h**2) + 1) # main diagonal
    c = np.full(N-1, -epsilon/(h**2)) # superdiagonal
    
    # forward elimination
    for i in range(1, N):
        factor = a[i-1] / b[i-1]
        b[i] -= factor * c[i-1]
        f[i] -= factor * f[i-1]
    
    # back substitution
    u = np.zeros(N)
    u[-1] = f[-1] / b[-1]
    for i in range(N-2, -1, -1):
        u[i] = (f[i] - c[i] * u[i+1]) / b[i]
    
    return u, np.array([i*h for i in range(1, N+1)])

plt.figure(figsize=(10, 8))

for n in range(1, 9):
    u, x = tridiagonal_solver(n)
    plt.plot(x, u, label=f'n={n}')
    
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution of the system A * u = f for different n')
plt.legend()
plt.grid(True)
plt.show()

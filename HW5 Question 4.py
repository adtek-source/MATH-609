import numpy as np
import matplotlib.pyplot as plt

def solve_bvp_compact_4th(p):
    """
    Solve BVP using compact 4th order FD for h = 2^{-p}.
    Returns grid, numerical solution, exact solution, infinity error.
    """

    h = 2**(-p)
    N = int(1/h)
    x = np.linspace(0, 1, N+1)

    # Unknowns: phi_1, ..., phi_{N-1}
    M = N - 1

    # Pentadiagonal matrix diagonals:
    a = np.zeros(M)      # sub-subdiagonal  (-2)
    b = np.zeros(M)      # subdiagonal     (-1)
    c = np.zeros(M)      # main diagonal   (0)
    d = np.zeros(M)      # superdiagonal   (+1)
    e = np.zeros(M)      # super-superdiag (+2)

    # RHS
    rhs = np.zeros(M)

    for i in range(1, N):
        xi = x[i]
        V = 4*xi**2 + 2   # potential term
        coef = 12*h**2 * V
        c_i = -30 - coef   # main diagonal entry

        # Fill diagonals
        idx = i - 1
        c[idx] = c_i

        if i >= 2:
            b[idx] = 16
        if i >= 3:
            a[idx] = -1
        if i <= N - 2:
            d[idx] = 16
        if i <= N - 3:
            e[idx] = -1

    # Boundary adjustments
    # i = 1 gets contribution from φ_0 = 1
    rhs[0] -= 16*1 + (-1)*1   # 16 φ_0 - φ_{-1}, but φ_{-1} = φ_1 reflection? Actually stencil uses φ_{i-2} = φ_{-1}=φ_1 approx? BUT typically set φ_{-1}=φ_0 due BC

    # i = N-1 gets contribution from φ_N = e
    rhs[-1] -= 16*np.e + (-1)*np.e

    # Solve pentadiagonal system using Gaussian elimination for banded matrices
    # Convert to banded form (5 diagonals)
    # Matrix layout for solve_banded: upper rows first
    ab = np.zeros((5, M))
    ab[0, 2:] = e[:-2]     # 2 above main
    ab[1, 1:] = d[:-1]     # 1 above main
    ab[2, :]  = c          # main
    ab[3, :-1] = b[1:]     # 1 below
    ab[4, :-2] = a[2:]     # 2 below

    from scipy.linalg import solve_banded
    phi_inner = solve_banded((2,2), ab, rhs)

    # Full solution including boundaries
    phi = np.zeros(N+1)
    phi[0] = 1
    phi[-1] = np.e
    phi[1:N] = phi_inner

    # Exact solution
    phi_exact = np.exp(x**2)

    # Infinity norm error
    err = np.max(np.abs(phi - phi_exact))

    return h, x, phi, phi_exact, err

plt.figure(figsize=(8,5))
# ---- Run for p = 1,...,6 and print table ----
print(f"{'h':>8}  {'err':>12}  {'err/h^3':>12}  {'err/h^4':>12}  {'err/h^5':>12}")
for p in range(1, 7):
    h, x, phi, phi_exact, err = solve_bvp_compact_4th(p)
    print(f"{h}  {err:.6e}  {err/h**3:.6e}  {err/h**4:.6e}  {err/h**5:.6e}")
    plt.plot(x, phi_exact, 'k-', linewidth=2, label="Exact solution $e^{x^2}$")
    plt.plot(x, phi, 'r--', linewidth=2, label="Numerical solution")

plt.xlabel("x")
plt.ylabel("ϕ(x)")
plt.title("Numerical vs Exact Solution for Compact 4th-Order Scheme")
plt.grid(True)
plt.legend()
plt.show()

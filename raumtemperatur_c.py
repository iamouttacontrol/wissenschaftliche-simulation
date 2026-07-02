# -*- coding: utf-8 -*-
"""
Stationary temperature distribution in a room via finite differences.

Stage 3: variable epsilon (concrete wall) - final solution.
Projektaufgabe 1 - Wissenschaftliche Simulation (SoSe 2026)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# grid of the room [0, Lx] x [0, Ly]
h = 0.1
Lx, Ly = 7.0, 4.0

Nx = int(round(Lx / h)) + 1
Ny = int(round(Ly / h)) + 1

xs = np.linspace(0.0, Lx, Nx)
ys = np.linspace(0.0, Ly, Ny)

N = Nx * Ny

# fixed (Dirichlet) temperatures in Kelvin
T_heater = 323.15
T_outside = 283.15

# material constant epsilon for air and the concrete wall
eps_air = 2.00e-5
eps_concrete = 8.75e-7


def is_heater(x, y):
    """Return True on the heater block [0, 0.5] x [3, 4]."""
    return (x <= 0.5 + 1e-9) and (y >= 3.0 - 1e-9)


def is_door(x, y):
    """Return True on the open door at the right wall, y in [0.2, 1.2]."""
    return (abs(x - Lx) < 1e-9) and (0.2 - 1e-9 <= y <= 1.2 + 1e-9)


def is_concrete(x, y):
    """Return True inside the concrete wall [3, 3.2] x [1.5, 4]."""
    return (3.0 - 1e-9 <= x <= 3.2 + 1e-9) and (y >= 1.5 - 1e-9)


def eps(x, y):
    """Material constant epsilon at point (x, y)."""
    return eps_concrete if is_concrete(x, y) else eps_air


def idx(i, j):
    """Map grid index (i, j) to a single unknown number k = j*Nx + i."""
    return j * Nx + i


def harmonic(a, b):
    """Harmonic mean of a and b (flux-conserving coupling at material jumps)."""
    return 2.0 * a * b / (a + b)


def solve():
    """
    Assemble and solve the finite-difference system A u = b.

    Heater and door are Dirichlet, the remaining walls are insulated
    (Neumann, u_n = 0). The concrete wall uses a smaller epsilon, coupled
    to its neighbours by the harmonic mean.

    Returns
    -------
    U : ndarray, shape (Ny, Nx)
        Temperature at every grid point.
    """
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)

    for j in range(Ny):
        for i in range(Nx):
            x, y = xs[i], ys[j]
            k = idx(i, j)

            # Dirichlet points: heater and door have a fixed temperature
            if is_heater(x, y):
                A[k, k] = 1.0
                b[k] = T_heater
                continue
            if is_door(x, y):
                A[k, k] = 1.0
                b[k] = T_outside
                continue

            # Neumann boundary: insulated wall via ghost points, the
            # opposite inner neighbour is counted twice
            on_boundary = (i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1)
            if on_boundary:
                A[k, k] = -4.0
                if i == 0:
                    A[k, idx(i + 1, j)] = 2.0
                elif i == Nx - 1:
                    A[k, idx(i - 1, j)] = 2.0
                else:
                    A[k, idx(i + 1, j)] = 1.0
                    A[k, idx(i - 1, j)] = 1.0
                if j == 0:
                    A[k, idx(i, j + 1)] = 2.0
                elif j == Ny - 1:
                    A[k, idx(i, j - 1)] = 2.0
                else:
                    A[k, idx(i, j + 1)] = 1.0
                    A[k, idx(i, j - 1)] = 1.0
                continue

            # interior points: 5-point stencil with harmonic eps coupling
            e0 = eps(x, y)
            e_e = harmonic(e0, eps(xs[i + 1], y))
            e_w = harmonic(e0, eps(xs[i - 1], y))
            e_n = harmonic(e0, eps(x, ys[j + 1]))
            e_s = harmonic(e0, eps(x, ys[j - 1]))
            A[k, idx(i + 1, j)] = e_e
            A[k, idx(i - 1, j)] = e_w
            A[k, idx(i, j + 1)] = e_n
            A[k, idx(i, j - 1)] = e_s
            A[k, k] = -(e_e + e_w + e_n + e_s)

    #np.savetxt('matrix_A.csv', A.toarray(), delimiter=',', fmt='%g')
    u = spla.spsolve(A.tocsr(), b)
    return u.reshape(Ny, Nx)


def plot(U, title):
    """Plot the temperature field U as a filled contour map."""
    plt.figure(figsize=(9, 5))
    cf = plt.contourf(xs, ys, U, levels=40, cmap='turbo')
    plt.colorbar(cf, label='T [K]')
    plt.gca().add_patch(plt.Rectangle((0, 3), 0.5, 1.0, fill=False,
                                      edgecolor='white', lw=2))
    plt.gca().add_patch(plt.Rectangle((3, 1.5), 0.2, 2.5, fill=False,
                                      edgecolor='black', lw=2))
    plt.plot([Lx, Lx], [0.2, 1.2], color='white', lw=3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(title)


print(f'grid: {Nx} x {Ny} = {N} unknowns')

U = solve()
print(f'stage 3: T in [{U.min():.2f}, {U.max():.2f}] K')
plot(U, 'Stage 3: variable epsilon (final)')

plt.show()

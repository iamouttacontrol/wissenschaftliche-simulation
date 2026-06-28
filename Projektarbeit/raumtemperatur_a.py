# -*- coding: utf-8 -*-
"""
Stationary temperature distribution in a room via finite differences.

Stage 1: constant epsilon, Dirichlet boundary only.
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
T_heater = 323.13
T_outside = 283.15


def is_heater(x, y):
    """Return True on the heater block [0, 0.5] x [3, 4]."""
    return (x <= 0.5 + 1e-9) and (y >= 3.0 - 1e-9)


def is_door(x, y):
    """Return True on the open door at the right wall, y in [0.2, 1.2]."""
    return (abs(x - Lx) < 1e-9) and (0.2 - 1e-9 <= y <= 1.2 + 1e-9)


def idx(i, j):
    """Map grid index (i, j) to a single unknown number k = j*Nx + i."""
    return j * Nx + i


def solve():
    """
    Assemble and solve the finite-difference system A u = b.

    Constant epsilon, so the plain Laplace stencil is used in the
    interior. Heater and door are Dirichlet; the remaining boundary is
    fixed to the outside temperature.

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

            # remaining boundary: fixed to the outside temperature
            on_boundary = (i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1)
            if on_boundary:
                A[k, k] = 1.0
                b[k] = T_outside
                continue

            # interior points: 5-point Laplace stencil
            A[k, k] = -4.0
            A[k, idx(i + 1, j)] = 1.0
            A[k, idx(i - 1, j)] = 1.0
            A[k, idx(i, j + 1)] = 1.0
            A[k, idx(i, j - 1)] = 1.0

    u = spla.spsolve(A.tocsr(), b)
    return u.reshape(Ny, Nx)


def plot(U, title):
    """Plot the temperature field U as a filled contour map."""
    plt.figure(figsize=(9, 5))
    cf = plt.contourf(xs, ys, U, levels=40, cmap='inferno')
    plt.colorbar(cf, label='T [K]')
    plt.gca().add_patch(plt.Rectangle((0, 3), 0.5, 1.0, fill=False,
                                      edgecolor='cyan', lw=2))
    plt.plot([Lx, Lx], [0.2, 1.2], color='white', lw=3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(title)


print(f'grid: {Nx} x {Ny} = {N} unknowns')

U = solve()
print(f'stage 1: T in [{U.min():.2f}, {U.max():.2f}] K')
plot(U, 'Stage 1: constant epsilon, Dirichlet only')

plt.show()

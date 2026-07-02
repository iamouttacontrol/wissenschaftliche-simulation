# -*- coding: utf-8 -*-
"""
Visualise the finite-difference system A u = b from raumtemperatur_c.py.

Produces:
  * the sparsity pattern of A (full + zoom)
  * a sign-coloured zoom that reveals the 5-point stencil
  * the right-hand side b mapped back onto the room grid
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---- same setup as raumtemperatur_c.py -----------------------------------
h = 0.1
Lx, Ly = 7.0, 4.0
Nx = int(round(Lx / h)) + 1
Ny = int(round(Ly / h)) + 1
xs = np.linspace(0.0, Lx, Nx)
ys = np.linspace(0.0, Ly, Ny)
N = Nx * Ny

T_heater = 323.15
T_outside = 283.15
eps_air = 2.00e-5
eps_concrete = 8.75e-7


def is_heater(x, y):
    return (x <= 0.5 + 1e-9) and (y >= 3.0 - 1e-9)


def is_door(x, y):
    return (abs(x - Lx) < 1e-9) and (0.2 - 1e-9 <= y <= 1.2 + 1e-9)


def is_concrete(x, y):
    return (3.0 - 1e-9 <= x <= 3.2 + 1e-9) and (y >= 1.5 - 1e-9)


def eps(x, y):
    return eps_concrete if is_concrete(x, y) else eps_air


def idx(i, j):
    return j * Nx + i


def harmonic(a, b):
    return 2.0 * a * b / (a + b)


def assemble():
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)
    # node_type: 0 interior, 1 Dirichlet (heater/door), 2 Neumann wall
    node_type = np.zeros(N, dtype=int)

    for j in range(Ny):
        for i in range(Nx):
            x, y = xs[i], ys[j]
            k = idx(i, j)

            if is_heater(x, y):
                A[k, k] = 1.0
                b[k] = T_heater
                node_type[k] = 1
                continue
            if is_door(x, y):
                A[k, k] = 1.0
                b[k] = T_outside
                node_type[k] = 1
                continue

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
                node_type[k] = 2
                continue

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

    return A.tocsr(), b, node_type


A, b, node_type = assemble()
print(f'A is {A.shape[0]} x {A.shape[1]} with {A.nnz} nonzeros '
      f'({100 * A.nnz / N**2:.3f}% filled)')

# ==========================================================================
# Figure 1: sparsity pattern of A (full + zoom)
# ==========================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

ax1.spy(A, markersize=0.3)
ax1.set_title(f'Sparsity pattern of A  ({N} x {N})')
ax1.set_xlabel('column k')
ax1.set_ylabel('row k')

# zoom into the first few block-rows so the diagonals are visible
z = 3 * Nx
ax2.spy(A[:z, :z], markersize=3)
ax2.set_title(f'Zoom: rows/cols 0..{z}\n(main diagonal + inner/outer bands)')
ax2.set_xlabel('column k')
ax2.set_ylabel('row k')
# mark the +/-Nx off-diagonals
for off in (-Nx, Nx):
    ax2.plot([max(0, off), z + min(0, off)],
             [max(0, -off), z + min(0, -off)],
             color='tab:orange', lw=0.8, ls='--', alpha=0.6)
ax2.text(Nx * 1.1, 2, r'offset $-N_x$', color='tab:orange', fontsize=8)

fig1.tight_layout()
fig1.savefig('viz_A_sparsity.png', dpi=130)

# ==========================================================================
# Figure 2: sign-coloured zoom -> reveals the 5-point stencil
# ==========================================================================
fig2, ax = plt.subplots(figsize=(7, 6))
block = A[:z, :z].toarray()
sign = np.sign(block)  # -1 diagonal, +1 neighbours, 0 empty
cmap = ListedColormap(['tab:blue', 'whitesmoke', 'tab:red'])
norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
im = ax.imshow(sign, cmap=cmap, norm=norm, interpolation='nearest')
cbar = fig2.colorbar(im, ticks=[-1, 0, 1], shrink=0.8)
cbar.ax.set_yticklabels(['negative\n(diagonal)', 'zero',
                         'positive\n(neighbours)'])
ax.set_title(f'Sign of the entries (zoom {z} x {z})\n'
             'diagonal band = self, side bands = the 4 neighbours')
ax.set_xlabel('column k')
ax.set_ylabel('row k')
fig2.tight_layout()
fig2.savefig('viz_A_signs.png', dpi=130)

# ==========================================================================
# Figure 3: right-hand side b mapped back onto the room grid
# ==========================================================================
fig3, (axb1, axb2) = plt.subplots(1, 2, figsize=(13, 5),
                                  gridspec_kw={'width_ratios': [1, 2]})

axb1.plot(b, np.arange(N), lw=0.6)
axb1.set_title('b as a flat vector')
axb1.set_xlabel('b[k]  [K]')
axb1.set_ylabel('unknown k')
axb1.set_ylim(N, 0)
axb1.grid(alpha=0.3)

B = b.reshape(Ny, Nx)
Bm = np.ma.masked_equal(B, 0.0)  # hide the zeros so the two blocks pop
pc = axb2.pcolormesh(xs, ys, Bm, cmap='coolwarm', shading='auto')
fig3.colorbar(pc, ax=axb2, label='b [K]')
axb2.set_title('b mapped onto the room grid\n(non-zero only at heater & door)')
axb2.set_aspect('equal')
axb2.set_xlabel('x [m]')
axb2.set_ylabel('y [m]')
axb2.text(0.25, 3.5, 'heater\n323.15 K', ha='center', va='center', fontsize=8)
axb2.text(6.4, 0.7, 'door\n283.15 K', ha='center', va='center', fontsize=8)

fig3.tight_layout()
fig3.savefig('viz_b.png', dpi=130)

print('saved: viz_A_sparsity.png, viz_A_signs.png, viz_b.png')

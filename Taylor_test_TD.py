#!/usr/bin/env python3
"""
This script was written by Claude. It is a standard script to compute the Taylor test
to check, whether the computed domain derivative does actually converge in second order to 
F(c+eps dc) - F(c)

Taylor test for the temporal domain derivative.

F(c) is the scattered electric field at the five receivers and all time steps for the
star-shaped perfect conductor with spherical-harmonics coefficients c (efie_td), and
J the domain derivative with respect to c (solve_maxwell_domain_derivative_pc_td).
For a direction dc the script prints, in the norm of the data misfit,

    R(+eps) = || F(c + eps dc) - F(c) - eps J dc ||                    -> O(eps^2)
    R(-eps) = || F(c - eps dc) - F(c) + eps J dc ||                    -> O(eps^2)
    e_cd    = || (F(c + eps dc) - F(c - eps dc)) / (2 eps) - J dc || / || J dc ||

J is the discretized domain derivative, not the exact derivative of the discrete forward
map. For small eps the remainder therefore levels off to slope 1 at about
eps * e_cd * ||J dc||, and e_cd decreases under mesh refinement.

Usage:  srun python -m mpi4py Taylor_test_TD.py --refinement 4
"""
import argparse
import types

import numpy as np
from mpi4py import MPI
from scipy.io import savemat

import starshaped as star
from auxiliaryUtils import get_norm_td
from cartandsph import cart2sph, ymn
from electric_problems_td import efie_td, solve_maxwell_domain_derivative_pc_td

parser = argparse.ArgumentParser()
parser.add_argument("--refinement", type=int, default=4, help="sphere refinement (3: 768, 4: 3072 RWG dofs)")
refinement = parser.parse_args().refinement

# ---------------------------------------------------------------- setting of Example 1
c0 = 299792458
T, M, RK = 3e3 * 2e-10 * c0, 400, 1            # final time (600 ns times c0), time steps, 3-stage Radau IIA
points = np.array([[-6, 0, 0, 0, 0], [0, 6, -6, 0, 0], [0, 0, 0, 6, -6]], dtype=float)   # receivers
idx = np.ones(points.shape[1], dtype=bool)
Eivars = types.SimpleNamespace(A=np.array([0., 1., 1.]), direction=np.array([1., 0., 0.]), example=6, a=0)

N = 3                                            # degree of the spherical harmonics
nr = (N + 1) * (N + 2) // 2                      # number of real coefficients
creal = 1.3 * np.array([[np.sqrt(4 * np.pi), 0, 0, 0, 0, 0, 0, 1.0, 0, -0.5]])   # exact object of Example 1
cimag = np.array([[0, 0.3, 0, 0, -0.2, 0]])
eps_list = 0.4 * 0.5 ** np.arange(8)             # step sizes: largest boundary displacement in m


# ---------------------------------------------------------------- direction dc
def radius(cr, ci, x):
    """sum cr Re Y_n^m + ci Im Y_n^m at unit vectors x (3, K), coefficient order of starshaped.Star."""
    _, theta, phi = cart2sph(x[0], x[1], x[2])
    angles = (theta + np.pi / 2, phi + np.pi)
    re = [ymn(*angles, n, m).real for n in range(N + 1) for m in range(n + 1)]
    im = [ymn(*angles, n, m).imag for n in range(1, N + 1) for m in range(1, n + 1)]
    return cr[0] @ np.array(re) + ci[0] @ np.array(im)


def sphere_points(K):
    """K quasi-uniform points on the unit sphere (Fibonacci lattice)."""
    k = np.arange(K) + 0.5
    z = 1 - 2 * k / K
    angle = np.pi * (1 + np.sqrt(5)) * k
    return np.vstack((np.sqrt(1 - z**2) * np.cos(angle), np.sqrt(1 - z**2) * np.sin(angle), z))


# random smooth direction, scaled such that the boundary moves by at most 1 m (eps in metres)
degree = np.array([n for n in range(N + 1) for m in range(n + 1)] + [n for n in range(1, N + 1) for m in range(1, n + 1)])
dc = np.random.default_rng(0).standard_normal((N + 1) ** 2) / (1 + degree * (degree + 1))
dc /= np.max(np.abs(radius(dc[None, :nr], dc[None, nr:], sphere_points(20000))))
dcr, dci = dc[None, :nr], dc[None, nr:]

# ---------------------------------------------------------------- solves
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
ells = np.random.default_rng(0).permutation(int(np.ceil(M / 2)) + 1)   # Laplace-domain frequencies, shuffled
my_ells = np.array_split(ells, size)[rank]                              # for load balance as in IP_efie_TD.py


def log(message):
    if rank == 0:
        print(message, flush=True)


def forward(cr, ci):
    """Scatterer and output of efie_td for the coefficients (cr, ci)."""
    scatterer = star.Star(cr, ci, np.zeros(3), refinement, 1)   # rank argument 1: Star writes no current.msh
    return scatterer, efie_td(scatterer.grid, Eivars, points, points, idx, T, M, RK, MPI, rank, comm, my_ells)


def F(cr, ci):
    """Scattered field at the receivers, shape (receivers, M+1, 3)."""
    _, out = forward(cr, ci)
    return out[2]


# reference solution F(c) and derivative J dc (the derivative reuses the LU factors of efie_td)
scatterer, out = forward(creal, cimag)
div_space, curl_space, F0, Htimesnu, lu_list, S_list = out[:6]
J = solve_maxwell_domain_derivative_pc_td(scatterer, div_space, curl_space, Htimesnu, N, T, M, points, RK,
                                          lu_list, S_list, MPI, rank, comm, size, my_ells)
Jdc = J @ dc                                     # J has shape (receivers, M+1, 3, (N+1)^2)
del out, lu_list, S_list                         # release the LU factors
log(f"refinement {refinement} ({div_space.global_dof_count} dofs): reference solution and derivative done")

# perturbed solutions F(c +- eps dc)
Fp, Fm = [], []
for eps in eps_list:
    Fp.append(F(creal + eps * dcr, cimag + eps * dci))
    Fm.append(F(creal - eps * dcr, cimag - eps * dci))
    log(f"eps = {eps:.4e} done")

# ---------------------------------------------------------------- Taylor remainders
if rank == 0:
    tau = T / M

    def norm(v):
        return get_norm_td(v, tau)

    def order(R):
        return np.log(R[:-1] / R[1:]) / np.log(eps_list[:-1] / eps_list[1:])

    Rp = np.array([norm(fp - F0 - eps * Jdc) for fp, eps in zip(Fp, eps_list)])
    Rm = np.array([norm(fm - F0 + eps * Jdc) for fm, eps in zip(Fm, eps_list)])
    ecd = np.array([norm((fp - fm) / (2 * eps) - Jdc) for fp, fm, eps in zip(Fp, Fm, eps_list)]) / norm(Jdc)

    print(f"\n{'eps [m]':>10} {'R(+eps)':>11} {'order':>6} {'R(-eps)':>11} {'order':>6} {'e_cd':>10}")
    for k, eps in enumerate(eps_list):
        op, om = (f"{order(Rp)[k - 1]:6.2f}", f"{order(Rm)[k - 1]:6.2f}") if k else ("", "")
        print(f"{eps:10.3e} {Rp[k]:11.4e} {op:>6} {Rm[k]:11.4e} {om:>6} {ecd[k]:10.3e}")

    # all results for Matlab, S = load('TaylorTest_ref4.mat'): fields have the dimensions
    # (receivers, time steps, components); Fp(k, :, :, :) and Fm(k, :, :, :) belong to epsilon(k).
    # T and tau are times multiplied by c0 (in m), t_ns are the time steps in ns.
    savemat(f"TaylorTest_ref{refinement}.mat", {
        "refinement": float(refinement), "dofs": float(div_space.global_dof_count), "M": float(M), "RK": float(RK),
        "T": T, "tau": tau, "t_ns": np.arange(M + 1) * tau / c0 * 1e9, "receivers": points,
        "N": float(N), "creal": creal, "cimag": cimag, "dc": dc, "epsilon": eps_list,
        "F0": F0, "J": J, "Jdc": Jdc, "Fp": np.array(Fp), "Fm": np.array(Fm),
        "Rp": Rp, "Rm": Rm, "ecd": ecd, "order_Rp": order(Rp), "order_Rm": order(Rm)}, oned_as="column")

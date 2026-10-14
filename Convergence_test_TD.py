#!/usr/bin/env python3
"""
This script was written by Claude. It is a standard script to compute a reference solution or,
to compute approximations to Maxwell's equations with several different time step sizes tau or 
mesh sizes h.

Convergence of the discretized temporal domain derivative.

Theorem 4.9 bounds the error at the receivers by C (tau^(2m-1) + h^2). The script computes the domain
derivative E' = J dc (Example 1, direction dc of Taylor_test_TD.py) and, for comparison, the scattered
field E^s at the receivers with the 3-stage Radau IIA method (m = 3). All runs are compared with a
reference solution on refinement 5 of starshaped.Star (12288 RWG dofs) with M = 1600 time steps:

    --study reference:  computes the reference solution and saves it to Convergence_reference.mat
    --study time:       refinement 5, M = 32, 40, 50, ..., 800                -> expected order 2m-1 = 5
    --study space:      M = 1600, meshes of StarN with n = 4, 5, 6, 7, 8, 10, 12  -> expected order 2

StarN splits every face of the octahedron into n^2 triangles instead of refining it recursively, so that
n is not restricted to powers of 2 (12 n^2 RWG dofs, 192 ... 1728 here). In the space study, h is the mean
edge length of the mesh; all meshes are at least 2.6 times coarser than the reference.

The error of a run is

    e = max |X - X_ref| / max |X_ref|

over the receivers, the components and the time points shared with the reference, t_j = j T / gcd(M, 1600).
The observed order between consecutive runs is log(e_k / e_{k+1}) / log(tau_k / tau_{k+1}), with h instead
of tau in the space study. Limitations:
  * the reference solution has an error of its own; on the finest meshes of the space study it is about
    (h_ref / h)^2 ~ 15 % of the error, which biases the last orders;
  * the all-at-once convolution quadrature uses the contour radius eps_mach^(1/(2M)), which limits the
    accuracy to about sqrt(eps_mach) ~ 1e-8;

Memory: every run keeps the LU factors of all its Laplace-domain systems, since the derivative reuses them.
On refinement 5 one factor takes 2.4 GB: M = 1600 needs 1760 of them (4.3 TB), M = 800 needs 1040 (2.5 TB).
With --recompute_lu the factors are not kept but computed again when the derivative needs them: twice the
assembly and factorization work, but only a few dense matrices per MPI rank in memory at any time.

Usage:  srun python -m mpi4py Convergence_test_TD.py --study reference    (then time and space)
"""
import argparse
import types

import numpy as np
from mpi4py import MPI
from scipy.io import loadmat, savemat

import starshaped as star
from cartandsph import cart2sph, ymn
from electric_problems_td import efie_td, solve_maxwell_domain_derivative_pc_td

parser = argparse.ArgumentParser()
parser.add_argument("--study", choices=["reference", "time", "space"], required=True)
parser.add_argument("--recompute_lu", action="store_true",
                    help="recompute the LU factors for the derivative instead of keeping them (saves memory)")
args = parser.parse_args()
study = args.study

# ---------------------------------------------------------------- setting of Example 1
c0 = 299792458
T = 3e3 * 2e-10 * c0                             # final time (600 ns times c0)
points = np.array([[-6, 0, 0, 0, 0], [0, 6, -6, 0, 0], [0, 0, 0, 6, -6]], dtype=float)   # receivers
idx = np.ones(points.shape[1], dtype=bool)
Eivars = types.SimpleNamespace(A=np.array([0., 1., 1.]), direction=np.array([1., 0., 0.]), example=6, a=0)

N = 3                                            # degree of the spherical harmonics
nr = (N + 1) * (N + 2) // 2                      # number of real coefficients
creal = 1.3 * np.array([[np.sqrt(4 * np.pi), 0, 0, 0, 0, 0, 0, 1.0, 0, -0.5]])   # exact object of Example 1
cimag = np.array([[0, 0.3, 0, 0, -0.2, 0]])

# runs (RK, refinement, M); RK = 1 is the 3-stage Radau IIA method, see RK_Data.py
REFERENCE = (1, 5, 1600)
if study == "time":
    runs = [(1, 5, M) for M in (32, 40, 50, 64, 80, 100, 128, 160, 200, 320, 400, 800)]
elif study == "space":                         # here the second entry is n of StarN
    runs = [(1, n, 1600) for n in (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26)]
else:
    runs = [REFERENCE]


# ---------------------------------------------------------------- LU factors on demand (--recompute_lu)
# _aux_hpc.RKElCQ_MPI stores what compute_lu_factors returns and passes it to scipy's lu_solve twice: right
# away (forward problem) and in the derivative. Here it returns a stand-in that hands out the factors at the
# first use, drops them, and computes them again from the operator at the second use.
if args.recompute_lu:
    import bempp.api.linalg.direct_solvers as direct_solvers
    from bempp.api import as_matrix
    from scipy.linalg import lu_factor

    def release(V):
        """Drop the dense matrices that bempp caches in the (scaled) boundary operator V."""
        V._cached = None
        if hasattr(V, "_op"):
            V._op._cached = None

    class LUOnDemand:
        """Unpacks into the LU factors of V like the tuple (lu, piv), without keeping them."""
        def __init__(self, V, factors):
            self.V, self.factors = V, factors

        def __iter__(self):
            factors = self.factors if self.factors is not None else lu_factor(as_matrix(self.V.weak_form()))
            self.factors = None
            release(self.V)
            return iter(factors)

    _compute_lu_factors = direct_solvers.compute_lu_factors

    def compute_lu_factors_on_demand(V):
        factors = _compute_lu_factors(V)
        release(V)
        return LUOnDemand(V, factors)

    direct_solvers.compute_lu_factors = compute_lu_factors_on_demand


# ---------------------------------------------------------------- meshes of the space study
class StarN(star.Star):
    """starshaped.Star whose sphere mesh splits every face of the octahedron into n^2 triangles, for any
    n >= 1 (the argument 'refinement' of Star is n); n = 2**r has as many dofs as refinement r (12 n^2)."""

    def triangulatesphere(self, n):
        points, triangles = [], []
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    A, B, C = np.diag([sx, sy, sz]).astype(float)       # corners of one octahedron face
                    if np.dot(np.cross(B - A, C - A), A + B + C) < 0:    # orient the face outwards
                        B, C = C, B
                    index = {}
                    for i in range(n + 1):
                        for j in range(n + 1 - i):
                            index[i, j] = len(points)
                            points.append(A + i / n * (B - A) + j / n * (C - A))
                    for i in range(n):
                        for j in range(n - i):
                            triangles.append((index[i, j], index[i + 1, j], index[i, j + 1]))
                            if i + j < n - 1:
                                triangles.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
        points = np.array(points)
        points /= np.linalg.norm(points, axis=1)[:, None]                 # project onto the sphere
        unique, inverse = np.unique(np.round(points, 12), axis=0, return_inverse=True)  # merge shared points
        return unique, inverse.reshape(-1)[np.array(triangles)] + 1      # 1-based, as in Star


# ---------------------------------------------------------------- direction dc (as in Taylor_test_TD.py)
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


degree = np.array([n for n in range(N + 1) for m in range(n + 1)] + [n for n in range(1, N + 1) for m in range(1, n + 1)])
dc = np.random.default_rng(0).standard_normal((N + 1) ** 2) / (1 + degree * (degree + 1))
dc /= np.max(np.abs(radius(dc[None, :nr], dc[None, nr:], sphere_points(20000))))

# ---------------------------------------------------------------- solves
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def log(message):
    if rank == 0:
        print(message, flush=True)


def mesh_size(grid):
    """Number of RWG dofs (edges), mean and maximal edge length of a bempp grid."""
    e = grid.elements
    edges = np.unique(np.sort(np.hstack([e[[0, 1]], e[[1, 2]], e[[2, 0]]]), axis=0), axis=1)
    lengths = np.linalg.norm(grid.vertices[:, edges[0]] - grid.vertices[:, edges[1]], axis=0)
    return edges.shape[1], lengths.mean(), lengths.max()


def solve(RK, refinement, M):
    """Scattered field E^s and domain derivative E' = J dc at the receivers, shapes (receivers, M+1, 3)."""
    ells = np.random.default_rng(0).permutation(int(np.ceil(M / 2)) + 1)   # Laplace-domain frequencies,
    my_ells = np.array_split(ells, size)[rank]                              # shuffled for load balance
    Scatterer = StarN if study == "space" else star.Star                  # space study: refinement is n
    scatterer = Scatterer(creal, cimag, np.zeros(3), refinement, 1)       # rank argument 1: no current.msh
    div_space, curl_space, Es, Htimesnu, lu_list, S_list = efie_td(
        scatterer.grid, Eivars, points, points, idx, T, M, RK, MPI, rank, comm, my_ells)[:6]
    J = solve_maxwell_domain_derivative_pc_td(scatterer, div_space, curl_space, Htimesnu, N, T, M, points, RK,
                                              lu_list, S_list, MPI, rank, comm, size, my_ells)
    return Es, J @ dc


def t_ns(M):
    """Time steps in ns (T is a time multiplied by c0)."""
    return np.arange(M + 1) * T / M / c0 * 1e9


def cell(arrays):
    """Matlab cell array from a list of arrays (which may have different shapes)."""
    c = np.empty(len(arrays), dtype=object)
    for k, a in enumerate(arrays):
        c[k] = a
    return c


def error(X, M, X_ref):
    """max |X - X_ref| / max |X_ref| at the time points shared with the reference, t_j = j T / gcd(M, M_ref)."""
    M_ref = REFERENCE[2]
    g = np.gcd(M, M_ref)
    return np.max(np.abs(X[:, ::M // g, :] - X_ref[:, ::M_ref // g, :])) / np.max(np.abs(X_ref))


def orders(e, p):
    """Observed orders between consecutive runs with errors e and step sizes p (tau or h)."""
    e, p = np.array(e), np.array(p)
    return np.log(e[:-1] / e[1:]) / np.log(p[:-1] / p[1:])


if study == "reference":
    Es, Ed = solve(*REFERENCE)
    if rank == 0:
        savemat("Convergence_reference.mat", {
            "RK": float(REFERENCE[0]), "refinement": float(REFERENCE[1]), "M": float(REFERENCE[2]),
            "t_ns": t_ns(REFERENCE[2]), "Es": Es, "Ed": Ed, "receivers": points, "dc": dc,
            "creal": creal, "cimag": cimag, "N": float(N), "T": T}, oned_as="column")
    log("reference solution saved to Convergence_reference.mat")
else:
    ref = loadmat("Convergence_reference.mat")
    if tuple(ref[key].item() for key in ("RK", "refinement", "M")) != REFERENCE:
        raise ValueError("Convergence_reference.mat was not computed with the settings in REFERENCE")
    log(f"\n{study} study, 3-stage Radau IIA, errors against refinement {REFERENCE[1]} with M = {REFERENCE[2]}, "
        f"expected order {5 if study == 'time' else 2}")
    log(f"{'run':>16} {'e(E^s)':>10} {'order':>6} {'e(E prime)':>11} {'order':>6}")
    done, Es, Ed, eS, eD, p, sizes = [], [], [], [], [], [], []
    for RK, refinement, M in runs:
        es, ed = solve(RK, refinement, M)
        if rank == 0:
            done.append((RK, refinement, M))
            Es.append(es)
            Ed.append(ed)
            eS.append(error(es, M, ref["Es"]))
            eD.append(error(ed, M, ref["Ed"]))
            if study == "space":                                          # (dofs, h_mean, h_max) of the mesh
                sizes.append(mesh_size(StarN(creal, cimag, np.zeros(3), refinement, 1).grid))
            p.append(T / M if study == "time" else sizes[-1][1])
            oS, oD = (f"{orders(eS, p)[-1]:6.2f}", f"{orders(eD, p)[-1]:6.2f}") if len(done) > 1 else ("", "")
            label = f"M = {M}" if study == "time" else f"n = {refinement}"
            print(f"{label:>16} {eS[-1]:10.3e} {oS:>6} {eD[-1]:11.3e} {oD:>6}", flush=True)
            # Matlab: S = load('Convergence_time.mat'); S.Ed{k} is E' of run k with the dimensions
            # (receivers, time steps, components). Rewritten after every run.
            results = {
                "RK": np.array([run[0] for run in done], dtype=float),
                "refinement": np.array([run[1] for run in done], dtype=float),
                "M": np.array([run[2] for run in done], dtype=float),
                "t_ns": cell([t_ns(run[2]) for run in done]), "Es": cell(Es), "Ed": cell(Ed),
                "e_Es": np.array(eS), "e_Ed": np.array(eD), "order_Es": orders(eS, p), "order_Ed": orders(eD, p),
                "receivers": points, "dc": dc, "creal": creal, "cimag": cimag, "N": float(N), "T": T}
            if study == "space":                                          # n of StarN instead of a refinement
                results["n"] = results.pop("refinement")
                results.update({"dofs": np.array([sz[0] for sz in sizes], dtype=float),
                                "h_mean": np.array([sz[1] for sz in sizes]), "h_max": np.array([sz[2] for sz in sizes])})
            savemat(f"Convergence_{study}.mat", results, oned_as="column")

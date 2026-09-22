"""
This script has been written with Claude assistance
===============================================================================
 Curl_Gamma( a div_Gamma . )  on RWG, tested with SNC
===============================================================================

Rotated test functions satisfy

    curl_Gamma(SNC_i) = Div(RWG_i)          <- dual_kind="snc", coarse grid

so with phi = a div_Gamma(j) the weak form is the div-div pairing

    A[i, j] = int_Gamma  a  (div psi_j^dom)(div psi_i^test)  dS

    A = D_test^T  diag( int_K a )  D_dom

Both D matrices must be built on the SAME element grid: the coarse grid for
SNC, the barycentric grid for RBC (BC/RBC only exist there, and the density's
barycentric_representation() is used to match).
Barycentric grids are not used in the shape optimization script.

div_Gamma of an RWG/BC function is piecewise CONSTANT, and for P1 `a` the
element integral int_K a is exact, so the assembly is exact.

BEMPP CONVENTIONS
--------------------------------------------------------
  * div psi_e = +- l_e / |K|  -- bempp DOES fold the edge length into the RWG
    basis.  integration_elements[e] == 2|K|, hence 2*l_e/integration_elements.
  * local dof ordering is edges (v0,v1), (v2,v0), (v1,v2).
  * dof transformations are handled via localised_space /
    map_to_localised_space / dof_transformation.
"""
import numpy as np
from scipy.sparse import coo_matrix

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def _elementwise_div(space):
    """D[e, i] = (Div phi_i)|_{T_e}.  Exact for RWG-type spaces (RWG, BC).

    An RWG/BC function is affine on each triangle, so its surface divergence is
    a single constant there: +- l_e / |K|, with the sign carried by
    local_multipliers.  integration_elements[e] is 2|K|, hence the factor 2.
    """
    g = space.grid
    gd = g.data()
    loc = space.localised_space
    r, c, v = [], [], []
    for e in space.support_elements:
        vv = [gd.vertices[:, gd.elements[i, e]] for i in range(3)]
        # local dof ordering: (v0,v1), (v2,v0), (v1,v2)
        L = [np.linalg.norm(vv[0] - vv[1]),
             np.linalg.norm(vv[2] - vv[0]),
             np.linalg.norm(vv[1] - vv[2])]
        for i in range(3):
            r.append(e)
            c.append(loc.local2global[e, i])
            v.append(2.0 * L[i] / gd.integration_elements[e]
                     * loc.local_multipliers[e, i])
    D = coo_matrix((v, (r, c)),
                   shape=(g.number_of_elements,
                          space.map_to_localised_space.shape[0])).tocsr()
    # Map localised dofs back to the space's own dofs.
    return D @ space.map_to_localised_space @ space.dof_transformation


def _p1_element_integrals(p1_space):
    """M[e, j] = int_{T_e} p_j.   So  M @ a.coefficients  ==  int_K a.

    Exact: int_K lambda_j = |K|/3 = integration_elements[e]/6.
    Returns the INTEGRAL -- the |K| factor is already included.

    Routed through localised_space / map_to_localised_space exactly as
    _elementwise_div is.  Indexing columns directly as 3*kk+j and sizing with
    dof_transformation.shape[0] only works for a BARYCENTRIC representation,
    where those two layouts coincide; on a coarse P1 space dof_transformation
    is (ndof, ndof) and the localised column indices overflow it.
    """
    g = p1_space.grid
    gd = g.data()
    loc = p1_space.localised_space
    r, c, v = [], [], []
    for e in p1_space.support_elements:
        for j in range(3):
            r.append(e)
            c.append(loc.local2global[e, j])
            v.append(gd.integration_elements[e] / 6.0
                     * loc.local_multipliers[e, j])
    M = coo_matrix((v, (r, c)),
                   shape=(g.number_of_elements,
                          p1_space.map_to_localised_space.shape[0])).tocsr()
    return M @ p1_space.map_to_localised_space @ p1_space.dof_transformation


# --------------------------------------------------------------------------
# Assembling the three pieces on a common element grid
# --------------------------------------------------------------------------
_PIECES_CACHE = {}


def _pieces(domain, a_space, dual_kind):
    """(D_dom, D_test, M_a), all indexed by the same elements.

    dual_kind="snc": everything on the coarse grid, and D_test IS D_dom
                     because curl(SNC_i) = Div(RWG_i).
    dual_kind="rbc": everything on the barycentric grid, D_test from the BC
                     space because curl(RBC_i) = Div(BC_i).
    """
    key = (id(domain), id(a_space), dual_kind)
    hit = _PIECES_CACHE.get(key)
    if hit is not None and hit[0] is domain and hit[1] is a_space:
        return hit[2], hit[3], hit[4]

    if dual_kind == "snc":
        D_dom = _elementwise_div(domain)
        D_test = D_dom                       # same basis in both roles
        M_a = _p1_element_integrals(a_space)
    elif dual_kind == "rbc":
        import bempp.api
        bc = bempp.api.function_space(domain.grid, "BC", 0)
        D_dom = _elementwise_div(domain.barycentric_representation())
        D_test = _elementwise_div(bc)
        M_a = _p1_element_integrals(a_space.barycentric_representation())
    else:
        raise ValueError("dual_kind must be 'snc' or 'rbc'")

    _PIECES_CACHE.clear()
    _PIECES_CACHE[key] = (domain, a_space, D_dom, D_test, M_a)
    return D_dom, D_test, M_a



# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------


def apply_all_columns(domain, a, j, a_space, dual_kind="snc", columns=None):
    """Apply Curl(a[:, cc] div .) to one j, for every column cc.

    Returns an (ndof, M) array of PROJECTIONS -- <op j, .> -- NOT coefficients,
    and NOT something to hand to GridFunction as a single object.  Build one
    per column:

        [bempp.api.GridFunction(rwg, projections=out[:, cc], dual_space=dual)
         for cc in range(out.shape[1])]

    Div j does not depend on cc, so it is formed once and shared.
    """
    D_dom, D_test, M_a = _pieces(domain, a_space, dual_kind)
    c = a.coefficients if hasattr(a, "coefficients") else np.asarray(a)
    c = np.asarray(c)
    if c.ndim != 2:
        raise ValueError("expected a 2-D (ndof_a, M) coefficient array")
    if columns is not None:
        c = c[:, np.asarray(columns)]

    jc = j.coefficients if hasattr(j, "coefficients") else np.asarray(j)
    d = D_dom @ jc                           # face-constant Div j, once
    m = M_a @ c                              # (n_elem, M) of int_K a
    return D_test.T @ (d[:, None] * m)


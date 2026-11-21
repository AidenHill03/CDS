"""Dynamical data and certificates for r = M o F^n o S.

Everything here is stated for the rational map defined by the stored
double-precision poles and residues.  "Certified" means: implied by an explicit
inequality that was evaluated in floating point with a safety factor, not by
sampling.  It is not interval arithmetic.

Theory used (paper = arXiv:2605.09903, Section 2):

* Trap (Lemma 2.10):  r(D(c_j, rho_j)) is inside D(c_j, rho_j / 2), so D(c_j, rho_j)
  lies in the basin of one attracting fixed point zeta_j.
* Capture radius:  if |S(z) - t_j| <= cap_j then r(z) is in D(c_j, rho_j).
* Cell certificate:  on a square cell with center z0 and half-diagonal delta,
  |S - t_j| <= |S(z0) - t_j| + B, where B bounds |S(z) - S(z0)| on the
  circumscribed disk by an order-10 Taylor enclosure (kernel ``taylor_bound``).
  If that is <= cap_j the whole closed cell lies in basin j.
* Exterior:  S - S(inf) is holomorphic outside the window W and vanishes at
  infinity, so its supremum outside W is attained on the boundary of W.
* Critical points:  S has degree m, hence 2m - 2 critical points on the sphere.
  The critical values of r are r(crit S) and the trap centers' targets.  A
  critical point inside a certified cell or outside W has a captured critical
  value automatically.  The others lie in the uncertified layer U; their number
  is 2 * #poles + winding(S', boundary of U) by the argument principle, and
  they are computed one by one (Ehrlich-Aberth) and iterated.
* If every critical value lies in an attracting basin, the Fatou set is the
  union of those basins (paper, Remark 2.12).
"""

from __future__ import annotations

import math

import numpy as np

from . import _approx_kernel as _kernel

EPS = np.finfo(float).eps


class Cancelled(Exception):
    """Raised when a caller's ``should_cancel()`` returns True mid-computation."""


def _poll(should_cancel):
    if should_cancel is not None and should_cancel():
        raise Cancelled()


# --------------------------------------------------------------------------
# Capture radii

def capture_radii(post_bound, targets, base_radius, trap_radii, scale, n):
    """Largest x <= base_radius with scale * postBound^n(x) <= trap_radii (per target)."""
    d = targets.size

    def ok(x):
        b = x.copy()
        for _ in range(n):
            b = post_bound(targets, b, d)
        return scale * b <= trap_radii

    hi = base_radius.astype(float).copy()
    good = ok(hi)
    lo = np.zeros(d)
    for _ in range(60):
        mid = (lo + hi) / 2
        g = ok(mid)
        lo = np.where(~good & g, mid, lo)
        hi = np.where(~good & ~g, mid, hi)
    return np.where(good, base_radius, lo)


# --------------------------------------------------------------------------
# Fixed-point multipliers

def _log10_dF_chain(u0, target, d, n):
    """log10 |(F^n)'(u0)| for u0 near a fixed critical point of F, without underflow.

    Uses the exact local form: with u = t(1+s) and t^(d-1) = -(d-1),
        s' = -sum_{j>=2} (j-1) C(d,j) s^j / (1 - d (1+s)^(d-1)),
        F'(u) = -(d-1)(1+s)^(d-1) * (-d) * ((1+s)^(d-1) - 1) / (1 - d (1+s)^(d-1))^2,
    and for t = 0:  u' = u^d / (1 + f u^(d-1)),  F'(u) = u^(d-1) (d + f u^(d-1)) / (1 + f u^(d-1))^2.
    Once |s| < 1e-60 the leading terms are used in the log domain.
    """
    f = d / (d - 1)
    total = 0.0
    if target == 0:
        u = complex(u0)
        log_u = None
        for _ in range(n):
            if log_u is None and abs(u) < 1e-60:
                log_u = math.log10(abs(u)) if u != 0 else -math.inf
            if log_u is not None:
                total += math.log10(d) + (d - 1) * log_u     # F' ~ d u^(d-1)
                log_u = d * log_u                            # F  ~ u^d
            else:
                den = 1 + f * u ** (d - 1)
                total += math.log10(abs(u ** (d - 1) * (d + f * u ** (d - 1)) / den ** 2))
                u = u ** d / den
        return total
    s = complex(u0) / complex(target) - 1
    log_s = None
    binom = [math.comb(d, j) for j in range(d + 1)]
    binom1 = [math.comb(d - 1, j) for j in range(d)]
    for _ in range(n):
        if log_s is None and abs(s) < 1e-60:
            log_s = math.log10(abs(s)) if s != 0 else -math.inf
        if log_s is not None:
            total += math.log10(d) + log_s                   # F' ~ d s
            log_s = math.log10(d / 2) + 2 * log_s            # s' ~ (d/2) s^2
        else:
            pm1 = sum(binom1[j] * s ** j for j in range(1, d))       # (1+s)^(d-1) - 1
            den = 1 - d * (1 + pm1)
            total += math.log10(abs((d - 1) * (1 + pm1) * d * pm1 / den ** 2))
            s = -sum((j - 1) * binom[j] * s ** j for j in range(2, d + 1)) / den
    return total


def fixed_point_data(runge_kernel, fixed_points, targets, scale, d, n, noise):
    """Multipliers of r at its attracting fixed points, as log10 |r'(zeta_j)|.

    r'(z) = scale * (F^n)'(S(z)) * S'(z).  If |S(zeta_j) - t_j| is below the
    rounding level ``noise[j]`` of the Runge sum, the value returned is an upper
    bound (computed with the noise level in place of the unresolved offset).
    """
    s0, s1, _, _ = runge_kernel.derivs(np.asarray(fixed_points, complex))
    out = []
    for j in range(targets.size):
        offset = abs(s0[j] - targets[j])
        bound_only = bool(offset <= noise[j])
        u0 = s0[j] if not bound_only else targets[j] + noise[j]
        log_chain = _log10_dF_chain(u0, targets[j], d, n)
        ds = abs(s1[j])
        log_mult = (math.log10(scale * ds) if ds > 0 else -math.inf) + log_chain
        out.append(dict(runge_offset=float(offset), runge_derivative=float(ds),
                        log10_abs_multiplier=float(log_mult), multiplier_is_upper_bound=bound_only))
    return out


# --------------------------------------------------------------------------
# Cell certificate

def certification_grid(x_edges, y_edges, poles, margin, subdivision):
    """Fine grid aligned with the pixel grid, covering image, all poles and a margin."""
    dx = (x_edges[-1] - x_edges[0]) / (x_edges.size - 1)
    dy = (y_edges[0] - y_edges[-1]) / (y_edges.size - 1)
    left = margin
    right = max(0, math.ceil((poles.real.max() - x_edges[-1]) / dx)) + margin
    up = max(0, math.ceil((poles.imag.max() - y_edges[0]) / dy)) + margin
    down = max(0, math.ceil((y_edges[-1] - poles.imag.min()) / dy)) + margin
    k = subdivision
    ncols = (x_edges.size - 1 + left + right) * k
    nrows = (y_edges.size - 1 + up + down) * k
    xs = x_edges[0] - left * dx + np.arange(ncols + 1) * (dx / k)
    ys = y_edges[0] + up * dy - np.arange(nrows + 1) * (dy / k)
    return xs, ys, (up * k, left * k)


def _certify(runge_kernel, z, delta, targets, caps, chunk=200000):
    labels = np.zeros(z.size, np.int32)
    bound = np.full(z.size, np.inf)
    for a in range(0, z.size, chunk):
        s, var = runge_kernel.taylor_bound(z[a:a + chunk], delta)
        err = np.abs(s[:, None] - targets[None, :])
        j = np.argmin(err, axis=1)
        e = err[np.arange(j.size), j]
        with np.errstate(invalid="ignore"):
            b = (e + var) * (1 + 1e-9) + 64 * EPS * (1 + np.abs(s))
        b[~np.isfinite(b)] = np.inf
        labels[a:a + chunk] = np.where(b <= caps[j], j + 1, 0)
        bound[a:a + chunk] = b
    return labels, bound


def certify_cells(runge_kernel, xs, ys, targets, caps, refinements=0, should_cancel=None):
    """Label each closed grid cell with the basin it provably lies in (0 = not certified).

    xs, ys are the edges of the coarsest grid.  Each refinement halves the cells
    that are still undecided and tests the four children; a certified cell
    passes its label to its children.  Returns labels and bounds on the finest
    (uniform) grid together with that grid's edges.
    """
    def centers(xe, ye):
        return ((xe[:-1] + xe[1:]) / 2)[None, :] + 1j * ((ye[:-1] + ye[1:]) / 2)[:, None]

    def half_diag(xe, ye):
        return 0.5 * math.hypot(xe[1] - xe[0], ye[0] - ye[1]) * (1 + 1e-9)

    z = centers(xs, ys)
    lab, bnd = _certify(runge_kernel, z.ravel(), half_diag(xs, ys), targets, caps)
    lab, bnd = lab.reshape(z.shape), bnd.reshape(z.shape)
    for _ in range(refinements):
        _poll(should_cancel)
        xs = np.linspace(xs[0], xs[-1], 2 * (xs.size - 1) + 1)
        ys = np.linspace(ys[0], ys[-1], 2 * (ys.size - 1) + 1)
        lab = np.repeat(np.repeat(lab, 2, axis=0), 2, axis=1)
        bnd = np.repeat(np.repeat(bnd, 2, axis=0), 2, axis=1)
        todo = lab == 0
        z = centers(xs, ys)[todo]
        l2, b2 = _certify(runge_kernel, z, half_diag(xs, ys), targets, caps)
        lab[todo], bnd[todo] = l2, b2
    return lab, bnd, xs, ys


def exterior_certificate(cell_labels, cell_bound, outside, cap_outside, shift):
    """True if every point outside the window is certified to lie in the exterior basin."""
    ring = np.ones(cell_labels.shape, bool)
    ring[1:-1, 1:-1] = False
    if not np.all(cell_labels[ring] == outside):
        return False, math.inf
    sup = float(cell_bound[ring].max()) + 2 * abs(shift)
    return bool(sup <= cap_outside), sup


# --------------------------------------------------------------------------
# Argument principle on the boundary of the uncertified layer

def _winding(runge_kernel, a, b, threshold, max_depth=14):
    """Sum of the change of arg S' along oriented segments a[i] -> b[i]; ok=False if unresolved."""
    ga = runge_kernel.derivs(a)[1]
    gb = runge_kernel.derivs(b)[1]
    total = 0.0
    ok = True
    for depth in range(max_depth + 1):
        if a.size == 0:
            break
        with np.errstate(invalid="ignore", divide="ignore"):
            dphi = np.angle(gb / ga)
        bad = ~(np.abs(dphi) <= threshold)
        total += float(dphi[~bad].sum())
        if not bad.any():
            a = a[:0]
            break
        if depth == max_depth:
            ok = False
            break
        a, b, ga, gb = a[bad], b[bad], ga[bad], gb[bad]
        mid = (a + b) / 2
        gm = runge_kernel.derivs(mid)[1]
        a, b = np.concatenate([a, mid]), np.concatenate([mid, b])
        ga, gb = np.concatenate([ga, gm]), np.concatenate([gm, gb])
    return total, ok


def anchor_disk_certificate(runge_kernel, center, radius, target, cap, points=128):
    """Certify |S - target| <= cap on the disk |z - center| <= radius and count zeros of S' in it.

    S has no pole inside the anchor ring, so the maximum of |S - target| over the
    disk is attained on the circle, where it is bounded by Taylor enclosures of
    short arcs.  The zero count is the winding number of S' along the inscribed
    polygon.  (An m-point ring alone gives S = t / (1 - w^m), whose derivative
    has an (m-1)-fold zero at the center; the other poles split it.)
    """
    ang = 2 * np.pi * np.arange(points) / points
    z = center + radius * np.exp(1j * ang)
    s, var = runge_kernel.taylor_bound(z, radius * np.pi / points * 1.01)
    sup = float(np.max(np.abs(s - target) + var))
    if not (sup <= cap):
        return None
    total_a, ok_a = _winding(runge_kernel, z, np.roll(z, -1), math.pi / 4)
    total_b, ok_b = _winding(runge_kernel, z, np.roll(z, -1), math.pi / 8)
    wa, wb = total_a / (2 * math.pi), total_b / (2 * math.pi)
    if not (ok_a and ok_b) or round(wa) != round(wb) or abs(wa - round(wa)) > 1e-6:
        return None
    return dict(center=complex(center), radius=float(radius), inner_radius=float(radius * math.cos(math.pi / points)),
                sup_bound=sup, critical_points_inside=int(round(wa)))


def _region_boundary(xs, ys, region):
    """Counter-clockwise boundary edges of a union of grid cells (region must not touch the grid edge)."""
    rows, cols = np.nonzero(region)
    x0, x1, y0, y1 = xs[cols], xs[cols + 1], ys[rows], ys[rows + 1]   # y0 = top, y1 = bottom
    seg_a, seg_b = [], []
    for sel, pa, pb in (
        (~region[rows + 1, cols], x0 + 1j * y1, x1 + 1j * y1),   # bottom: left -> right
        (~region[rows, cols + 1], x1 + 1j * y1, x1 + 1j * y0),   # right: bottom -> top
        (~region[rows - 1, cols], x1 + 1j * y0, x0 + 1j * y0),   # top: right -> left
        (~region[rows, cols - 1], x0 + 1j * y0, x0 + 1j * y1),   # left: top -> bottom
    ):
        seg_a.append(pa[sel])
        seg_b.append(pb[sel])
    return np.concatenate(seg_a), np.concatenate(seg_b)


def count_zeros_in_region(runge_kernel, xs, ys, region, poles, pieces=8):
    """Zeros of the numerator of S' in a union of grid cells, by the argument principle.

    count = winding(S', boundary) + 2 * (poles inside).  Returns (count or None, details).
    """
    if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
        return None, dict(reason="region touches the window boundary")
    a, b = _region_boundary(xs, ys, region)
    t = np.arange(pieces) / pieces
    A = (a[:, None] + (b - a)[:, None] * t[None, :]).ravel()
    B = (a[:, None] + (b - a)[:, None] * (t[None, :] + 1 / pieces)).ravel()
    col = np.floor((poles.real - xs[0]) / (xs[1] - xs[0])).astype(int)
    row = np.floor((ys[0] - poles.imag) / (ys[0] - ys[1])).astype(int)
    inside = (col >= 0) & (col < region.shape[1]) & (row >= 0) & (row < region.shape[0])
    n_poles = int(np.count_nonzero(region[row[inside], col[inside]]))
    results = []
    for threshold in (math.pi / 4, math.pi / 8):
        total, ok = _winding(runge_kernel, A, B, threshold)
        w = total / (2 * math.pi)
        if not ok or abs(w - round(w)) > 1e-6:
            return None, dict(reason="argument of S' not resolved on the boundary", winding=w)
        results.append(int(round(w)))
    if results[0] != results[1]:
        return None, dict(reason="winding number depends on the sampling", winding=results)
    return 2 * n_poles + results[0], dict(winding=results[0], boundary_edges=int(a.size), poles=n_poles)


def count_critical_points_in_layer(runge_kernel, xs, ys, certified, poles, pieces=8):
    """Number of zeros of S' in U = union of uncertified cells, by the argument principle.

    The boundary of U consists of edges of certified cells, where S' is
    well conditioned.  Every pole must lie in U.  Returns (count or None, details).
    """
    U = ~certified
    col = np.floor((poles.real - xs[0]) / (xs[1] - xs[0])).astype(int)
    row = np.floor((ys[0] - poles.imag) / (ys[0] - ys[1])).astype(int)
    inside = (col >= 0) & (col < certified.shape[1]) & (row >= 0) & (row < certified.shape[0])
    if not inside.all() or certified[row, col].any():
        return None, dict(reason="a pole lies outside the uncertified layer")
    return count_zeros_in_region(runge_kernel, xs, ys, U, poles, pieces)


def localize_missing(runge_kernel, xs, ys, layer, poles, known, cells_per_pixel, disks, tile_pixels=4,
                     should_cancel=None):
    """Find zeros of S' in the layer that the global solver missed.

    The layer is cut into tiles whose cut lines run along half-pixel lines (the
    poles sit on pixel lines).  In each tile the argument principle gives the
    exact number of zeros; tiles with fewer known zeros are searched with
    Newton from every layer cell center.  Tiles that meet an anchor disk are
    skipped (their zeros are counted separately).
    """
    k = cells_per_pixel
    T = tile_pixels * k
    off = k // 2
    xc = (xs[:-1] + xs[1:]) / 2
    yc = (ys[:-1] + ys[1:]) / 2
    dx, dy = xs[1] - xs[0], ys[0] - ys[1]
    kcol = np.floor((known.real - xs[0]) / dx).astype(int)
    krow = np.floor((ys[0] - known.imag) / dy).astype(int)
    new = []
    checked = deficit = 0
    nrow, ncol = layer.shape
    for r0 in range(off - T, nrow, T):
        for c0 in range(off - T, ncol, T):
            _poll(should_cancel)
            ra, rb, ca, cb = max(r0, 0), min(r0 + T, nrow), max(c0, 0), min(c0 + T, ncol)
            if ra >= rb or ca >= cb or not layer[ra:rb, ca:cb].any():
                continue
            if disks and any(abs((xc[ca] + xc[cb - 1]) / 2 + 1j * (yc[ra] + yc[rb - 1]) / 2 - dk["center"])
                             < dk["radius"] + T * max(dx, dy) for dk in disks):
                continue
            region = np.zeros(layer.shape, bool)
            region[ra:rb, ca:cb] = layer[ra:rb, ca:cb]
            # Pad so that the region never touches the grid edge in the count.
            count, _ = count_zeros_in_region(runge_kernel, xs, ys, region, poles)
            checked += 1
            have = int(np.count_nonzero((krow >= ra) & (krow < rb) & (kcol >= ca) & (kcol < cb)
                                        & region[np.clip(krow, 0, nrow - 1), np.clip(kcol, 0, ncol - 1)]))
            if count is None or count <= have:
                continue
            deficit += count - have
            rr, cc = np.nonzero(region)
            z = xc[cc] + 1j * yc[rr]
            for _ in range(60):
                _, s1, s2, _ = runge_kernel.derivs(z)
                with np.errstate(all="ignore"):
                    dz = s1 / s2
                dz[~np.isfinite(dz)] = 0
                big = np.abs(dz) > T * dx
                dz[big] *= T * dx / np.abs(dz[big])
                z = z - dz
            _, s1, s2, _ = runge_kernel.derivs(z)
            with np.errstate(all="ignore"):
                ok = np.abs(s1 / s2) <= 1e-12 * (1 + np.abs(z))
            z = z[ok]
            zc = np.floor((z.real - xs[0]) / dx).astype(int)
            zr = np.floor((ys[0] - z.imag) / dy).astype(int)
            inside = (zr >= 0) & (zr < nrow) & (zc >= 0) & (zc < ncol)
            z, zr, zc = z[inside], zr[inside], zc[inside]
            z = z[region[zr, zc]]
            pool = np.concatenate([known, np.array(new, complex)])
            for q in z:
                if np.min(np.abs(pool - q)) > 1e-7:
                    new.append(q)
                    pool = np.append(pool, q)
    return np.array(new, complex), dict(tiles_checked=checked, tile_deficit=deficit)


# --------------------------------------------------------------------------
# Critical points of S: zeros of S' = -sum res/(z-p)^2

def aberth_critical_points(runge_kernel, poles, in_layer, in_window, expected=None, sampler=None,
                           max_iter=300, tol=1e-13, rounds=6, should_cancel=None):
    """Ehrlich-Aberth iteration for the 2m-2 zeros of the numerator of S'.

    With N = S' * prod (z-p_k)^2 the Newton correction is
        N/N' = S' / (S'' + 2 S' sum_k 1/(z-p_k)).
    in_layer(z), in_window(z): boolean masks.  Points that leave the window or
    stall inside certified cells are not needed (their critical values are
    covered by the cell and exterior certificates) and are frozen.  If
    ``expected`` (the argument-principle count for the layer) is not reached,
    frozen points are restarted from ``sampler(n)`` (random points of the layer)
    while the zeros already found keep repelling them.
    Returns (z, status) with status 1 = converged, 0 = frozen outside the layer,
    -1 = not converged inside the layer.
    """
    from scipy.spatial import cKDTree
    m = poles.size
    pts = np.column_stack([poles.real, poles.imag])
    nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
    step = 0.35 * nn * np.exp(0.9j)
    z = np.concatenate([poles + step, poles - step])[: 2 * m - 2]
    status = np.full(z.size, -1, np.int8)
    cap = 20 * float(np.median(nn))

    def iterate(active, iters):
        stall = np.zeros(z.size, np.int32)
        for it in range(iters):
            _poll(should_cancel)
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            _, s1, s2, c1 = runge_kernel.derivs(z[idx])
            # Repulsion from zeros already found and from the other moving
            # points only: a frozen point is not a zero and must not repel.
            use = active | (status == 1)
            rep = _kernel.repulsion(z[idx], z[use])     # O(active x used), not O(used^2)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                w = s1 / (s2 + 2 * c1 * s1)
                dz = w / (1 - w * rep)
            bad = ~np.isfinite(dz)
            dz[bad] = 1e-3 * cap * np.exp(2j * np.pi * (0.618 * (idx[bad] + it)))
            big = np.abs(dz) > cap                 # an iterate may not jump across the picture
            dz[big] *= cap / np.abs(dz[big])
            z[idx] -= dz
            done = np.abs(dz) <= tol * (1 + np.abs(z[idx]))
            status[idx[done]] = 1
            active[idx[done]] = False
            rest = idx[~done]
            out = ~in_window(z[rest])
            status[rest[out]] = 0
            active[rest[out]] = False
            rest = rest[~out]
            safe = ~in_layer(z[rest])
            stall[rest[safe]] += 1
            stall[rest[~safe]] = 0
            give_up = stall[rest] > 40
            status[rest[give_up]] = 0
            active[rest[give_up]] = False
        status[active] = np.where(in_layer(z[active]), -1, 0)

    iterate(np.ones(z.size, bool), max_iter)
    for _ in range(rounds):
        if expected is None or sampler is None:
            break
        found = int(np.count_nonzero((status == 1) & in_layer(z)))
        free = np.flatnonzero(status != 1)
        if found >= expected or free.size == 0:
            break
        take = free[: min(free.size, 4 * (expected - found) + 8)]
        z[take] = sampler(take.size)
        status[take] = -1
        active = np.zeros(z.size, bool)
        active[take] = True
        iterate(active, 120)
    # Finish: a handful of iterates still moving inside the layer are usually
    # slow (a zero of S' close to another zero or to a pole); give them time.
    if expected is not None and np.count_nonzero((status == 1) & in_layer(z)) < expected:
        slow = (status == -1) & in_layer(z)
        if 0 < slow.sum() <= 64:
            iterate(slow.copy(), 4000)
    # Polish converged points with plain Newton on S'.
    conv = np.flatnonzero(status == 1)
    for _ in range(3):
        _, s1, s2, _ = runge_kernel.derivs(z[conv])
        with np.errstate(invalid="ignore", divide="ignore"):
            z[conv] -= np.where(s2 != 0, s1 / s2, 0)
    return z, status


def infinity_data(poles, residues, shift, min_panel_nodes, anchor_degree):
    """Behaviour of S at infinity:  S(1/w) = shift + a_1 w + a_2 w^2 + ...,  a_k = sum res p^(k-1).

    A Gauss panel with q nodes integrates polynomials of degree <= 2q-1 exactly
    and every raster contour is closed, so the edge poles give a_k = 0 for
    k <= 2*q_min.  An m-point anchor ring gives a_k = 0 for k <= m-1.  Hence, in
    exact arithmetic, S - S(Inf) vanishes at infinity to order at least
        K = min(2*q_min, m-1) + 1,
    i.e. infinity is a critical point of S of multiplicity at least K-1, with
    critical value S(Inf) = shift.  The stored coefficients realize the vanishing
    a_k as numbers of size ~1e-17, which splits that point into simple critical
    points away from the picture; the exterior and cell certificates cover them.
    """
    order = min(2 * int(min_panel_nodes), int(anchor_degree) - 1) + 1
    center = (poles.real.min() + poles.real.max()) / 2 + 0.5j * (poles.imag.min() + poles.imag.max())
    q = poles - center            # moments about the center: same vanishing order, better scaled
    rel = []
    power = np.ones_like(q)
    for _ in range(order + 2):
        ref = np.sum(np.abs(residues) * np.abs(power))
        rel.append(abs(np.sum(residues * power)) / ref if ref > 0 else 0.0)
        power = power * q
    return dict(value=complex(shift), sum_residues=complex(np.sum(residues)),
                relative_moments=np.array(rel), min_vanishing_order=order,
                min_critical_multiplicity=order - 1)


def critical_analysis(r, runge_kernel, poles, residues, xs, ys, cell_labels, centers, radii,
                      exterior_ok, max_iterations, anchors=(), cells_per_pixel=8,
                      should_cancel=None, progress=None):
    """Locate the critical points of S in the uncertified layer and test their orbits under r.

    anchors: iterable of (center, ring_radius, target, capture_radius) for the
    auxiliary rings.  The disk of 0.7 ring radii around each is certified
    separately (maximum principle) and its critical points are counted, not
    computed: they sit on a nearly (m-1)-fold zero of S' and are ill conditioned.
    """
    m = poles.size
    certified = cell_labels > 0
    dxs, dys = xs[1] - xs[0], ys[0] - ys[1]
    xc, yc = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    half_diag = 0.5 * math.hypot(dxs, dys)

    disks = []
    count_mask = certified.copy()          # cells treated as certified for the count
    for center, ring_radius, target, cap in anchors:
        disk = anchor_disk_certificate(runge_kernel, center, 0.7 * ring_radius, target, cap)
        if disk is None:
            continue
        disks.append(disk)
        # Put every cell that meets the disk into the layer, so that the disk lies inside it.
        near = np.abs((xc[None, :] + 1j * yc[:, None]) - center) <= disk["radius"] + half_diag
        count_mask &= ~near
    disk_c = np.array([k["center"] for k in disks], complex)
    disk_r = np.array([k["inner_radius"] for k in disks], float)

    def in_disks(z):
        if disk_c.size == 0:
            return np.zeros(z.shape, bool)
        return np.any(np.abs(z[..., None] - disk_c) <= disk_r, axis=-1)

    def cell_index(z):
        col = np.floor((z.real - xs[0]) / dxs).astype(np.int64)
        row = np.floor((ys[0] - z.imag) / dys).astype(np.int64)
        ok = (col >= 0) & (col < certified.shape[1]) & (row >= 0) & (row < certified.shape[0]) & np.isfinite(z)
        return row, col, ok

    def in_window(z):
        return cell_index(z)[2]

    def in_layer(z):
        row, col, ok = cell_index(z)
        out = np.zeros(z.shape, bool)
        out[ok] = ~count_mask[row[ok], col[ok]]
        return out & ~in_disks(z)

    expected, count_info = count_critical_points_in_layer(runge_kernel, xs, ys, count_mask, poles)
    in_anchor_disks = int(sum(k["critical_points_inside"] for k in disks))
    if expected is not None:
        expected -= in_anchor_disks
    rng = np.random.default_rng(12345)
    lrow, lcol = np.nonzero(~count_mask)

    def sampler(count):
        pick = rng.integers(0, lrow.size, count)
        return (xs[lcol[pick]] + dxs * rng.random(count)) + 1j * (ys[lrow[pick]] - dys * rng.random(count))

    if progress is not None and expected is not None:
        progress(f"Critical points: {expected} expected in the undecided layer (argument principle); solving")
    z, status = aberth_critical_points(runge_kernel, poles, in_layer, in_window, expected, sampler,
                                       should_cancel=should_cancel)
    conv = status == 1
    layer = conv & in_layer(z)
    pts = z[layer]
    localization = None
    if expected is not None and pts.size < expected:
        if progress is not None:
            progress(f"Critical points: {expected - pts.size} missing; searching tile by tile")
        extra, localization = localize_missing(runge_kernel, xs, ys, ~count_mask, poles, pts,
                                               cells_per_pixel, disks, should_cancel=should_cancel)
        if extra.size:
            extra = extra[in_layer(extra)]
            pts = np.concatenate([pts, extra])
        localization["found"] = int(extra.size)
    # Distinctness and residual of the computed zeros.
    if pts.size > 1:
        from scipy.spatial import cKDTree
        xy = np.column_stack([pts.real, pts.imag])
        tree = cKDTree(xy)
        keep = np.ones(pts.size, bool)
        merged = set()
        for i, j in tree.query_pairs(1e-7):
            keep[max(i, j)] = False
            merged.add(min(i, j))
        merged = [i for i in merged if keep[i]]
        old_index = np.flatnonzero(keep)
        pts = pts[keep]
        xy = xy[keep]
        min_sep = float(cKDTree(xy).query(xy, k=2)[0][:, 1].min()) if pts.size > 1 else math.inf
    else:
        merged, old_index = [], np.arange(pts.size)
        min_sep = math.inf
    # Several iterates converging to one point = a multiple zero of S' (a
    # degenerate critical point), or zeros closer than the solver can separate.
    # Either way the multiplicity is the winding number of S' on a small circle.
    multiplicity = np.ones(pts.size, np.int64)
    if len(merged):
        pos = {int(o): k for k, o in enumerate(old_index)}
        ring = np.exp(2j * np.pi * np.arange(64) / 64)
        for i in merged:
            k = pos[int(i)]
            rad = 1e-5 * min(min_sep, float(np.abs(poles - pts[k]).min()))
            a_ = pts[k] + rad * ring
            w, ok = _winding(runge_kernel, a_, np.roll(a_, -1), math.pi / 8)
            multiplicity[k] = int(round(w / (2 * math.pi))) if ok else 1
    _, s1, s2, _ = runge_kernel.derivs(pts)
    with np.errstate(invalid="ignore", divide="ignore"):
        newton = np.abs(s1 / s2)
    values = r(pts) if pts.size else pts
    labels, entry = r.classify(values, centers, radii, max_iterations)
    stuck = int(np.count_nonzero((status == -1) & in_layer(z)))
    count_ok = expected is not None and expected == int(multiplicity.sum())
    all_captured = bool(np.all(labels > 0))
    verified = bool(count_ok and all_captured and exterior_ok)
    if verified:
        verdict = "verified numerically"
    elif not all_captured:
        verdict = "NOT verified: a critical value was not captured"
    elif not exterior_ok:
        verdict = "not verified: exterior certificate failed"
    else:
        verdict = "not verified: critical-point count not confirmed"
    hist = {int(k): int(v) for k, v in zip(*np.unique(entry[np.isfinite(entry)], return_counts=True))}
    return dict(
        performed=True, method="aberth", total_on_sphere=2 * m - 2,
        layer_count_argument_principle=expected, layer_count_found=int(multiplicity.sum()),
        distinct_found=int(pts.size), multiple_points=pts[multiplicity > 1],
        multiple_point_orders=multiplicity[multiplicity > 1],
        count_confirmed=bool(count_ok), count_details=count_info,
        anchor_disks=disks, in_anchor_disks=in_anchor_disks, localization=localization,
        outside_layer=None if expected is None else 2 * m - 2 - expected - in_anchor_disks,
        critical_points=pts, critical_values=values, basin_labels=labels, entry_iterations=entry,
        entry_histogram=hist, all_captured=all_captured, unconverged_in_layer=stuck,
        max_newton_step=float(newton.max()) if pts.size else 0.0, min_separation=min_sep,
        fatou_set_verified=verified, verdict=verdict,
    )

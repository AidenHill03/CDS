// =============================================================================
// modules/approximation/kernel.cpp -- the approximation module's C++ kernel
// (Python extension `_approx_kernel`).
//
// The map itself, r = M o F^(n) o S (Fisher-Hill-Lazebnik-Thompson, Section
// 2; F is the model polynomial P in reciprocal coordinates), is evaluated by
// the ENGINE's general composed-map type, cdx::ComposedMap -- this file only
// builds the three stages and iterates them. What lives here is what is
// specific to this research question:
//
//   * S = shift + sum_k res_k/(z - p_k) evaluated for blocks of points with
//     the pole loop outside (vectorizes without -ffast-math), plus S', S'',
//     sum 1/(z-p) (critical points) and an order-10 Taylor enclosure of S on
//     a disk (cell certificates);
//   * orbit classification into the certified trapping disks;
//   * F^(n) alone, and the Ehrlich-Aberth repulsion sums.
//
// Semantics follow the original image_to_basins.m: NaN in -> NaN out,
// complex infinity in -> S = shift, an exact hit on a pole -> Inf.
// Never compile with -ffast-math (IEEE Inf/NaN rules are relied on).
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cdx/composed.hpp"
#include "cdx/rational.hpp"

namespace py = pybind11;
using cplx = std::complex<double>;

namespace {

constexpr int kBlock = 64;
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

int g_threads = 0;  // 0 = hardware concurrency

int thread_count(std::size_t work) {
    int t = g_threads > 0 ? g_threads : static_cast<int>(std::thread::hardware_concurrency());
    if (t < 1) t = 1;
    // Small jobs are not worth a thread.
    std::size_t maxUseful = std::max<std::size_t>(1, work / 256);
    return static_cast<int>(std::min<std::size_t>(t, maxUseful));
}

template <class Fn>
void parallel_for(std::size_t count, Fn fn) {
    int t = thread_count(count);
    if (t <= 1) {
        fn(std::size_t(0), count);
        return;
    }
    // Dynamic scheduling in chunks: orbit lengths vary a lot near the Julia set.
    std::size_t chunk = std::max<std::size_t>(kBlock, count / (static_cast<std::size_t>(t) * 16));
    std::atomic<std::size_t> next(0);
    std::vector<std::thread> pool;
    pool.reserve(t);
    for (int i = 0; i < t; ++i) {
        pool.emplace_back([&]() {
            for (;;) {
                std::size_t a = next.fetch_add(chunk);
                if (a >= count) break;
                fn(a, std::min(count, a + chunk));
            }
        });
    }
    for (auto& th : pool) th.join();
}

inline bool c_isnan(double re, double im) { return std::isnan(re) || std::isnan(im); }
inline bool c_isinf(double re, double im) { return std::isinf(re) || std::isinf(im); }
inline bool c_isfinite(double re, double im) { return std::isfinite(re) && std::isfinite(im); }

// u^k for a small non-negative integer k, by repeated multiplication.
inline cplx ipow(cplx u, int k) {
    cplx out(1.0, 0.0);
    cplx base = u;
    while (k > 0) {
        if (k & 1) out *= base;
        k >>= 1;
        if (k) base *= base;
    }
    return out;
}

struct Runge {
    std::vector<double> pr, pi, rr, ri;  // poles and residues, split re/im
    std::vector<double> ra;              // |residues|
    double sr = 0.0, si = 0.0;          // constant (RungeShift)

    Runge(py::array_t<cplx, py::array::c_style | py::array::forcecast> poles,
          py::array_t<cplx, py::array::c_style | py::array::forcecast> residues,
          cplx shift) {
        if (poles.size() != residues.size())
            throw std::invalid_argument("poles and residues must have the same length");
        auto p = poles.unchecked();
        auto r = residues.unchecked();
        std::size_t m = static_cast<std::size_t>(poles.size());
        pr.resize(m); pi.resize(m); rr.resize(m); ri.resize(m);
        const cplx* pp = poles.data();
        const cplx* rp = residues.data();
        ra.resize(m);
        for (std::size_t k = 0; k < m; ++k) {
            pr[k] = pp[k].real(); pi[k] = pp[k].imag();
            rr[k] = rp[k].real(); ri[k] = rp[k].imag();
            ra[k] = std::abs(rp[k]);
        }
        (void)p; (void)r;
        sr = shift.real(); si = shift.imag();
    }

    std::size_t degree() const { return pr.size(); }

    // Slow, careful path for one finite point, used only when the fast sum
    // was not finite.  Mirrors MATLAB: v = shift + (1./(z-poles)) * residues,
    // and a non-finite v with an exact pole hit becomes Inf.
    cplx careful(cplx z) const {
        cplx acc(0.0, 0.0);
        bool exact = false;
        for (std::size_t k = 0; k < pr.size(); ++k) {
            cplx dz = z - cplx(pr[k], pi[k]);
            if (dz.real() == 0.0 && dz.imag() == 0.0) exact = true;
            acc += (cplx(1.0, 0.0) / dz) * cplx(rr[k], ri[k]);
        }
        cplx v = cplx(sr, si) + acc;
        if (!c_isfinite(v.real(), v.imag()) && exact) return cplx(kInf, 0.0);
        return v;
    }

    // S, S', S'' and C(z) = sum_k 1/(z - p_k) at count points.  out[0..7] are
    // re/im arrays for the four quantities.  Used by the critical-point solver
    // and the argument-principle count.  Non-finite input gives NaN.
    void derivs_block(const double* zr, const double* zi, std::size_t count, double* const* out) const {
        const std::size_t m = pr.size();
        double xr[kBlock], xi[kBlock], a[8][kBlock];
        for (std::size_t s = 0; s < count; s += kBlock) {
            const int nb = static_cast<int>(std::min(count, s + kBlock) - s);
            for (int i = 0; i < nb; ++i) {
                xr[i] = zr[s + i]; xi[i] = zi[s + i];
                for (int c = 0; c < 8; ++c) a[c][i] = 0.0;
            }
            for (std::size_t k = 0; k < m; ++k) {
                const double px = pr[k], py = pi[k], qx = rr[k], qy = ri[k];
                for (int i = 0; i < nb; ++i) {
                    const double dx = xr[i] - px, dy = xi[i] - py;
                    const double den = dx * dx + dy * dy;
                    const double ix = dx / den, iy = -dy / den;            // 1/(z-p)
                    const double jx = ix * ix - iy * iy, jy = 2.0 * ix * iy;  // 1/(z-p)^2
                    const double kx = jx * ix - jy * iy, ky = jx * iy + jy * ix;  // 1/(z-p)^3
                    a[0][i] += ix * qx - iy * qy;  a[1][i] += ix * qy + iy * qx;
                    a[2][i] -= jx * qx - jy * qy;  a[3][i] -= jx * qy + jy * qx;
                    a[4][i] += 2.0 * (kx * qx - ky * qy);  a[5][i] += 2.0 * (kx * qy + ky * qx);
                    a[6][i] += ix;  a[7][i] += iy;
                }
            }
            for (int i = 0; i < nb; ++i) {
                const bool ok = c_isfinite(xr[i], xi[i]);
                a[0][i] += sr; a[1][i] += si;
                for (int c = 0; c < 8; ++c) out[c][s + i] = ok ? a[c][i] : kNaN;
            }
        }
    }

    // S(z0) and L(z0, delta) = sum_k |res_k| / (|z0-p_k| (|z0-p_k| - delta)).
    // For |z - z0| <= delta:  |S(z) - S(z0)| <= delta * L.  L = Inf if a pole
    // lies within delta of z0.  mind receives the distance to the nearest pole.
    void lipschitz_block(const double* zr, const double* zi, std::size_t count, double delta,
                         double* outr, double* outi, double* outL, double* mind) const {
        const std::size_t m = pr.size();
        double xr[kBlock], xi[kBlock], ar[kBlock], ai[kBlock], al[kBlock], md[kBlock];
        for (std::size_t s = 0; s < count; s += kBlock) {
            const int nb = static_cast<int>(std::min(count, s + kBlock) - s);
            for (int i = 0; i < nb; ++i) {
                xr[i] = zr[s + i]; xi[i] = zi[s + i]; ar[i] = ai[i] = al[i] = 0.0; md[i] = kInf;
            }
            for (std::size_t k = 0; k < m; ++k) {
                const double px = pr[k], py = pi[k], qx = rr[k], qy = ri[k], qa = ra[k];
                for (int i = 0; i < nb; ++i) {
                    const double dx = xr[i] - px, dy = xi[i] - py;
                    const double den = dx * dx + dy * dy;
                    const double ix = dx / den, iy = -dy / den;
                    ar[i] += ix * qx - iy * qy;
                    ai[i] += ix * qy + iy * qx;
                    const double dist = std::sqrt(den);
                    al[i] += qa / (dist * (dist - delta));
                    md[i] = dist < md[i] ? dist : md[i];
                }
            }
            for (int i = 0; i < nb; ++i) {
                outr[s + i] = sr + ar[i]; outi[s + i] = si + ai[i];
                outL[s + i] = (md[i] > delta) ? al[i] : kInf;
                mind[s + i] = md[i];
            }
        }
    }

    // Taylor enclosure of S on the disk |z - z0| <= delta, order kTaylor:
    //   S(z) = sum_{k<=K} c_k (z-z0)^k + R,  c_k = (-1)^k sum_j res_j/(z0-p_j)^(k+1),
    //   |R| <= sum_j |res_j| (delta/dist_j)^(K+1) / (dist_j - delta).
    // Returns S(z0) and B = sum_{k=1..K} |c_k| delta^k + remainder bound, so that
    // |S(z) - S(z0)| <= B on the disk.  The c_k keep the cancellation between
    // poles, which the plain Lipschitz sum above throws away.
    static constexpr int kTaylor = 10;
    void taylor_block(const double* zr, const double* zi, std::size_t count, double delta,
                      double* outr, double* outi, double* outB) const {
        const std::size_t m = pr.size();
        double xr[kBlock], xi[kBlock], cr[kTaylor + 1][kBlock], ci[kTaylor + 1][kBlock], rem[kBlock], md[kBlock];
        for (std::size_t s = 0; s < count; s += kBlock) {
            const int nb = static_cast<int>(std::min(count, s + kBlock) - s);
            for (int i = 0; i < nb; ++i) {
                xr[i] = zr[s + i]; xi[i] = zi[s + i]; rem[i] = 0.0; md[i] = kInf;
                for (int k = 0; k <= kTaylor; ++k) cr[k][i] = ci[k][i] = 0.0;
            }
            for (std::size_t j = 0; j < m; ++j) {
                const double px = pr[j], py = pi[j], qx = rr[j], qy = ri[j], qa = ra[j];
                for (int i = 0; i < nb; ++i) {
                    const double dx = xr[i] - px, dy = xi[i] - py;
                    const double den = dx * dx + dy * dy;
                    const double ix = dx / den, iy = -dy / den;
                    double wx = ix * qx - iy * qy, wy = ix * qy + iy * qx;   // res/(z0-p)
                    cr[0][i] += wx; ci[0][i] += wy;
                    for (int k = 1; k <= kTaylor; ++k) {
                        const double tx = -(wx * ix - wy * iy), ty = -(wx * iy + wy * ix);
                        wx = tx; wy = ty;
                        cr[k][i] += wx; ci[k][i] += wy;
                    }
                    const double dist = std::sqrt(den);
                    const double ratio = delta / dist;
                    double rp = ratio;
                    for (int k = 0; k < kTaylor; ++k) rp *= ratio;
                    rem[i] += qa * rp / (dist - delta);
                    md[i] = dist < md[i] ? dist : md[i];
                }
            }
            for (int i = 0; i < nb; ++i) {
                double b = rem[i], dp = 1.0;
                for (int k = 1; k <= kTaylor; ++k) { dp *= delta; b += std::hypot(cr[k][i], ci[k][i]) * dp; }
                outr[s + i] = sr + cr[0][i]; outi[s + i] = si + ci[0][i];
                outB[s + i] = (md[i] > delta) ? b : kInf;
            }
        }
    }

    // Evaluate S at count points (re/im arrays), writing re/im outputs.
    void eval_block(const double* zr, const double* zi, std::size_t count,
                    double* outr, double* outi) const {
        const std::size_t m = pr.size();
        double xr[kBlock], xi[kBlock], ar[kBlock], ai[kBlock];
        std::size_t idx[kBlock];
        for (std::size_t s = 0; s < count; s += kBlock) {
            std::size_t e = std::min(count, s + kBlock);
            int nb = 0;
            for (std::size_t j = s; j < e; ++j) {
                double a = zr[j], b = zi[j];
                if (c_isnan(a, b)) { outr[j] = kNaN; outi[j] = kNaN; }
                else if (c_isinf(a, b)) { outr[j] = sr; outi[j] = si; }
                else { xr[nb] = a; xi[nb] = b; idx[nb] = j; ar[nb] = 0.0; ai[nb] = 0.0; ++nb; }
            }
            if (nb == 0) continue;
            const double* PR = pr.data(); const double* PI = pi.data();
            const double* RR = rr.data(); const double* RI = ri.data();
            for (std::size_t k = 0; k < m; ++k) {
                const double px = PR[k], py = PI[k], qx = RR[k], qy = RI[k];
                for (int i = 0; i < nb; ++i) {
                    const double dx = xr[i] - px, dy = xi[i] - py;
                    const double den = dx * dx + dy * dy;
                    const double ix = dx / den, iy = -dy / den;
                    ar[i] += ix * qx - iy * qy;
                    ai[i] += ix * qy + iy * qx;
                }
            }
            for (int i = 0; i < nb; ++i) {
                double vr = sr + ar[i], vi = si + ai[i];
                if (!c_isfinite(vr, vi)) {
                    cplx v = careful(cplx(xr[i], xi[i]));
                    vr = v.real(); vi = v.imag();
                }
                outr[idx[i]] = vr; outi[idx[i]] = vi;
            }
        }
    }
};

// One application of F, as written in iteratePostMap (two branches for
// |u| <= 1 and |u| > 1 to avoid overflow).  Non-finite u is returned as is.
inline cplx post_once(cplx u, int d, double factor) {
    if (!c_isfinite(u.real(), u.imag())) return u;
    if (std::abs(u) <= 1.0) {
        cplx den = 1.0 + factor * ipow(u, d - 1);
        if (den.real() == 0.0 && den.imag() == 0.0) return cplx(kInf, 0.0);
        return ipow(u, d) / den;
    }
    cplx den = ipow(cplx(1.0, 0.0) / u, d - 1) + factor;
    if (den.real() == 0.0 && den.imag() == 0.0) return cplx(kInf, 0.0);
    return u / den;
}

inline cplx post_n(cplx u, int d, int n) {
    const double factor = static_cast<double>(d) / static_cast<double>(d - 1);
    for (int k = 0; k < n; ++k) u = post_once(u, d, factor);
    return u;
}

// r = M o F^(n) o S as an engine ComposedMap. Built from the raw data so the
// Python side and this kernel describe the same three stages.
cdx::RationalMap partial_fractions(cplx shift, const cplx* poles, const cplx* residues, std::size_t m) {
    cdx::RationalMap S;
    S.add_poly(shift, 0);   // the constant; S(inf) = shift
    S.pole_terms().reserve(m);
    for (std::size_t k = 0; k < m; ++k) {
        // Pushed directly: add_pole's collision check is O(m) per pole, and the
        // construction already guarantees distinct poles (PoleCollision check).
        cdx::PoleTerm t;
        t.location = poles[k];
        t.strength = residues[k];
        S.pole_terms().push_back(t);
    }
    return S;
}

std::string model_map_source(int d) {
    // F(z) = z^d / (1 + d/(d-1) z^(d-1)): the model polynomial in reciprocal coordinates.
    return "z^" + std::to_string(d) + "/(1 + " + std::to_string(d) + "/" + std::to_string(d - 1) +
           "*z^" + std::to_string(d - 1) + ")";
}

struct BasinMap {
    Runge S;
    int d, n;
    double or_, oi_, scale;
    cdx::ComposedMap composed;

    BasinMap(py::array_t<cplx, py::array::c_style | py::array::forcecast> poles,
             py::array_t<cplx, py::array::c_style | py::array::forcecast> residues,
             cplx shift, int d_, int n_, cplx origin, double scale_)
        : S(poles, residues, shift), d(d_), n(n_), or_(origin.real()), oi_(origin.imag()), scale(scale_) {
        if (d < 2) throw std::invalid_argument("d must be at least 2");
        if (n < 1) throw std::invalid_argument("n must be at least 1");
        std::vector<cdx::ComposedStage> stages;
        stages.push_back({partial_fractions(shift, poles.data(), residues.data(),
                                            static_cast<std::size_t>(poles.size())), cplx(0.0, 0.0), 1});
        stages.push_back({cdx::RationalMap::from_expression(model_map_source(d), "", {}), cplx(0.0, 0.0), n});
        cdx::RationalMap M;
        M.add_poly(origin, 0);
        M.add_poly(cplx(scale_, 0.0), 1);
        stages.push_back({M, cplx(0.0, 0.0), 1});
        composed = cdx::ComposedMap(std::move(stages));
    }

    // r on count points, in place on re/im buffers.
    void apply(double* zr, double* zi, std::size_t count) const {
        for (std::size_t j = 0; j < count; ++j) composed.step(zr[j], zi[j]);
    }
};

// ---- numpy helpers -------------------------------------------------------

using CArr = py::array_t<cplx, py::array::c_style | py::array::forcecast>;
using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<py::ssize_t> shape_of(const py::array& a) {
    return std::vector<py::ssize_t>(a.shape(), a.shape() + a.ndim());
}

void split(const CArr& z, std::vector<double>& re, std::vector<double>& im) {
    std::size_t N = static_cast<std::size_t>(z.size());
    re.resize(N); im.resize(N);
    const cplx* p = z.data();
    for (std::size_t j = 0; j < N; ++j) { re[j] = p[j].real(); im[j] = p[j].imag(); }
}

CArr join(const std::vector<double>& re, const std::vector<double>& im, const std::vector<py::ssize_t>& shape) {
    CArr out(shape);
    cplx* q = out.mutable_data();
    for (std::size_t j = 0; j < re.size(); ++j) q[j] = cplx(re[j], im[j]);
    return out;
}

CArr runge_call(const Runge& S, CArr z) {
    auto shape = shape_of(z);
    std::vector<double> re, im;
    split(z, re, im);
    {
        py::gil_scoped_release release;
        parallel_for(re.size(), [&](std::size_t a, std::size_t b) {
            S.eval_block(re.data() + a, im.data() + a, b - a, re.data() + a, im.data() + a);
        });
    }
    return join(re, im, shape);
}

py::tuple runge_derivs(const Runge& S, CArr z) {
    auto shape = shape_of(z);
    std::vector<double> re, im;
    split(z, re, im);
    const std::size_t N = re.size();
    std::vector<std::vector<double>> buf(8, std::vector<double>(N));
    {
        py::gil_scoped_release release;
        parallel_for(N, [&](std::size_t a, std::size_t b) {
            double* out[8];
            for (int c = 0; c < 8; ++c) out[c] = buf[c].data() + a;
            S.derivs_block(re.data() + a, im.data() + a, b - a, out);
        });
    }
    return py::make_tuple(join(buf[0], buf[1], shape), join(buf[2], buf[3], shape),
                          join(buf[4], buf[5], shape), join(buf[6], buf[7], shape));
}

py::tuple runge_lipschitz(const Runge& S, CArr z, double delta) {
    if (!(delta >= 0.0)) throw std::invalid_argument("delta must be non-negative");
    auto shape = shape_of(z);
    std::vector<double> re, im;
    split(z, re, im);
    const std::size_t N = re.size();
    std::vector<double> vr(N), vi(N);
    py::array_t<double> L(shape), D(shape);
    double* Lp = L.mutable_data();
    double* Dp = D.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(N, [&](std::size_t a, std::size_t b) {
            S.lipschitz_block(re.data() + a, im.data() + a, b - a, delta, vr.data() + a, vi.data() + a, Lp + a, Dp + a);
        });
    }
    return py::make_tuple(join(vr, vi, shape), L, D);
}

py::tuple runge_taylor(const Runge& S, CArr z, double delta) {
    if (!(delta >= 0.0)) throw std::invalid_argument("delta must be non-negative");
    auto shape = shape_of(z);
    std::vector<double> re, im;
    split(z, re, im);
    const std::size_t N = re.size();
    std::vector<double> vr(N), vi(N);
    py::array_t<double> B(shape);
    double* Bp = B.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(N, [&](std::size_t a, std::size_t b) {
            S.taylor_block(re.data() + a, im.data() + a, b - a, delta, vr.data() + a, vi.data() + a, Bp + a);
        });
    }
    return py::make_tuple(join(vr, vi, shape), B);
}

// R_i = sum_j 1/(w_i - z_j) over sources z_j != w_i: Aberth repulsion of
// the active points w from all current iterates z.  O(len(w) * len(z)).
CArr repulsion(CArr w, CArr z) {
    auto shape = shape_of(w);
    std::vector<double> wr, wi, zr, zi;
    split(w, wr, wi);
    split(z, zr, zi);
    const std::size_t N = wr.size(), M = zr.size();
    std::vector<double> ar(N), ai(N);
    {
        py::gil_scoped_release release;
        parallel_for(N, [&](std::size_t a, std::size_t b) {
            for (std::size_t i = a; i < b; ++i) {
                const double x = wr[i], y = wi[i];
                double sx = 0.0, sy = 0.0;
                for (std::size_t j = 0; j < M; ++j) {
                    const double dx = x - zr[j], dy = y - zi[j];
                    const double den = dx * dx + dy * dy;
                    const double inv = (den > 0.0) ? 1.0 / den : 0.0;
                    sx += dx * inv; sy -= dy * inv;
                }
                ar[i] = sx; ai[i] = sy;
            }
        });
    }
    return join(ar, ai, shape);
}

// A_i = sum_{j != i} 1/(z_i - z_j): the Aberth repulsion term.
CArr pair_sum(CArr z) {
    auto shape = shape_of(z);
    std::vector<double> re, im;
    split(z, re, im);
    const std::size_t N = re.size();
    std::vector<double> ar(N), ai(N);
    {
        py::gil_scoped_release release;
        parallel_for(N, [&](std::size_t a, std::size_t b) {
            for (std::size_t i = a; i < b; ++i) {
                const double x = re[i], y = im[i];
                double sx = 0.0, sy = 0.0;
                for (std::size_t j = 0; j < N; ++j) {
                    const double dx = x - re[j], dy = y - im[j];
                    const double den = dx * dx + dy * dy;
                    const double w = (den > 0.0) ? 1.0 / den : 0.0;  // skips j == i and exact coincidences
                    sx += dx * w; sy -= dy * w;
                }
                ar[i] = sx; ai[i] = sy;
            }
        });
    }
    return join(ar, ai, shape);
}

CArr map_call(const BasinMap& r, CArr z) {
    auto shape = shape_of(z);
    std::vector<double> re, im;
    split(z, re, im);
    {
        py::gil_scoped_release release;
        parallel_for(re.size(), [&](std::size_t a, std::size_t b) {
            r.apply(re.data() + a, im.data() + a, b - a);
        });
    }
    return join(re, im, shape);
}

CArr post_call(int d, int n, CArr u) {
    if (d < 2) throw std::invalid_argument("d must be at least 2");
    auto shape = shape_of(u);
    CArr out(shape);
    const cplx* p = u.data();
    cplx* q = out.mutable_data();
    std::size_t N = static_cast<std::size_t>(u.size());
    {
        py::gil_scoped_release release;
        parallel_for(N, [&](std::size_t a, std::size_t b) {
            for (std::size_t j = a; j < b; ++j) q[j] = post_n(p[j], d, n);
        });
    }
    return out;
}

// classifyOrbits: for k = 0..maxIterations, a point whose current value lies
// in trap j (the LAST j with |value - center_j| <= radius_j, as in the MATLAB
// loop) gets label j (1-based) and entry k.  NaN values stop unresolved.
py::tuple classify_call(const BasinMap& r, CArr z, CArr centers, DArr radii, int maxIterations) {
    if (centers.size() != radii.size())
        throw std::invalid_argument("centers and radii must have the same length");
    if (maxIterations < 0) throw std::invalid_argument("maxIterations must be non-negative");
    auto shape = shape_of(z);
    std::size_t N = static_cast<std::size_t>(z.size());
    std::vector<double> re, im;
    split(z, re, im);
    const int nt = static_cast<int>(centers.size());
    std::vector<double> cr(nt), ci(nt), rad(nt);
    for (int j = 0; j < nt; ++j) {
        cr[j] = centers.data()[j].real(); ci[j] = centers.data()[j].imag(); rad[j] = radii.data()[j];
    }
    py::array_t<std::int32_t> labels(shape);
    py::array_t<double> entry(shape);
    std::int32_t* L = labels.mutable_data();
    double* E = entry.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(N, [&](std::size_t a, std::size_t b) {
            // Work on the chunk in rounds, like the MATLAB code, so each
            // round evaluates r on a dense batch of still-active points.
            std::vector<std::size_t> active;
            active.reserve(b - a);
            for (std::size_t j = a; j < b; ++j) { L[j] = 0; E[j] = kNaN; active.push_back(j); }
            std::vector<double> vr(b - a), vi(b - a);
            for (std::size_t j = a; j < b; ++j) { vr[j - a] = re[j]; vi[j - a] = im[j]; }
            std::size_t count = b - a;
            for (int k = 0; k <= maxIterations && count > 0; ++k) {
                std::size_t keep = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vr[i], y = vi[i];
                    int hit = 0;
                    for (int j = 0; j < nt; ++j)
                        if (std::hypot(x - cr[j], y - ci[j]) <= rad[j]) hit = j + 1;
                    if (hit) { L[active[i]] = hit; E[active[i]] = k; continue; }
                    if (c_isnan(x, y)) continue;
                    active[keep] = active[i]; vr[keep] = x; vi[keep] = y; ++keep;
                }
                count = keep;
                if (k == maxIterations || count == 0) break;
                r.apply(vr.data(), vi.data(), count);
            }
        });
    }
    return py::make_tuple(labels, entry);
}

}  // namespace

PYBIND11_MODULE(_approx_kernel, m) {
    m.doc() = "C++ kernel of modules.approximation (arXiv:2605.09903, Section 2).";
    m.def("model_map_source", &model_map_source, py::arg("d"),
          "The expression the kernel uses for F, so the Python side builds the identical stage.");

    py::class_<Runge>(m, "Runge")
        .def(py::init<CArr, CArr, cplx>(), py::arg("poles"), py::arg("residues"), py::arg("shift") = cplx(0.0, 0.0))
        .def("__call__", &runge_call, py::arg("z"), "Evaluate S(z) elementwise (any shape).")
        .def("derivs", &runge_derivs, py::arg("z"),
             "Return (S, S', S'', sum_k 1/(z-p_k)) elementwise.")
        .def("taylor_bound", &runge_taylor, py::arg("z"), py::arg("delta"),
             "Return (S(z), B) with |S(w)-S(z)| <= B for |w-z| <= delta (order-10 Taylor enclosure).")
        .def("lipschitz", &runge_lipschitz, py::arg("z"), py::arg("delta"),
             "Return (S(z), L, dist): |S(w)-S(z)| <= delta*L for |w-z| <= delta; dist = nearest pole.")
        .def_property_readonly("degree", &Runge::degree);

    py::class_<BasinMap>(m, "BasinMap")
        .def(py::init<CArr, CArr, cplx, int, int, cplx, double>(),
             py::arg("poles"), py::arg("residues"), py::arg("shift"), py::arg("d"), py::arg("n"),
             py::arg("origin"), py::arg("scale"))
        .def("__call__", &map_call, py::arg("z"), "Evaluate r(z) = origin + scale*F^n(S(z)) elementwise.")
        .def("runge", [](const BasinMap& r, CArr z) { return runge_call(r.S, z); }, py::arg("z"))
        .def("classify", &classify_call, py::arg("z"), py::arg("centers"), py::arg("radii"),
             py::arg("max_iterations"),
             "Return (labels int32, entry float64) for orbits of r started at z.")
        .def_readonly("d", &BasinMap::d)
        .def_readonly("n", &BasinMap::n)
        .def_readonly("scale", &BasinMap::scale)
        .def_property_readonly("degree_S", [](const BasinMap& r) { return r.S.degree(); })
        .def_property_readonly("log10_degree", [](const BasinMap& r) { return r.composed.log10_degree(); });

    m.def("post_map", &post_call, py::arg("d"), py::arg("n"), py::arg("u"),
          "Apply F^(n), F(u) = u^d / (1 + d/(d-1) u^(d-1)), elementwise.");
    m.def("repulsion", &repulsion, py::arg("w"), py::arg("z"),
          "sum_j 1/(w_i - z_j) over z_j != w_i, for each i.");
    m.def("pair_sum", &pair_sum, py::arg("z"), "sum_{j != i} 1/(z_i - z_j) for each i.");
    m.def("set_num_threads", [](int t) { g_threads = std::max(0, t); }, py::arg("threads"),
          "Number of worker threads (0 = all hardware threads).");
    m.def("get_num_threads", []() { return thread_count(std::size_t(1) << 30); });
}

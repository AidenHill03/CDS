// =============================================================================
// cdx/renderer.cpp -- implementation of the Complex Dynamics core.
// =============================================================================
#include "cdx/renderer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace cdx {

namespace {
constexpr double kInf     = std::numeric_limits<double>::infinity();
constexpr double kHuge    = 1e300;
constexpr double kTinyDen = 1e-300;
inline bool is_bad(double v) { return !(v == v); }   // NaN test
}  // namespace

// -----------------------------------------------------------------------------
// Family names
// -----------------------------------------------------------------------------
std::string to_string(Family f) {
    switch (f) {
        case Family::Quadratic: return "quadratic";
        case Family::Cubic:     return "cubic";
        case Family::Quintic:   return "quintic";
        case Family::McMullen2: return "mcmullen2";
        case Family::McMullen3: return "mcmullen3";
        case Family::Newton3:   return "newton3";
    }
    return "unknown";
}

bool family_from_string(const std::string& s, Family& out) {
    if (s == "quadratic" || s == "mandelbrot") { out = Family::Quadratic; return true; }
    if (s == "cubic"     || s == "multibrot3") { out = Family::Cubic;     return true; }
    if (s == "quintic"   || s == "multibrot5") { out = Family::Quintic;   return true; }
    if (s == "mcmullen2")                      { out = Family::McMullen2; return true; }
    if (s == "mcmullen3")                      { out = Family::McMullen3; return true; }
    if (s == "newton3")                        { out = Family::Newton3;   return true; }
    return false;
}

// -----------------------------------------------------------------------------
// Map
// -----------------------------------------------------------------------------
int Map::degree() const {
    switch (family_) {
        case Family::Quadratic: return 2;
        case Family::Cubic:     return 3;
        case Family::Quintic:   return 5;
        case Family::McMullen2: return 4;   // z^2 + a/z^2 has degree 2n = 4
        case Family::McMullen3: return 6;   // z^3 + a/z^3 has degree 2n = 6
        case Family::Newton3:   return 3;
    }
    return 2;
}

void Map::step_with(Family f, double pr, double pi, double& zr, double& zi) {
    const double a = zr, b = zi;
    const double a2 = a * a, b2 = b * b;

    switch (f) {
        case Family::Quadratic:
            zr = a2 - b2 + pr;
            zi = 2.0 * a * b + pi;
            return;

        case Family::Cubic:
            zr = a * (a2 - 3.0 * b2) + pr;
            zi = b * (3.0 * a2 - b2) + pi;
            return;

        case Family::Quintic:
            zr = a * (a2 * a2 - 10.0 * a2 * b2 + 5.0 * b2 * b2) + pr;
            zi = b * (5.0 * a2 * a2 - 10.0 * a2 * b2 + b2 * b2) + pi;
            return;

        case Family::McMullen2: {
            const double sr = a2 - b2, si = 2.0 * a * b;      // z^2
            const double den = sr * sr + si * si;
            if (den < kTinyDen) { zr = kHuge; zi = 0.0; return; }
            zr = sr + (pr * sr + pi * si) / den;
            zi = si + (pi * sr - pr * si) / den;
            return;
        }

        case Family::McMullen3: {
            const double cr = a * (a2 - 3.0 * b2);            // z^3
            const double ci = b * (3.0 * a2 - b2);
            const double den = cr * cr + ci * ci;
            if (den < kTinyDen) { zr = kHuge; zi = 0.0; return; }
            zr = cr + (pr * cr + pi * ci) / den;
            zi = ci + (pi * cr - pr * ci) / den;
            return;
        }

        case Family::Newton3: {
            const double cr = a * (a2 - 3.0 * b2);            // z^3
            const double ci = b * (3.0 * a2 - b2);
            const double nr = cr - 1.0, ni = ci;              // z^3 - 1
            const double sr = 3.0 * (a2 - b2), si = 6.0 * a * b;  // 3 z^2
            const double den = sr * sr + si * si;
            if (den < kTinyDen) { zr = kHuge; zi = 0.0; return; }
            zr = a - (nr * sr + ni * si) / den;
            zi = b - (ni * sr - nr * si) / den;
            return;
        }
    }
}

void Map::step(double& zr, double& zi) const {
    step_with(family_, param_.real(), param_.imag(), zr, zi);
}

Cplx Map::critical_point(Family f, Cplx param) {
    switch (f) {
        case Family::Quadratic:
        case Family::Cubic:
        case Family::Quintic:
            return {0.0, 0.0};                 // only critical point of z^n + a

        case Family::McMullen2: {              // z^4 = a
            const double m = std::pow(std::abs(param), 0.25);
            const double t = std::arg(param) * 0.25;
            return {m * std::cos(t), m * std::sin(t)};
        }
        case Family::McMullen3: {              // z^6 = a
            const double m = std::pow(std::abs(param), 1.0 / 6.0);
            const double t = std::arg(param) / 6.0;
            return {m * std::cos(t), m * std::sin(t)};
        }
        case Family::Newton3:
            return {0.0, 0.0};                 // parameter-independent
    }
    return {0.0, 0.0};
}

// -----------------------------------------------------------------------------
// Chordal metric
// -----------------------------------------------------------------------------
double chordal_distance(double zr, double zi, double wr, double wi) {
    const bool z_inf = std::isinf(zr) || std::isinf(zi) ||
                       is_bad(zr) || is_bad(zi) || (zr * zr + zi * zi) > kHuge;
    const bool w_inf = std::isinf(wr) || std::isinf(wi);

    if (z_inf && w_inf) return 0.0;
    if (w_inf) return 2.0 / std::sqrt(1.0 + zr * zr + zi * zi);
    if (z_inf) return 2.0 / std::sqrt(1.0 + wr * wr + wi * wi);

    const double dr = zr - wr, di = zi - wi;
    const double num = 2.0 * std::sqrt(dr * dr + di * di);
    const double den = std::sqrt((1.0 + zr * zr + zi * zi) *
                                 (1.0 + wr * wr + wi * wi));
    return num / den;
}

// -----------------------------------------------------------------------------
// Renderer helpers
// -----------------------------------------------------------------------------
void Renderer::zoom(Cplx target, double factor) {
    if (factor <= 0.0) return;
    view_.center = target;
    view_.scale /= factor;
}

double Renderer::precision_floor() const {
    // Neighbouring grid points must remain distinct doubles. The spacing is
    // 2*scale/(res-1); it must exceed the ULP at the centre's magnitude.
    const double mag = std::max(1.0, std::abs(view_.center));
    return mag * std::numeric_limits<double>::epsilon() * view_.resolution;
}

template <typename F>
void Renderer::parallel_columns(F body) const {
    const int cols = view_.resolution;
    unsigned n = settings_.threads > 0
                     ? static_cast<unsigned>(settings_.threads)
                     : std::thread::hardware_concurrency();
    if (n == 0) n = 1;
    n = std::min<unsigned>(n, static_cast<unsigned>(std::max(1, cols)));

    if (n == 1) {
        for (int c = 0; c < cols; ++c) body(c);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(n);
    const int chunk = (cols + static_cast<int>(n) - 1) / static_cast<int>(n);
    for (unsigned t = 0; t < n; ++t) {
        const int lo = static_cast<int>(t) * chunk;
        const int hi = std::min(cols, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&body, lo, hi] {
            for (int c = lo; c < hi; ++c) body(c);
        });
    }
    for (auto& th : pool) th.join();
}

// -----------------------------------------------------------------------------
// Julia
// -----------------------------------------------------------------------------
Image Renderer::render_julia() const {
    const int  res  = view_.resolution;
    const double R2 = settings_.escape_radius * settings_.escape_radius;
    const double inv_log2 = 1.0 / std::log(2.0);
    Image img(res, res);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            double zr = c.real(), zi = c.imag();
            double out = 0.0;

            for (int n = 0; n < settings_.max_iter; ++n) {
                map_.step(zr, zi);
                const double m2 = zr * zr + zi * zi;

                if (m2 > R2) {                       // per-pixel early exit
                    const double nu = static_cast<double>(n + 1) + 1.0 -
                                      std::log(std::log(std::sqrt(m2))) * inv_log2;
                    out = is_bad(nu) ? static_cast<double>(n + 1) : nu;
                    break;
                }
                if (is_bad(m2) || m2 > kHuge) { out = static_cast<double>(n + 1); break; }
            }
            img.at(col, row) = out;
        }
    });
    return img;
}

// -----------------------------------------------------------------------------
// Parameter plane
// -----------------------------------------------------------------------------
Image Renderer::render_parameter() const {
    const int    res = view_.resolution;
    const double R2  = settings_.escape_radius * settings_.escape_radius;
    const double inv_log2 = 1.0 / std::log(2.0);
    const Family fam = map_.family();
    Image img(res, res);

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx p  = view_.coord(col, row);      // the PIXEL is the parameter
            const Cplx c0 = Map::critical_point(fam, p);
            double zr = c0.real(), zi = c0.imag();
            const double pr = p.real(), pi = p.imag();
            double out = 0.0;

            for (int n = 0; n < settings_.max_iter; ++n) {
                Map::step_with(fam, pr, pi, zr, zi);
                const double m2 = zr * zr + zi * zi;

                if (m2 > R2) {
                    const double nu = static_cast<double>(n + 1) + 1.0 -
                                      std::log(std::log(std::sqrt(m2))) * inv_log2;
                    out = is_bad(nu) ? static_cast<double>(n + 1) : nu;
                    break;
                }
                if (is_bad(m2) || m2 > kHuge) { out = static_cast<double>(n + 1); break; }
            }
            img.at(col, row) = out;
        }
    });
    return img;
}

// -----------------------------------------------------------------------------
// Basins
// -----------------------------------------------------------------------------
Image Renderer::render_basin(const std::vector<Cycle>& cycles) const {
    const int res = view_.resolution;
    Image img(res, res);
    if (cycles.empty()) return img;

    // Flatten to parallel arrays: better locality in the innermost test.
    std::vector<double> ar, ai;
    std::vector<int>    aid;
    for (const auto& cyc : cycles) {
        for (const auto& pt : cyc.points) {
            ar.push_back(pt.real());
            ai.push_back(pt.imag());
            aid.push_back(cyc.id);
        }
    }
    const int nattr = static_cast<int>(ar.size());
    const double tol = settings_.tol;

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            double zr = c.real(), zi = c.imag();
            double label = 0.0;

            for (int n = 0; n < settings_.max_iter; ++n) {
                map_.step(zr, zi);

                for (int k = 0; k < nattr; ++k) {
                    if (chordal_distance(zr, zi, ar[k], ai[k]) < tol) {
                        label = static_cast<double>(aid[k]);
                        break;
                    }
                }
                if (label > 0.0) break;
                if (is_bad(zr) || is_bad(zi)) break;
            }
            img.at(col, row) = label;
        }
    });
    return img;
}

// -----------------------------------------------------------------------------
// Green's function
// -----------------------------------------------------------------------------
Image Renderer::render_greens(bool* normalized) const {
    const int    res  = view_.resolution;
    const double escR = settings_.escape_radius;
    Image img(res, res);

    double norm = std::pow(static_cast<double>(map_.degree()),
                           static_cast<double>(settings_.max_iter));
    const bool ok = !(is_bad(norm) || norm > kHuge || norm <= 0.0);
    if (!ok) norm = 1.0;
    if (normalized) *normalized = ok;

    parallel_columns([&](int col) {
        for (int row = 0; row < res; ++row) {
            const Cplx c = view_.coord(col, row);
            double zr = c.real(), zi = c.imag();
            double acc = 0.0;

            for (int n = 0; n < settings_.max_iter; ++n) {
                map_.step(zr, zi);
                const double mag = std::sqrt(zr * zr + zi * zi);
                if (is_bad(mag)) break;
                acc += std::log(mag > 1.0 ? mag : 1.0);
                if (mag > escR) break;
            }
            acc /= norm;
            if (is_bad(acc) || acc > kHuge || acc < -kHuge) acc = 0.0;
            img.at(col, row) = acc;
        }
    });
    return img;
}

}  // namespace cdx

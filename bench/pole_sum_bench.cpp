// bench/pole_sum_bench.cpp -- how much would evaluating the Runge sum
//     S(z) = sum_k res_k / (z - p_k)
// in BLOCKS of points (pole loop outside, point loop inside) gain over the
// one-point-at-a-time loop cdx::ComposedMap::step runs today -- and does it
// change any result?
//
// Both loops below add the poles' contributions to each point in the SAME
// order k = 0, 1, 2, ..., so each point's sum is the same sequence of IEEE
// operations either way; the blocked form only changes which point is being
// worked on between two consecutive poles. The program checks that the two
// outputs are bit-identical and prints the single-thread rate of each.
//
// A measurement tool for a PROPOSAL (a block-evaluation path through
// ComposedMap is an engine API change); nothing in the engine uses it.
//
//   c++ -std=c++17 -O3 bench/pole_sum_bench.cpp -o bench/out/pole_sum_bench && bench/out/pole_sum_bench
// (same flags as cdx/CMakeLists.txt: -O3, no -ffast-math)
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kBlock = 64;

// cdx/src/composed.cpp, eval_terms' simple-pole loop, verbatim.
void scalar(const std::vector<double>& LR, const std::vector<double>& LI,
            const std::vector<double>& SR, const std::vector<double>& SI,
            const std::vector<double>& zr, const std::vector<double>& zi,
            std::vector<double>& outr, std::vector<double>& outi) {
    const std::size_t m = LR.size();
    for (std::size_t j = 0; j < zr.size(); ++j) {
        const double xr = zr[j], xi = zi[j];
        double ar = 0.0, ai = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            const double dx = xr - LR[k], dy = xi - LI[k];
            const double den = dx * dx + dy * dy;
            const double ix = dx / den, iy = -dy / den;
            ar += ix * SR[k] - iy * SI[k];
            ai += ix * SI[k] + iy * SR[k];
        }
        outr[j] = ar; outi[j] = ai;
    }
}

// The same arithmetic, pole loop outside, up to kBlock points inside.
void blocked(const std::vector<double>& LR, const std::vector<double>& LI,
             const std::vector<double>& SR, const std::vector<double>& SI,
             const std::vector<double>& zr, const std::vector<double>& zi,
             std::vector<double>& outr, std::vector<double>& outi) {
    const std::size_t m = LR.size();
    double xr[kBlock], xi[kBlock], ar[kBlock], ai[kBlock];
    for (std::size_t s = 0; s < zr.size(); s += kBlock) {
        const int nb = static_cast<int>(std::min<std::size_t>(kBlock, zr.size() - s));
        for (int i = 0; i < nb; ++i) { xr[i] = zr[s + i]; xi[i] = zi[s + i]; ar[i] = 0.0; ai[i] = 0.0; }
        for (std::size_t k = 0; k < m; ++k) {
            const double px = LR[k], py = LI[k], qx = SR[k], qy = SI[k];
            for (int i = 0; i < nb; ++i) {
                const double dx = xr[i] - px, dy = xi[i] - py;
                const double den = dx * dx + dy * dy;
                const double ix = dx / den, iy = -dy / den;
                ar[i] += ix * qx - iy * qy;
                ai[i] += ix * qy + iy * qx;
            }
        }
        for (int i = 0; i < nb; ++i) { outr[s + i] = ar[i]; outi[s + i] = ai[i]; }
    }
}

template <typename F>
double seconds(F f, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt < best) best = dt;
    }
    return best;
}

}  // namespace

int main() {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t m : {1000u, 4900u, 46288u}) {
        const std::size_t npts = 4096;
        std::vector<double> LR(m), LI(m), SR(m), SI(m), zr(npts), zi(npts);
        for (std::size_t k = 0; k < m; ++k) { LR[k] = u(rng); LI[k] = u(rng); SR[k] = 1e-3 * u(rng); SI[k] = 1e-3 * u(rng); }
        for (std::size_t j = 0; j < npts; ++j) { zr[j] = u(rng); zi[j] = u(rng); }
        std::vector<double> ar(npts), ai(npts), br(npts), bi(npts);
        const double ts = seconds([&] { scalar(LR, LI, SR, SI, zr, zi, ar, ai); }, 5);
        const double tb = seconds([&] { blocked(LR, LI, SR, SI, zr, zi, br, bi); }, 5);
        const bool same = std::memcmp(ar.data(), br.data(), npts * sizeof(double)) == 0 &&
                          std::memcmp(ai.data(), bi.data(), npts * sizeof(double)) == 0;
        const double terms = static_cast<double>(m) * static_cast<double>(npts);
        std::printf("poles %6zu: scalar %.3g terms/s, blocked %.3g terms/s, ratio %.2fx, outputs %s\n",
                    m, terms / ts, terms / tb, ts / tb, same ? "BIT-IDENTICAL" : "DIFFER");
    }
    return 0;
}

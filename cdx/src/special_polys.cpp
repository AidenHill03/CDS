// =============================================================================
// cdx/special_polys.cpp -- see special_polys.hpp for what each family is and
// why it is computed the way it is.
// =============================================================================
#include "cdx/special_polys.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace cdx {
namespace special {

namespace {

// ---- SymPoly arithmetic -------------------------------------------------------
SymPoly sp_const(long double v) { SymPoly p; p.c = {{v}}; return p; }

// s_k + offset, k = 0 (alpha / lambda) or 1 (beta).
SymPoly sp_linear(int k, long double offset) {
    SymPoly p;
    if (k == 0) p.c = {{offset}, {1.0L}};
    else        p.c = {{offset, 1.0L}};
    return p;
}

SymPoly sp_add(const SymPoly& a, const SymPoly& b, long double sb = 1.0L) {
    SymPoly out;
    const std::size_t ni = std::max(a.c.size(), b.c.size());
    std::size_t nj = 0;
    for (const auto& r : a.c) nj = std::max(nj, r.size());
    for (const auto& r : b.c) nj = std::max(nj, r.size());
    out.c.assign(ni, std::vector<long double>(nj, 0.0L));
    for (std::size_t i = 0; i < a.c.size(); ++i)
        for (std::size_t j = 0; j < a.c[i].size(); ++j) out.c[i][j] += a.c[i][j];
    for (std::size_t i = 0; i < b.c.size(); ++i)
        for (std::size_t j = 0; j < b.c[i].size(); ++j) out.c[i][j] += sb * b.c[i][j];
    return out;
}

SymPoly sp_scale(const SymPoly& a, long double s) {
    SymPoly out = a;
    for (auto& r : out.c) for (auto& v : r) v *= s;
    return out;
}

SymPoly sp_mul(const SymPoly& a, const SymPoly& b) {
    std::size_t aj = 0, bj = 0;
    for (const auto& r : a.c) aj = std::max(aj, r.size());
    for (const auto& r : b.c) bj = std::max(bj, r.size());
    SymPoly out;
    out.c.assign(a.c.size() + b.c.size() - 1, std::vector<long double>(aj + bj - 1, 0.0L));
    for (std::size_t i = 0; i < a.c.size(); ++i)
        for (std::size_t j = 0; j < a.c[i].size(); ++j) {
            if (a.c[i][j] == 0.0L) continue;
            for (std::size_t k = 0; k < b.c.size(); ++k)
                for (std::size_t l = 0; l < b.c[k].size(); ++l)
                    out.c[i + k][j + l] += a.c[i][j] * b.c[k][l];
        }
    return out;
}

// ---- TPoly arithmetic -----------------------------------------------------------
TPoly tp_const(long double v) { return {sp_const(v)}; }

TPoly tp_add(const TPoly& a, const TPoly& b, long double sb = 1.0L) {
    TPoly out(std::max(a.size(), b.size()), sp_const(0.0L));
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) out[i] = sp_add(out[i], b[i], sb);
    return out;
}

TPoly tp_mul_sym(const TPoly& a, const SymPoly& s) {
    TPoly out;
    out.reserve(a.size());
    for (const auto& c : a) out.push_back(sp_mul(c, s));
    return out;
}

TPoly tp_scale(const TPoly& a, long double s) {
    TPoly out;
    out.reserve(a.size());
    for (const auto& c : a) out.push_back(sp_scale(c, s));
    return out;
}

TPoly tp_shift(const TPoly& a) {   // multiply by t
    TPoly out;
    out.reserve(a.size() + 1);
    out.push_back(sp_const(0.0L));
    for (const auto& c : a) out.push_back(c);
    return out;
}

TPoly tp_from_numeric(const std::vector<long double>& v) {
    TPoly out;
    out.reserve(v.size());
    for (long double x : v) out.push_back(sp_const(x));
    return out;
}

// ---- generic three-term recurrence driver ---------------------------------------
// p_{k+1} = A_k(t) p_k + B_k p_{k-1}, with p_0, p_1 given.
template <class Step>
TPoly recur(int n, TPoly p0, TPoly p1, Step step) {
    if (n == 0) return p0;
    TPoly prev = std::move(p0), cur = std::move(p1);
    for (int k = 1; k < n; ++k) {
        TPoly next = step(k, cur, prev);
        prev = std::move(cur);
        cur = std::move(next);
    }
    return cur;
}

// ---- exact Bernoulli numbers -----------------------------------------------------
// Sign-magnitude big integer, base 2^32 -- just the operations the
// Stirling-number formula needs (add, multiply by a machine word, convert).
struct BigInt {
    bool neg = false;
    std::vector<std::uint32_t> mag;   // little-endian limbs, no leading zeros

    static BigInt from_u64(std::uint64_t v) {
        BigInt b;
        while (v) { b.mag.push_back(static_cast<std::uint32_t>(v)); v >>= 32; }
        return b;
    }
    bool is_zero() const { return mag.empty(); }
    void trim() { while (!mag.empty() && mag.back() == 0) mag.pop_back(); if (mag.empty()) neg = false; }

    void mul_u32(std::uint32_t m) {
        std::uint64_t carry = 0;
        for (auto& limb : mag) {
            const std::uint64_t v = static_cast<std::uint64_t>(limb) * m + carry;
            limb = static_cast<std::uint32_t>(v);
            carry = v >> 32;
        }
        if (carry) mag.push_back(static_cast<std::uint32_t>(carry));
        trim();
    }
    void mul_u64(std::uint64_t m) {
        BigInt lo = *this, hi = *this;
        lo.mul_u32(static_cast<std::uint32_t>(m));
        hi.mul_u32(static_cast<std::uint32_t>(m >> 32));
        if (!hi.mag.empty()) hi.mag.insert(hi.mag.begin(), 0u);   // * 2^32
        *this = add(lo, hi);
    }

    static int cmp_mag(const BigInt& a, const BigInt& b) {
        if (a.mag.size() != b.mag.size()) return a.mag.size() < b.mag.size() ? -1 : 1;
        for (std::size_t i = a.mag.size(); i-- > 0;)
            if (a.mag[i] != b.mag[i]) return a.mag[i] < b.mag[i] ? -1 : 1;
        return 0;
    }
    static std::vector<std::uint32_t> add_mag(const BigInt& a, const BigInt& b) {
        std::vector<std::uint32_t> out;
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < std::max(a.mag.size(), b.mag.size()); ++i) {
            const std::uint64_t v = carry + (i < a.mag.size() ? a.mag[i] : 0u) +
                                    (i < b.mag.size() ? b.mag[i] : 0u);
            out.push_back(static_cast<std::uint32_t>(v));
            carry = v >> 32;
        }
        if (carry) out.push_back(static_cast<std::uint32_t>(carry));
        return out;
    }
    static std::vector<std::uint32_t> sub_mag(const BigInt& a, const BigInt& b) {   // |a| >= |b|
        std::vector<std::uint32_t> out;
        std::int64_t borrow = 0;
        for (std::size_t i = 0; i < a.mag.size(); ++i) {
            std::int64_t v = static_cast<std::int64_t>(a.mag[i]) - borrow -
                             (i < b.mag.size() ? static_cast<std::int64_t>(b.mag[i]) : 0);
            borrow = v < 0 ? 1 : 0;
            if (v < 0) v += (static_cast<std::int64_t>(1) << 32);
            out.push_back(static_cast<std::uint32_t>(v));
        }
        return out;
    }
    static BigInt add(const BigInt& a, const BigInt& b) {
        BigInt out;
        if (a.neg == b.neg) {
            out.mag = add_mag(a, b);
            out.neg = a.neg;
        } else if (cmp_mag(a, b) >= 0) {
            out.mag = sub_mag(a, b);
            out.neg = a.neg;
        } else {
            out.mag = sub_mag(b, a);
            out.neg = b.neg;
        }
        out.trim();
        return out;
    }
    long double to_ld() const {
        long double v = 0.0L;
        for (std::size_t i = mag.size(); i-- > 0;) v = v * 4294967296.0L + mag[i];
        return neg ? -v : v;
    }
};

std::uint64_t gcd_u64(std::uint64_t a, std::uint64_t b) { while (b) { auto t = a % b; a = b; b = t; } return a; }

const std::vector<long double>& bernoulli_table() {
    // B_0 .. B_(kMaxDegree+1): Euler polynomials of degree n need B_(n+1).
    static const std::vector<long double> table = [] {
        const int kmax = kMaxDegree + 1;
        std::uint64_t L = 1;   // lcm(1 .. kmax+1): every j+1 below divides it
        for (std::uint64_t m = 2; m <= static_cast<std::uint64_t>(kmax + 1); ++m) L = L / gcd_u64(L, m) * m;

        std::vector<BigInt> fact(kmax + 1);   // j!
        fact[0] = BigInt::from_u64(1);
        for (int j = 1; j <= kmax; ++j) { fact[j] = fact[j - 1]; fact[j].mul_u32(static_cast<std::uint32_t>(j)); }

        std::vector<long double> out(kmax + 1);
        // Stirling numbers of the second kind, row by row: S(k, j).
        std::vector<BigInt> S(1, BigInt::from_u64(1));   // row k = 0: S(0,0) = 1
        for (int k = 0; k <= kmax; ++k) {
            if (k > 0) {
                std::vector<BigInt> next(k + 1);
                for (int j = 0; j <= k; ++j) {
                    BigInt a = j < static_cast<int>(S.size()) ? S[j] : BigInt{};
                    a.mul_u32(static_cast<std::uint32_t>(j));
                    BigInt b = j >= 1 ? S[j - 1] : BigInt{};
                    next[j] = BigInt::add(a, b);
                }
                S = std::move(next);
            }
            // B_k = sum_j (-1)^j j! S(k,j) / (j+1) = N / L.
            BigInt N;
            for (int j = 0; j <= k; ++j) {
                if (S[j].is_zero()) continue;
                BigInt term = S[j];
                // j! * S(k,j) * (L / (j+1)) -- multiply limb by limb of j!
                // via repeated u32 multiplication (j! itself is a BigInt).
                BigInt prod;
                for (std::size_t limb = fact[j].mag.size(); limb-- > 0;) {
                    if (!prod.is_zero()) prod.mag.insert(prod.mag.begin(), 0u);   // * 2^32
                    BigInt part = term;
                    part.mul_u32(fact[j].mag[limb]);
                    prod = BigInt::add(prod, part);
                }
                prod.mul_u64(L / static_cast<std::uint64_t>(j + 1));
                if (j % 2 == 1) prod.neg = !prod.is_zero();
                N = BigInt::add(N, prod);
            }
            out[k] = N.to_ld() / static_cast<long double>(L);
        }
        return out;
    }();
    return table;
}

long double binom(int n, int k) {
    if (k < 0 || k > n) return 0.0L;
    long double r = 1.0L;
    for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

long double factorial(int n) {
    long double r = 1.0L;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

// ---- cyclotomic: exact integer arithmetic ---------------------------------------
using IPoly = std::vector<std::int64_t>;

// a / b for monic b dividing a exactly.
IPoly ipoly_div_exact(IPoly a, const IPoly& b) {
    const std::size_t db = b.size() - 1;
    IPoly q(a.size() - db, 0);
    for (std::size_t i = a.size(); i-- > db;) {
        const std::int64_t coef = a[i];   // b monic
        q[i - db] = coef;
        for (std::size_t j = 0; j <= db; ++j) a[i - db + j] -= coef * b[j];
    }
    return q;
}

const IPoly& cyclotomic_int(int n) {
    static std::map<int, IPoly> memo;
    auto it = memo.find(n);
    if (it != memo.end()) return it->second;
    IPoly p(n + 1, 0);
    p[0] = -1;
    p[n] = 1;   // t^n - 1
    for (int d = 1; d < n; ++d)
        if (n % d == 0) p = ipoly_div_exact(p, cyclotomic_int(d));
    return memo[n] = p;
}

}  // namespace

bool is_constant(const SymPoly& p) {
    for (std::size_t i = 0; i < p.c.size(); ++i)
        for (std::size_t j = 0; j < p.c[i].size(); ++j)
            if ((i || j) && p.c[i][j] != 0.0L) return false;
    return true;
}

long double bernoulli_number(int k) {
    const auto& t = bernoulli_table();
    if (k < 0 || k >= static_cast<int>(t.size())) throw std::out_of_range("bernoulli_number");
    return t[k];
}

TPoly special_poly(Family family, int n) {
    const TPoly one = tp_const(1.0L);
    const TPoly t = {sp_const(0.0L), sp_const(1.0L)};

    switch (family) {
        case Family::Hermite:   // H_{k+1} = 2t H_k - 2k H_{k-1}
            return recur(n, one, tp_scale(t, 2.0L), [](int k, const TPoly& c, const TPoly& p) {
                return tp_add(tp_scale(tp_shift(c), 2.0L), p, -2.0L * k);
            });
        case Family::HermiteE:  // He_{k+1} = t He_k - k He_{k-1}
            return recur(n, one, t, [](int k, const TPoly& c, const TPoly& p) {
                return tp_add(tp_shift(c), p, -1.0L * k);
            });
        case Family::Legendre:  // (k+1) P_{k+1} = (2k+1) t P_k - k P_{k-1}
            return recur(n, one, t, [](int k, const TPoly& c, const TPoly& p) {
                return tp_scale(tp_add(tp_scale(tp_shift(c), 2.0L * k + 1), p, -1.0L * k),
                                1.0L / (k + 1));
            });
        case Family::Laguerre: {  // (k+1) L_{k+1} = (2k+1+alpha-t) L_k - (k+alpha) L_{k-1}
            TPoly l1 = {sp_linear(0, 1.0L), sp_const(-1.0L)};   // 1 + alpha - t
            return recur(n, one, l1, [](int k, const TPoly& c, const TPoly& p) {
                TPoly a = tp_mul_sym(c, sp_linear(0, 2.0L * k + 1));
                a = tp_add(a, tp_shift(c), -1.0L);
                a = tp_add(a, tp_mul_sym(p, sp_linear(0, 1.0L * k)), -1.0L);
                return tp_scale(a, 1.0L / (k + 1));
            });
        }
        case Family::Gegenbauer: {  // (k+1) C_{k+1} = 2(k+lam) t C_k - (k+2lam-1) C_{k-1}
            TPoly c1 = {sp_const(0.0L), sp_scale(sp_linear(0, 0.0L), 2.0L)};   // 2 lam t
            return recur(n, one, c1, [](int k, const TPoly& c, const TPoly& p) {
                TPoly a = tp_scale(tp_mul_sym(tp_shift(c), sp_linear(0, 1.0L * k)), 2.0L);
                SymPoly f = sp_add(sp_scale(sp_linear(0, 0.0L), 2.0L), sp_const(k - 1.0L));
                a = tp_add(a, tp_mul_sym(p, f), -1.0L);
                return tp_scale(a, 1.0L / (k + 1));
            });
        }
        case Family::Jacobi: {
            // sum_s C(n+a, n-s) C(n+b, s) 2^-n (t-1)^s (t+1)^(n-s)
            TPoly out(n + 1, sp_const(0.0L));
            for (int s = 0; s <= n; ++s) {
                SymPoly k = sp_const(1.0L / (factorial(n - s) * factorial(s)) / std::pow(2.0L, n));
                for (int i = 1; i <= n - s; ++i) k = sp_mul(k, sp_linear(0, 1.0L * (s + i)));
                for (int i = 1; i <= s; ++i) k = sp_mul(k, sp_linear(1, 1.0L * (n - s + i)));
                // (t-1)^s (t+1)^(n-s), numerically
                std::vector<long double> e(1, 1.0L);
                for (int m = 0; m < n; ++m) {
                    const long double r = m < s ? -1.0L : 1.0L;   // root factor (t + r)
                    std::vector<long double> ne(e.size() + 1, 0.0L);
                    for (std::size_t j = 0; j < e.size(); ++j) { ne[j] += r * e[j]; ne[j + 1] += e[j]; }
                    e = std::move(ne);
                }
                for (int j = 0; j <= n; ++j)
                    if (e[j] != 0.0L) out[j] = sp_add(out[j], sp_scale(k, e[j]));
            }
            return out;
        }
        case Family::ChebyshevT:
        case Family::ChebyshevU:   // X_{k+1} = 2t X_k - X_{k-1}
            return recur(n, one, family == Family::ChebyshevT ? t : tp_scale(t, 2.0L),
                         [](int, const TPoly& c, const TPoly& p) {
                             return tp_add(tp_scale(tp_shift(c), 2.0L), p, -1.0L);
                         });
        case Family::Bessel: {    // y_{k+1} = (2k+1) t y_k + y_{k-1}
            TPoly y1 = {sp_const(1.0L), sp_const(1.0L)};
            return recur(n, one, y1, [](int k, const TPoly& c, const TPoly& p) {
                return tp_add(tp_scale(tp_shift(c), 2.0L * k + 1), p, 1.0L);
            });
        }
        case Family::Bernoulli: {
            std::vector<long double> v(n + 1);
            for (int j = 0; j <= n; ++j) v[j] = binom(n, j) * bernoulli_number(n - j);
            return tp_from_numeric(v);
        }
        case Family::Euler: {
            std::vector<long double> v(n + 1);
            for (int j = 0; j <= n; ++j)
                v[j] = 2.0L / (n + 1) * binom(n + 1, j) * bernoulli_number(n + 1 - j) *
                       (1.0L - std::pow(2.0L, n + 1 - j));
            return tp_from_numeric(v);
        }
        case Family::Cyclotomic: {
            const IPoly& p = cyclotomic_int(n);
            std::vector<long double> v(p.begin(), p.end());
            return tp_from_numeric(v);
        }
    }
    throw std::logic_error("special_poly: unknown family");
}

}  // namespace special
}  // namespace cdx

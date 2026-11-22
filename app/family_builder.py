"""app/family_builder.py -- maps built from classical polynomials.

The logic behind the Equation tab's "Build ▾" popover (app/build_popover.py)
and the four builder presets in the Library: pick a polynomial p(z) (a
special polynomial, z^n - 1, or custom text), pick a map built on it, and get
an EXPRESSION in the rational grammar -- written with the parser's own
special-polynomial functions and d(...) (cdx/rational_parser.hpp), e.g.

    z - a*hermite(5,z)/d(hermite(5,z))

so the user can still read and edit it in the f(z) field. Nothing here
expands a polynomial: cdx does that when it parses the text.

No Qt here; cdx only for validating and measuring the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import cdx

# A generic parameter value for measuring a built map (degree, free critical
# points): away from the special values (0, 1, ...) where a family degenerates.
GENERIC_A = complex(0.71, 0.29)

MAX_DEGREE = 40
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


@dataclass(frozen=True)
class PolyType:
    key: str
    label: str              # in the type list
    symbols: tuple[str, ...]   # family parameters: "alpha", "beta", "lam"
    min_degree: int = 1


POLY_TYPES: list[PolyType] = [
    PolyType("hermite", "Hermite H", ()),
    PolyType("hermite_e", "Hermite He", ()),
    PolyType("legendre", "Legendre P", ()),
    PolyType("laguerre", "Laguerre L", ()),
    PolyType("gen_laguerre", "Generalized Laguerre L^(α)", ("alpha",)),
    PolyType("chebyshev_t", "Chebyshev T", ()),
    PolyType("chebyshev_u", "Chebyshev U", ()),
    PolyType("gegenbauer", "Gegenbauer C^(λ)", ("lam",)),
    PolyType("jacobi", "Jacobi P^(α,β)", ("alpha", "beta")),
    PolyType("bessel", "Bessel y", ()),
    PolyType("bernoulli", "Bernoulli B", ()),
    PolyType("euler", "Euler E", ()),
    PolyType("cyclotomic", "Cyclotomic Φ", ()),
    PolyType("roots_of_unity", "zⁿ − 1", ()),
    PolyType("custom", "custom…", ()),
]
POLY_BY_KEY = {t.key: t for t in POLY_TYPES}

SYMBOL_LABEL = {"alpha": "α", "beta": "β", "lam": "λ"}
SYMBOL_DEFAULT = {"alpha": "a", "beta": "0", "lam": "1"}


@dataclass(frozen=True)
class MapKind:
    key: str
    label: str      # in the map list
    formula: str    # shown beside it
    name: str       # in generated family names ("" -> special-cased)


MAP_KINDS: list[MapKind] = [
    MapKind("p_plus_a", "p itself + parameter", "p(z) + a", ""),
    MapKind("a_times_p", "a · p(z)", "a·p(z)", ""),
    MapKind("newton", "Newton", "z − p/p′", "Newton"),
    MapKind("relaxed_newton", "Relaxed Newton", "z − a·p/p′", "Relaxed Newton"),
    MapKind("halley", "Halley", "z − 2pp′/(2p′² − pp″)", "Halley"),
    MapKind("chebyshev_method", "Chebyshev's method", "z − (1 + pp″/(2p′²))·p/p′", "Chebyshev's method"),
    MapKind("damped_halley", "Damped Halley", "the Halley step scaled by a", "Damped Halley"),
]
MAP_BY_KEY = {m.key: m for m in MAP_KINDS}


@dataclass(frozen=True)
class BuilderChoice:
    poly: str = "hermite"
    degree: int = 5
    alpha: str = "a"
    beta: str = "0"
    lam: str = "1"
    custom: str = "z^3 - 1"
    map_kind: str = "relaxed_newton"


class BuilderError(ValueError):
    pass


def _symbol_text(text: str, what: str) -> str:
    t = text.strip()
    if not t:
        raise BuilderError(f"{what} is empty -- a number or a")
    return t


def poly_expression(c: BuilderChoice) -> str:
    """p(z) as grammar text: hermite(5,z), laguerre(4,a,z), (z^5 - 1), (custom)."""
    n = c.degree
    t = POLY_BY_KEY.get(c.poly)
    if t is None:
        raise BuilderError(f"unknown polynomial type {c.poly!r}")
    if not (t.min_degree <= n <= MAX_DEGREE):
        raise BuilderError(f"degree must be between {t.min_degree} and {MAX_DEGREE}")
    simple = {"hermite": "hermite", "hermite_e": "hermite_e", "legendre": "legendre",
              "laguerre": "laguerre", "chebyshev_t": "chebyshev_t", "chebyshev_u": "chebyshev_u",
              "bessel": "bessel_poly", "bernoulli": "bernoulli", "euler": "euler_poly",
              "cyclotomic": "cyclotomic"}
    if c.poly in simple:
        return f"{simple[c.poly]}({n},z)"
    if c.poly == "gen_laguerre":
        return f"laguerre({n},{_symbol_text(c.alpha, 'α')},z)"
    if c.poly == "gegenbauer":
        return f"gegenbauer({n},{_symbol_text(c.lam, 'λ')},z)"
    if c.poly == "jacobi":
        return f"jacobi({n},{_symbol_text(c.alpha, 'α')},{_symbol_text(c.beta, 'β')},z)"
    if c.poly == "roots_of_unity":
        return f"(z^{n} - 1)"
    text = c.custom.strip()
    if not text:
        raise BuilderError("custom p(z) is empty")
    return f"({text})"


def map_expression(c: BuilderChoice) -> str:
    """The whole map as grammar text, e.g. z - a*hermite(5,z)/d(hermite(5,z))."""
    p = poly_expression(c)
    k = c.map_kind
    if k == "p_plus_a":
        return f"{p} + a"
    if k == "a_times_p":
        return f"a*{p}"
    if k == "newton":
        return f"z - {p}/d({p})"
    if k == "relaxed_newton":
        return f"z - a*{p}/d({p})"
    halley_step = f"2*{p}*d({p})/(2*d({p})^2 - {p}*d(d({p})))"
    if k == "halley":
        return f"z - {halley_step}"
    if k == "damped_halley":
        return f"z - a*{halley_step}"
    if k == "chebyshev_method":
        return f"z - (1 + {p}*d(d({p}))/(2*d({p})^2))*{p}/d({p})"
    raise BuilderError(f"unknown map {k!r}")


def poly_label(c: BuilderChoice) -> str:
    """'Hermite H₅', 'Laguerre L₄^(a)', 'Jacobi P₃^(a,0)', 'z⁵ − 1', 'custom p'."""
    n = str(c.degree).translate(_SUB)
    names = {"hermite": "Hermite H", "hermite_e": "Hermite He", "legendre": "Legendre P",
             "laguerre": "Laguerre L", "chebyshev_t": "Chebyshev T", "chebyshev_u": "Chebyshev U",
             "bessel": "Bessel y", "bernoulli": "Bernoulli B", "euler": "Euler E",
             "cyclotomic": "Cyclotomic Φ"}
    if c.poly in names:
        return names[c.poly] + n
    if c.poly == "gen_laguerre":
        return f"Laguerre L{n}^({c.alpha.strip()})"
    if c.poly == "gegenbauer":
        return f"Gegenbauer C{n}^({c.lam.strip()})"
    if c.poly == "jacobi":
        return f"Jacobi P{n}^({c.alpha.strip()},{c.beta.strip()})"
    if c.poly == "roots_of_unity":
        return f"z{str(c.degree).translate(_SUP)} − 1"
    return f"p = {c.custom.strip()}"


def family_name(c: BuilderChoice) -> str:
    """The generated Library name, e.g. 'Relaxed Newton · Hermite H₅'."""
    p = poly_label(c)
    if c.map_kind == "p_plus_a":
        return f"{p} + a"
    if c.map_kind == "a_times_p":
        return f"a · {p}"
    return f"{MAP_BY_KEY[c.map_kind].name} · {p}"


def unique_name(name: str, existing) -> str:
    existing = set(existing)
    if name not in existing:
        return name
    k = 2
    while f"{name} ({k})" in existing:
        k += 1
    return f"{name} ({k})"


@dataclass
class Built:
    expression: str
    parameters: list[str]
    degree: int
    free_critical_points: int | None
    warnings: list[str]


def build(c: BuilderChoice, measure: bool = True) -> Built:
    """Builds and validates the choice. Raises BuilderError with cdx's own
    message when the expression does not parse, or when it uses any
    parameter other than a (the map's single parameter)."""
    expr = map_expression(c)
    try:
        info = cdx.parse_rational_info(expr)
    except ValueError as e:
        raise BuilderError(str(e)) from None
    params = list(info["parameters"])
    if any(p != "a" for p in params):
        raise BuilderError(f"only the parameter a may appear (found {', '.join(params)})")
    degree, free = -1, None
    if measure:
        m = cdx.RationalMap.from_expression(expr, "a" if params else "", {}, "built")
        a = GENERIC_A if params else 0j
        degree = m.degree(a)
        try:
            free = len(m.free_critical_points(a))
        except (ValueError, RuntimeError):
            free = None
    return Built(expr, params, degree, free, list(info["warnings"]))


def map_for(c: BuilderChoice, name: str | None = None, notes: str = "") -> cdx.RationalMap:
    expr = map_expression(c)
    params = cdx.parse_rational_parameters(expr)
    m = cdx.RationalMap.from_expression(expr, "a" if params else "", {}, name or family_name(c))
    if notes:
        m.notes = notes
    return m


# ---- the four builder presets -------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    choice: BuilderChoice
    notes: str
    parameter_view: tuple[complex, float] | None = None   # parameter-plane framing (center, half-width)


PRESETS: list[Preset] = [
    Preset(BuilderChoice(poly="legendre", degree=5, map_kind="newton"),
           "Family builder: Newton's method z − p/p′ on the Legendre polynomial P₅. Its five "
           "(real) roots are superattracting fixed points; no parameter, so the Library shows "
           "its basins."),
    Preset(BuilderChoice(poly="hermite", degree=4, map_kind="relaxed_newton"),
           "Family builder: relaxed Newton z − a·p/p′ on the Hermite polynomial H₄. a = 1 is "
           "Newton's method; the parameter plane shows where the relaxed iteration still "
           "converges to H₄'s roots.",
           (complex(1.0, 0.0), 1.25)),
    Preset(BuilderChoice(poly="chebyshev_t", degree=6, map_kind="newton"),
           "Family builder: Newton's method on the Chebyshev polynomial T₆ (roots cos((2k−1)π/12))."),
    Preset(BuilderChoice(poly="laguerre", degree=4, map_kind="halley"),
           "Family builder: Halley's method z − 2pp′/(2p′² − pp″) on the Laguerre polynomial L₄ "
           "(degree 2·4 − 1 = 7)."),
]


def preset_names() -> list[str]:
    return [family_name(p.choice) for p in PRESETS]


def preset_maps() -> list[cdx.RationalMap]:
    return [map_for(p.choice, notes=p.notes) for p in PRESETS]


def preset_parameter_views() -> dict[str, tuple[complex, float]]:
    return {family_name(p.choice): p.parameter_view for p in PRESETS if p.parameter_view is not None}

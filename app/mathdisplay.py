"""app/mathdisplay.py -- typeset display of rational-map formulas.

Replaces the old regex "pretty_formula" with a small math typesetter: a
Python Pratt parser for the SAME grammar as cdx's rational_parser.cpp
(including the special polynomials and d(...)), a display AST built from
the text the user TYPED (not the expanded canonical P/Q -- typed structure is
what people recognise), and TeX-like boxes painted with QPainter. No
matplotlib, sympy or MathJax.

PARSER PARITY. The parser mirrors cdx character for character: the same
whitespace, strtod-style numbers ("2i" shorthand, hex, exponents), integer-
literal exponents, unary minus binding TIGHTER than ^ (so "-z^2" means
(-z)^2 and is displayed that way -- the preview's job is to show what cdx
computes), identifier rules ('_' only inside a special-polynomial name), the
special-polynomial arity/degree rules, and every rejected function call.
What it cannot see are the ALGEBRAIC rejections cdx makes after expanding
(a literal-zero denominator such as "5/(z-z)", or a family parameter that
turns out to depend on z or divide by a parameter). So display entry points
that take source text (FormulaView.set_source, typeset_source) gate on cdx
itself first: cdx is authoritative, and nothing it rejects is displayed.
app/test_mathdisplay.py checks both directions over every expression in the
C++ and Python test corpora.

LAYOUT. Boxes with width / ascent / descent: Glyph, Space, Row, Frac
(stacked, with a rule on the math axis), Scripts (super/subscript), Fence
(brackets sized to their content: the font's glyph when it is tall enough,
otherwise a drawn path). Conventions: implicit multiplication ("2*(a-1)*z"
-> 2(a - 1)z), a centred dot only before a factor that starts with a digit,
a real minus sign, thin spaces around binary + and -, italic variables
(Unicode math-italic code points in STIX Two Math), Greek parameter names
(alpha -> alpha-glyph, lam -> lambda), upright function names, special
polynomials as H_5(z), L_4^(a)(z), P_3^(alpha,beta)(x), and d(...) as
p'(z) when its argument is a named polynomial of z, else d/dz[...].

COMPACT MODE (map chip, library rows, Info): fitted to a fixed height. A
fraction stacks when the whole formula fits at no less than 70% of the base
size; otherwise fractions fall back to an inline "(...)/(...)" layout. If
it is still too wide, the MIDDLE is elided (both ends kept), and the full
typeset formula is shown in a tooltip.

FONT. STIX Two Math (bundled, OFL) when it registered; otherwise Inter with
Qt's italic, silently.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from app import theme

# =================================================================================================
# Parser
# =================================================================================================


class DisplayParseError(ValueError):
    def __init__(self, message: str, pos: int):
        super().__init__(f"{message} (at character {pos + 1})")
        self.pos = pos


@dataclass(frozen=True)
class Num:
    text: str


@dataclass(frozen=True)
class Imag:
    coef: str | None     # "2" for "2i"; None for a bare i / j
    letter: str = "i"


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Neg:
    x: object


@dataclass(frozen=True)
class Pos:
    x: object


@dataclass(frozen=True)
class Bin:
    op: str
    l: object
    r: object


@dataclass(frozen=True)
class Pow:
    base: object
    exp: int


@dataclass(frozen=True)
class Paren:
    x: object


@dataclass(frozen=True)
class Call:
    name: str            # as typed (alias or long name)
    family: str
    n: int
    params: tuple        # alpha / lam / (alpha, beta) sub-expressions
    x: object


@dataclass(frozen=True)
class Deriv:
    x: object


MAX_DEGREE = 40
WARN_DEGREE = 25

# name -> (family, number of symbol arguments); "laguerre"/"L"/"P" are
# overloaded on the argument count (see _resolve).
_FAMILIES: dict[str, tuple[str, int, str]] = {
    "hermite": ("Hermite", 0, "2 arguments (n, x)"),
    "H": ("Hermite", 0, "2 arguments (n, x)"),
    "hermite_e": ("HermiteE", 0, "2 arguments (n, x)"),
    "He": ("HermiteE", 0, "2 arguments (n, x)"),
    "legendre": ("Legendre", 0, "2 arguments (n, x)"),
    "chebyshev_t": ("ChebyshevT", 0, "2 arguments (n, x)"),
    "T": ("ChebyshevT", 0, "2 arguments (n, x)"),
    "chebyshev_u": ("ChebyshevU", 0, "2 arguments (n, x)"),
    "U": ("ChebyshevU", 0, "2 arguments (n, x)"),
    "gegenbauer": ("Gegenbauer", 1, "3 arguments (n, lam, x)"),
    "jacobi": ("Jacobi", 2, "4 arguments (n, alpha, beta, x)"),
    "bessel_poly": ("Bessel", 0, "2 arguments (n, x)"),
    "bernoulli": ("Bernoulli", 0, "2 arguments (n, x)"),
    "euler_poly": ("Euler", 0, "2 arguments (n, x)"),
    "cyclotomic": ("Cyclotomic", 0, "2 arguments (n, x)"),
}
_OVERLOADED = {"laguerre", "L", "P"}
SPECIAL_NAMES = frozenset(_FAMILIES) | _OVERLOADED

# Standard symbol for each family (upright).
FAMILY_SYMBOL = {
    "Hermite": "H", "HermiteE": "He", "Legendre": "P", "Laguerre": "L", "Gegenbauer": "C",
    "Jacobi": "P", "ChebyshevT": "T", "ChebyshevU": "U", "Bessel": "y", "Bernoulli": "B",
    "Euler": "E", "Cyclotomic": "Φ",
}

_TRANSCENDENTAL = frozenset({"exp", "log", "ln", "sin", "cos", "tan", "asin", "acos", "atan",
                             "sinh", "cosh", "tanh", "sqrt", "conj", "abs", "re", "im", "arg"})

_WS = " \t\n\v\f\r"
# strtod's accepted forms starting at a digit or '.': hex (C99) first, then decimal.
_NUM_RE = re.compile(r"0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?"
                     r"|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _is_digit(c: str) -> bool:
    return "0" <= c <= "9"


def _is_alpha(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def _resolve(name: str, nargs: int) -> tuple[str, int] | str:
    """(family, symbol count), or the arity error message."""
    if name in ("laguerre", "L"):
        if nargs == 2:
            return ("Laguerre", 0)
        if nargs == 3:
            return ("Laguerre", 1)
        return f"{name} expects 2 or 3 arguments (n, x) or (n, alpha, x)"
    if name == "P":
        if nargs == 2:
            return ("Legendre", 0)
        if nargs == 4:
            return ("Jacobi", 2)
        return "P expects 2 arguments (n, x) or 4 (n, alpha, beta, x)"
    family, symbols, sig = _FAMILIES[name]
    if nargs != symbols + 2:
        return f"{name} expects {sig}"
    return (family, symbols)


class _Parser:
    """Pratt parser over characters (cdx scans characters too, so this keeps
    positions and whitespace handling identical). Binding powers: + - 10,
    * / 20; '^' is not an infix operator here but part of the operand rule
    (factor := unary ('^' int)?), applied at most once -- exactly cdx's
    grammar, which is what makes "-z^2" (-z)^2 and "z^2^3" an error."""

    _LBP = {"+": 10, "-": 10, "*": 20, "/": 20}

    def __init__(self, src: str):
        self.s = src
        self.pos = 0

    def skip(self) -> None:
        while self.pos < len(self.s) and self.s[self.pos] in _WS:
            self.pos += 1

    def peek(self) -> str:
        self.skip()
        return self.s[self.pos] if self.pos < len(self.s) else ""

    def accept(self, c: str) -> bool:
        self.skip()
        if self.pos < len(self.s) and self.s[self.pos] == c:
            self.pos += 1
            return True
        return False

    def fail(self, msg: str):
        raise DisplayParseError(msg, self.pos)

    def parse(self):
        node = self.expression(0)
        if self.peek():
            self.fail("unexpected trailing input")
        return node

    def expression(self, rbp: int):
        left = self.unary()
        self.skip()
        if self.accept("^"):
            left = Pow(left, self.int_exponent())
        while True:
            c = self.peek()
            lbp = self._LBP.get(c)
            if lbp is None or lbp <= rbp:
                return left
            self.pos += 1
            left = Bin(c, left, self.expression(lbp))

    def int_exponent(self) -> int:
        self.skip()
        paren = self.accept("(")
        self.skip()
        neg = False
        if self.accept("-"):
            neg = True
        else:
            self.accept("+")
        self.skip()
        start = self.pos
        while self.pos < len(self.s) and _is_digit(self.s[self.pos]):
            self.pos += 1
        if self.pos == start:
            self.fail("exponent must be a literal integer")
        value = int(self.s[start:self.pos])
        if paren:
            self.skip()
            if not self.accept(")"):
                self.fail("expected ')' after exponent")
        return -value if neg else value

    def unary(self):
        self.skip()
        if self.accept("-"):
            return Neg(self.unary())
        if self.accept("+"):
            return Pos(self.unary())
        return self.atom()

    def atom(self):
        self.skip()
        if self.pos >= len(self.s):
            self.fail("unexpected end of expression")
        c = self.s[self.pos]
        if c == "(":
            self.pos += 1
            x = self.expression(0)
            if not self.accept(")"):
                self.fail("expected ')'")
            return Paren(x)
        if _is_digit(c) or c == ".":
            m = _NUM_RE.match(self.s, self.pos)
            if not m:
                self.fail("malformed number")
            self.pos = m.end()
            if self.pos < len(self.s) and self.s[self.pos] in "ij":
                letter = self.s[self.pos]
                self.pos += 1
                return Imag(m.group(), letter)
            return Num(m.group())
        if _is_alpha(c):
            start = self.pos
            while self.pos < len(self.s) and (_is_alpha(self.s[self.pos]) or _is_digit(self.s[self.pos])
                                              or self.s[self.pos] == "_"):
                self.pos += 1
            ident = self.s[start:self.pos]
            if ident == "z":
                return Var("z")
            if ident in ("i", "j"):
                return Imag(None, ident)
            self.skip()
            if self.pos < len(self.s) and self.s[self.pos] == "(":
                if ident == "d":
                    self.pos += 1
                    return self.derivative()
                if ident in SPECIAL_NAMES:
                    self.pos += 1
                    return self.special(ident)
                if ident in _TRANSCENDENTAL:
                    self.fail(f"transcendental function '{ident}' is not supported")
                self.fail(f"unknown function '{ident}'")
            if "_" in ident:
                self.pos = start + ident.index("_")
                self.fail("unexpected character '_'")
            return Var(ident)
        self.fail(f"unexpected character '{c}'")

    def derivative(self):
        x = self.expression(0)
        if self.peek() == ",":
            self.fail("d expects 1 argument (an expression in z)")
        if not self.accept(")"):
            self.fail("expected ')' after d(...)")
        return Deriv(x)

    def special(self, name: str):
        self.skip()
        dstart = self.pos
        while self.pos < len(self.s) and _is_digit(self.s[self.pos]):
            self.pos += 1
        if self.pos == dstart:
            self.fail(f"{name}: degree must be an integer literal")
        digits = self.s[dstart:self.pos]
        if self.peek() not in (",", ")"):
            self.pos = dstart
            self.fail(f"{name}: degree must be an integer literal")
        n = 1_000_000 if len(digits) > 6 else int(digits)
        args = []
        while self.accept(","):
            args.append(self.expression(0))
        if not self.accept(")"):
            self.fail(f"expected ',' or ')' in {name}(...)")
        resolved = _resolve(name, len(args) + 1)
        if isinstance(resolved, str):
            self.fail(resolved)
        family, nsym = resolved
        if n > MAX_DEGREE:
            self.fail(f"{name}: degree must be at most {MAX_DEGREE}")
        if family == "Cyclotomic" and n < 1:
            self.fail(f"{name}: index must be at least 1")
        return Call(name, family, n, tuple(args[:nsym]), args[-1])


def parse_display(text: str):
    """The display AST of `text` (Python parser only; raises
    DisplayParseError). See the module docstring for what it can't see."""
    return _Parser(text).parse()


def cdx_error(text: str) -> str | None:
    """cdx's own verdict on `text`: None if it parses, else its message."""
    import cdx
    try:
        cdx.parse_rational_parameters(text)
    except ValueError as e:
        return str(e)
    return None


def typeset_source(text: str):
    """The display AST for `text` when cdx accepts it, else raises
    DisplayParseError carrying cdx's message -- the gate every display path
    uses (cdx is authoritative)."""
    err = cdx_error(text)
    if err is not None:
        raise DisplayParseError(err.rsplit(" (at character", 1)[0], 0)
    return parse_display(text)


# =================================================================================================
# Plain-text rendering (search haystacks, accessible names)
# =================================================================================================
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
MINUS = "−"
PRIMES = {1: "′", 2: "″", 3: "‴"}

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "lam": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_NAME_RE = re.compile(r"([A-Za-z]+?)([0-9]*)")


def _split_name(name: str) -> tuple[str, str, bool]:
    """(symbol, subscript digits, italic) for a variable name: a single
    letter or a Greek name -> italic symbol with its trailing digits as a
    subscript; anything else -> the whole name, upright (so "ab" never reads
    as a*b, which displays as "ab" in italic)."""
    m = _NAME_RE.fullmatch(name)
    if m:
        letters, digits = m.group(1), m.group(2)
        if letters in GREEK:
            return GREEK[letters], digits, GREEK[letters].islower()
        if len(letters) == 1:
            return letters, digits, True
    return name, "", False


def _num_text(text: str) -> str:
    if text.startswith("."):
        text = "0" + text
    if text.endswith(".") and not text.lower().startswith("0x"):
        text = text[:-1]
    return text


def plain_text(node) -> str:
    """A linear Unicode rendering (H₅(z), L₄^(a)(z), z², 2(a − 1)z)."""
    def go(n) -> str:
        if isinstance(n, Num):
            return _num_text(n.text)
        if isinstance(n, Imag):
            return (_num_text(n.coef) if n.coef else "") + n.letter
        if isinstance(n, Var):
            sym, sub, _ = _split_name(n.name)
            return sym + sub.translate(_SUB)
        if isinstance(n, (Neg, Pos)):
            sign = MINUS if isinstance(n, Neg) else "+"
            return sign + (f"({go(n.x)})" if isinstance(n.x, (Neg, Pos)) else go(n.x))
        if isinstance(n, Paren):
            return "(" + go(n.x) + ")"
        if isinstance(n, Pow):
            base = go(n.base)
            if isinstance(n.base, (Neg, Pos, Bin)) or (isinstance(n.base, Imag) and n.base.coef):
                base = f"({base})"
            return base + str(n.exp).translate(_SUP)
        if isinstance(n, Bin):
            r = f"({go(n.r)})" if isinstance(n.r, (Neg, Pos)) else go(n.r)
            if n.op in "+-":
                return f"{go(n.l)} {'+' if n.op == '+' else MINUS} {r}"
            if n.op == "*":
                dot = "·" if (not isinstance(n.r, (Neg, Pos)) and _starts_with_digit(n.r)) else ""
                return go(n.l) + dot + r
            def side(m):
                return go(m) if _is_inline_atom(m) else f"({go(m)})"
            return f"{side(n.l)}/{side(n.r)}"
        if isinstance(n, Call):
            sup = ""
            if n.params:
                sup = "^(" + ",".join(go(p) for p in n.params) + ")"
            return f"{FAMILY_SYMBOL[n.family]}{str(n.n).translate(_SUB)}{sup}({go(n.x)})"
        if isinstance(n, Deriv):
            pf = _prime_form(n)
            if pf is not None:
                call, k = pf
                sup = "^(" + ",".join(go(p) for p in call.params) + ")" if call.params else ""
                return f"{FAMILY_SYMBOL[call.family]}{str(call.n).translate(_SUB)}{sup}{PRIMES[k]}({go(call.x)})"
            k, inner = 0, n
            while isinstance(inner, Deriv):
                k, inner = k + 1, _strip_paren(inner.x)
            power = str(k).translate(_SUP) if k > 1 else ""
            return f"d{power}/dz{power}[{go(inner)}]"
        return "?"
    return go(node)


def _starts_with_digit(n) -> bool:
    if isinstance(n, (Num,)):
        return True
    if isinstance(n, Imag):
        return n.coef is not None
    if isinstance(n, Pow):
        return _starts_with_digit(n.base)
    if isinstance(n, Bin) and n.op in "*+-":
        return _starts_with_digit(n.l)
    return False


# =================================================================================================
# Fonts
# =================================================================================================
FONT_MATH = theme.FONT_MATH


@lru_cache(maxsize=1)
def math_font_available() -> bool:
    return FONT_MATH in theme.load_fonts()


def _math_italic(text: str) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append("ℎ" if ch == "h" else chr(0x1D44E + ord(ch) - 97))
        elif "A" <= ch <= "Z":
            out.append(chr(0x1D434 + ord(ch) - 65))
        elif "α" <= ch <= "ω":
            out.append(chr(0x1D6FC + ord(ch) - 0x3B1))
        else:
            out.append(ch)
    return "".join(out)


class _Fonts:
    """Fonts and metrics per pixel size. `stix` False -> Inter fallback."""

    def __init__(self):
        self.stix = math_font_available()
        self._cache: dict[tuple[int, bool], tuple[QFont, QFontMetricsF]] = {}

    def get(self, px: int, italic: bool) -> tuple[QFont, QFontMetricsF]:
        px = max(5, int(px))
        key = (px, italic and not self.stix)
        hit = self._cache.get(key)
        if hit is None:
            font = QFont()
            font.setFamilies([FONT_MATH, theme.FONT_UI] if self.stix else [theme.FONT_UI])
            font.setPixelSize(px)
            font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
            font.setKerning(True)
            if key[1]:
                font.setItalic(True)
            hit = (font, QFontMetricsF(font))
            self._cache[key] = hit
        return hit

    def italic_text(self, text: str) -> str:
        return _math_italic(text) if self.stix else text

    @property
    def dot(self) -> str:
        return "⋅" if self.stix else "·"

    @property
    def ellipsis(self) -> str:
        return "⋯" if self.stix else "…"


_FONTS: _Fonts | None = None


def _fonts() -> _Fonts:
    global _FONTS
    if _FONTS is None:
        _FONTS = _Fonts()
    return _FONTS


# =================================================================================================
# Boxes
# =================================================================================================
class Box:
    w: float = 0.0
    asc: float = 0.0
    desc: float = 0.0

    @property
    def h(self) -> float:
        return self.asc + self.desc

    def paint(self, p: QPainter, x: float, y: float) -> None:   # y = baseline
        raise NotImplementedError

    def items(self) -> list["Box"]:
        """Flattened top-level sequence for middle elision."""
        return [self]

    def key(self) -> str:
        return type(self).__name__


class Glyph(Box):
    def __init__(self, text: str, px: float, italic: bool = False):
        f = _fonts()
        self.px = px
        self.text = f.italic_text(text) if italic else text
        self.source_text = text
        self.font, fm = f.get(px, italic)
        r = fm.tightBoundingRect(self.text)
        self.w = fm.horizontalAdvance(self.text)
        self.asc = max(0.0, -r.top())
        self.desc = max(0.0, r.bottom())
        self.italic = italic

    def paint(self, p, x, y):
        p.setFont(self.font)
        p.drawText(QPointF(x, y), self.text)

    def key(self) -> str:
        return "G:" + self.source_text


class Space(Box):
    def __init__(self, w: float):
        self.w = w

    def paint(self, p, x, y):
        pass

    def key(self) -> str:
        return "S"


class Row(Box):
    def __init__(self, children: list[Box]):
        self.children = [c for c in children if c is not None]
        self.w = sum(c.w for c in self.children)
        self.asc = max((c.asc for c in self.children), default=0.0)
        self.desc = max((c.desc for c in self.children), default=0.0)

    def paint(self, p, x, y):
        for c in self.children:
            c.paint(p, x, y)
            x += c.w

    def items(self):
        out = []
        for c in self.children:
            out.extend(c.items())
        return out


def _axis(px: float) -> float:
    f = _fonts()
    _, fm = f.get(px, False)
    r = fm.tightBoundingRect(MINUS)
    return -(r.top() + r.bottom()) / 2.0


class Frac(Box):
    def __init__(self, num: Box, den: Box, px: float, compact: bool):
        self.num, self.den, self.px = num, den, px
        self.t = max(1.0, 0.06 * px)
        self.gap = (0.10 if compact else 0.14) * px
        self.pad = 0.12 * px
        self.axis = _axis(px)
        self.w = max(num.w, den.w) + 2 * self.pad
        self.asc = self.axis + self.t / 2 + self.gap + num.desc + num.asc
        self.desc = max(0.0, -self.axis + self.t / 2 + self.gap + den.asc + den.desc)

    def paint(self, p, x, y):
        nx = x + (self.w - self.num.w) / 2
        dx = x + (self.w - self.den.w) / 2
        self.num.paint(p, nx, y - (self.axis + self.t / 2 + self.gap + self.num.desc))
        self.den.paint(p, dx, y + (-self.axis + self.t / 2 + self.gap + self.den.asc))
        rule_y = y - self.axis
        color = _current_color(p)
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRect(QRectF(x + self.pad * 0.4, rule_y - self.t / 2, self.w - self.pad * 0.8, self.t))
        p.restore()


def _current_color(p: QPainter) -> QColor:
    return p.pen().color()


class Scripts(Box):
    def __init__(self, base: Box, px: float, sup: Box | None = None, sub: Box | None = None,
                 sup_shift: float | None = None):
        self.base, self.sup, self.sub, self.px = base, sup, sub, px
        self.sup_shift = sup_shift
        self._layout()

    def _layout(self):
        base, sup, sub, px = self.base, self.sup, self.sub, self.px
        tall = base.h > 1.05 * px
        self.kern = 0.06 * px if (isinstance(base, Glyph) and base.italic) else 0.02 * px
        self.u = 0.0
        self.d = 0.0
        if sup is not None:
            self.u = self.sup_shift if self.sup_shift is not None else \
                max(0.40 * px, (base.asc - 0.30 * px) if tall else 0.0, sup.desc + 0.12 * px)
        if sub is not None:
            self.d = max(0.17 * px, (base.desc + 0.05 * px) if tall else 0.0, sub.asc - 0.45 * px)
        if sup is not None and sub is not None and self.sup_shift is None:
            gap = (self.u - sup.desc) - (sub.asc - self.d)
            if gap < 0.08 * px:
                self.d += 0.08 * px - gap
        scripts_w = max((sup.w + self.kern) if sup else 0.0, sub.w if sub else 0.0)
        self.w = base.w + scripts_w + 0.03 * px
        self.asc = max(base.asc, (self.u + sup.asc) if sup else 0.0)
        self.desc = max(base.desc, (self.d + sub.desc) if sub else 0.0)

    def with_sup(self, sup: Box) -> "Scripts":
        return Scripts(self.base, self.px, sup=sup, sub=self.sub, sup_shift=None)

    def paint(self, p, x, y):
        self.base.paint(p, x, y)
        sx = x + self.base.w
        if self.sup is not None:
            self.sup.paint(p, sx + self.kern, y - self.u)
        if self.sub is not None:
            self.sub.paint(p, sx, y + self.d)

    def items(self):
        # (...)^2 splits like its bracket, the script riding on ")" at the
        # same height -- so middle elision can cut inside it.
        if isinstance(self.base, Fence) and not self.base.drawn and self.sub is None:
            close = Scripts(self.base.close_g, self.px, sup=self.sup, sup_shift=self.u)
            return [self.base.open_g] + self.base.content.items() + [close]
        return [self]

    def key(self) -> str:
        return "Sc:" + self.base.key()


class Fence(Box):
    """Content between brackets sized to it: the font's own glyphs when they
    already cover it, otherwise drawn paths."""

    def __init__(self, content: Box, px: float, kind: str = "()", slack: float = 1.0):
        self.content, self.px, self.kind = content, px, kind
        self.open_g = Glyph(kind[0], px)
        self.close_g = Glyph(kind[1], px)
        # slack > 1 (compact mode) lets the plain glyph cover content that
        # pokes a little past it (a superscript), as inline math does.
        fits = (content.asc <= self.open_g.asc * 1.02 * slack + 0.02 * px
                and content.desc <= self.open_g.desc * 1.05 * slack + 0.02 * px)
        self.drawn = not fits
        if not self.drawn:
            self.bw_open, self.bw_close = self.open_g.w, self.close_g.w
            self.asc = max(content.asc, self.open_g.asc)
            self.desc = max(content.desc, self.open_g.desc)
        else:
            margin = 0.10 * px
            self.top = content.asc + margin
            self.bottom = content.desc + margin
            height = self.top + self.bottom
            bw = (0.30 + 0.06 * min(height / px, 4.0)) * px if kind == "()" else 0.32 * px
            self.bw_open = self.bw_close = bw
            self.asc, self.desc = self.top, self.bottom
        self.w = self.bw_open + content.w + self.bw_close

    def _paint_drawn(self, p: QPainter, x: float, y: float, left: bool) -> None:
        top, bottom = y - self.top, y + self.bottom
        h = bottom - top
        bw = self.bw_open
        px = self.px
        p.save()
        color = _current_color(p)
        if self.kind == "()":
            t = max(0.9, 0.075 * px)
            path = QPainterPath()
            inset = 0.30 * bw
            if left:
                tip, belly = x + bw - inset, x + inset * 0.6
                path.moveTo(tip, top)
                path.cubicTo(belly, top + h * 0.22, belly, bottom - h * 0.22, tip, bottom)
                path.cubicTo(belly + t, bottom - h * 0.24, belly + t, top + h * 0.24, tip, top)
            else:
                tip, belly = x + inset, x + bw - inset * 0.6
                path.moveTo(tip, top)
                path.cubicTo(belly, top + h * 0.22, belly, bottom - h * 0.22, tip, bottom)
                path.cubicTo(belly - t, bottom - h * 0.24, belly - t, top + h * 0.24, tip, top)
            p.setPen(QPen(color, max(0.6, 0.02 * px)))
            p.setBrush(color)
            p.drawPath(path)
        else:
            t = max(1.0, 0.06 * px)
            pen = QPen(color, t)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if left:
                a, b = x + bw * 0.75, x + bw * 0.35
            else:
                a, b = x + bw * 0.25, x + bw * 0.65
            path = QPainterPath()
            path.moveTo(a, top + t / 2)
            path.lineTo(b, top + t / 2)
            path.lineTo(b, bottom - t / 2)
            path.lineTo(a, bottom - t / 2)
            p.drawPath(path)
        p.restore()

    def paint(self, p, x, y):
        if self.drawn:
            self._paint_drawn(p, x, y, True)
            self.content.paint(p, x + self.bw_open, y)
            self._paint_drawn(p, x + self.bw_open + self.content.w, y, False)
        else:
            self.open_g.paint(p, x, y)
            self.content.paint(p, x + self.bw_open, y)
            self.close_g.paint(p, x + self.bw_open + self.content.w, y)

    def items(self):
        if self.drawn:
            return [self]
        return [self.open_g] + self.content.items() + [self.close_g]

    def key(self) -> str:
        return "F" + self.kind


class Raised(Box):
    """A box shifted vertically by dy (positive = up)."""

    def __init__(self, inner: Box, dy: float):
        self.inner, self.dy = inner, dy
        self.w = inner.w
        self.asc = max(0.0, inner.asc + dy)
        self.desc = max(0.0, inner.desc - dy)

    def paint(self, p, x, y):
        self.inner.paint(p, x, y - self.dy)

    def key(self) -> str:
        return "R:" + self.inner.key()


class Scaled(Box):
    """A box shrunk uniformly (last-resort fit)."""

    def __init__(self, inner: Box, s: float):
        self.inner, self.s = inner, s
        self.w, self.asc, self.desc = inner.w * s, inner.asc * s, inner.desc * s

    def paint(self, p, x, y):
        p.save()
        p.translate(x, y)
        p.scale(self.s, self.s)
        self.inner.paint(p, 0.0, 0.0)
        p.restore()


# =================================================================================================
# AST -> boxes
# =================================================================================================
@dataclass
class _Style:
    px: float
    inline_frac: bool = False   # compact fallback: (num)/(den) on one line
    compact: bool = False
    frac_depth: int = 0


def _script_px(px: float) -> float:
    return max(6.0, px * 0.70)


def _strip_paren(n):
    while isinstance(n, Paren):
        n = n.x
    return n


def _is_inline_atom(n) -> bool:
    return isinstance(n, (Num, Var, Paren, Call)) or (isinstance(n, Imag) and n.coef is None) \
        or (isinstance(n, Pow) and _is_inline_atom(n.base)) \
        or (isinstance(n, Deriv) and _prime_form(n) is not None)


def _prime_form(n):
    """(call, k) when d^k(...) is d applied k <= 3 times to a named
    polynomial OF z itself -- shown as p'(z), p''(z), p'''(z). With any other
    argument (e.g. H5(2z-1)) the chain rule makes p'(x) wrong, so those use
    the d/dz[...] form."""
    k = 0
    while isinstance(n, Deriv):
        k += 1
        n = _strip_paren(n.x)
    if isinstance(n, Call) and _strip_paren(n.x) == Var("z") and 1 <= k <= 3:
        return n, k
    return None


class _Builder:
    def __init__(self, style: _Style):
        self.st = style

    def thin(self, px) -> Space:
        return Space(px / 6.0)

    def build(self, n, px: float, frac_depth: int = 0) -> Box:
        b = self._build(n, px, frac_depth)
        return b

    def _var(self, name: str, px: float) -> Box:
        sym, sub, italic = _split_name(name)
        g = Glyph(sym, px, italic=italic)
        if sub:
            return Scripts(g, px, sub=Glyph(sub, _script_px(px)))
        return g

    def _number(self, text: str, px: float) -> Box:
        text = _num_text(text)
        m = re.fullmatch(r"([0-9.]+)[eE]([+-]?)([0-9]+)", text)
        if m:
            mant, sign, exp = m.groups()
            exp_txt = (MINUS if sign == "-" else "") + exp.lstrip("0") or "0"
            ten = Scripts(Glyph("10", px), px, sup=Glyph(exp_txt, _script_px(px)))
            if mant in ("1", "1.0"):
                return ten
            return Row([Glyph(mant, px), Space(px / 12), Glyph("×", px), Space(px / 12), ten])
        return Glyph(text, px)

    def _wrap(self, n, px, frac_depth, kind="()") -> Box:
        return Fence(self._build(n, px, frac_depth), px, kind, 1.25 if self.st.inline_frac else 1.0)

    def _build(self, n, px: float, fd: int) -> Box:
        st = self.st
        if isinstance(n, Num):
            return self._number(n.text, px)
        if isinstance(n, Imag):
            unit = Glyph(n.letter, px)
            if n.coef is None:
                return unit
            return Row([self._number(n.coef, px), unit])
        if isinstance(n, Var):
            return self._var(n.name, px)
        if isinstance(n, Paren):
            return self._wrap(n.x, px, fd)
        if isinstance(n, (Neg, Pos)):
            sign = Glyph(MINUS if isinstance(n, Neg) else "+", px)
            inner = n.x
            body = self._wrap(inner, px, fd) if isinstance(inner, (Neg, Pos)) else self._build(inner, px, fd)
            return Row([sign, Space(px / 30), body])
        if isinstance(n, Pow):
            base = n.base
            needs = isinstance(base, (Neg, Pos, Bin)) or (isinstance(base, Imag) and base.coef is not None) \
                or (isinstance(base, Num) and re.search(r"[eE]", base.text) and not base.text.lower().startswith("0x")) \
                or (isinstance(base, Deriv) and _prime_form(base) is None)
            bb = self._wrap(base, px, fd) if needs else self._build(base, px, fd)
            exp = Glyph(str(n.exp).replace("-", MINUS), _script_px(px))
            if isinstance(bb, Scripts) and bb.sup is None:
                return bb.with_sup(exp)
            if isinstance(bb, Row) and bb.children and isinstance(bb.children[-1], Fence):
                # A function application (H5(z), p'(z)): the exponent belongs
                # on its closing bracket, not above the symbol's own scripts.
                return Row(bb.children[:-1] + [Scripts(bb.children[-1], px, sup=exp)])
            return Scripts(bb, px, sup=exp)
        if isinstance(n, Bin):
            if n.op in "+-":
                l = self._build(n.l, px, fd)
                r = self._wrap(n.r, px, fd) if isinstance(n.r, (Neg, Pos)) else self._build(n.r, px, fd)
                op = Glyph("+" if n.op == "+" else MINUS, px)
                return Row([l, self.thin(px), op, self.thin(px), r])
            if n.op == "*":
                l = self._build(n.l, px, fd)
                signed = isinstance(n.r, (Neg, Pos))
                r = self._wrap(n.r, px, fd) if signed else self._build(n.r, px, fd)
                parts = [l]
                if not signed and _starts_with_digit(n.r):
                    parts += [Space(px / 18), Glyph(_fonts().dot, px), Space(px / 18)]
                elif isinstance(l, Frac) and isinstance(r, Frac):
                    parts.append(Space(px / 10))
                parts.append(r)
                return Row(parts)
            # division
            if st.inline_frac:
                def side(m):
                    return self._build(m, px, fd) if _is_inline_atom(m) else self._wrap(m, px, fd)
                return Row([side(n.l), Space(px / 24), Glyph("/", px), Space(px / 24), side(n.r)])
            inner_px = px if fd == 0 or st.compact else max(px * 0.85, st.px * 0.6)
            num = self._build(_strip_paren(n.l), inner_px, fd + 1)
            den = self._build(_strip_paren(n.r), inner_px, fd + 1)
            return Frac(num, den, px, st.compact)
        if isinstance(n, Call):
            return self._call(n, px, fd, primes=0)
        if isinstance(n, Deriv):
            pf = _prime_form(n)
            if pf is not None:
                call, k = pf
                return self._call(call, px, fd, primes=k)
            k = 0
            inner = n
            while isinstance(inner, Deriv):
                k += 1
                inner = _strip_paren(inner.x)
            sp = _script_px(px)
            if st.inline_frac:
                op_parts = [Glyph("d", px)]
                if k > 1:
                    op_parts = [Scripts(Glyph("d", px), px, sup=Glyph(str(k), sp))]
                op_parts += [Glyph("/", px), Glyph("d", px)]
                zbox = Glyph("z", px, italic=True)
                op_parts.append(Scripts(zbox, px, sup=Glyph(str(k), sp)) if k > 1 else zbox)
                op = Row(op_parts)
            else:
                dnum = Glyph("d", px) if k == 1 else Scripts(Glyph("d", px), px, sup=Glyph(str(k), sp))
                zbox = Glyph("z", px, italic=True)
                dden = Row([Glyph("d", px), Scripts(zbox, px, sup=Glyph(str(k), sp)) if k > 1 else zbox])
                op = Frac(dnum, dden, px, st.compact)
            return Row([op, Space(px / 18), self._wrap(inner, px, fd, "[]")])
        raise TypeError(f"unknown display node {n!r}")

    def _call(self, n: Call, px: float, fd: int, primes: int) -> Box:
        sp = _script_px(px)
        sym = Glyph(FAMILY_SYMBOL[n.family], px)
        sub = Glyph(str(n.n), sp)
        sup_parts: list[Box] = []
        if n.params:
            inner: list[Box] = []
            for i, prm in enumerate(n.params):
                if i:
                    inner += [Glyph(",", sp), Space(sp / 8)]
                inner.append(self._build(prm, sp, fd))
            sup_parts.append(Fence(Row(inner), sp))
        prime_text = PRIMES.get(primes, "")
        shift = None
        if primes and n.params:
            # The prime glyph is already drawn at superscript height: after a
            # raised (alpha) it is lowered back toward that row's baseline.
            sup_parts.append(Raised(Glyph(prime_text, px), -0.40 * px))
        elif primes:
            sup_parts.append(Glyph(prime_text, px))
            shift = 0.0   # ...and needs no extra raise on its own
        sup = Row(sup_parts) if sup_parts else None
        head = Scripts(sym, px, sup=sup, sub=sub, sup_shift=shift)
        return Row([head, self._wrap(n.x, px, fd)])


def layout_display(node, px: float) -> Box:
    """Full-size typeset layout (Equation-tab preview, tooltips)."""
    return _Builder(_Style(px)).build(node, px)


def _contains_div(n) -> bool:
    if isinstance(n, Bin):
        return n.op == "/" or _contains_div(n.l) or _contains_div(n.r)
    for attr in ("x", "base", "l", "r"):
        if hasattr(n, attr) and _contains_div(getattr(n, attr)):
            return True
    if isinstance(n, Call):
        return any(_contains_div(p) for p in n.params)
    return isinstance(n, Deriv) and _prime_form(n) is None


@dataclass
class Compact:
    box: Box
    stacked: bool
    elided: bool
    px: float


STACK_MIN_SCALE = 0.70


def layout_compact(node, px: float, max_w: float, max_h: float) -> Compact:
    """Fits `node` into max_w x max_h: stacked fractions if they fit at >= 70%
    of `px`, else inline "(...)/(...)"; then the middle is elided if it is
    still too wide; a uniform shrink is the last resort, so the result
    NEVER exceeds max_w x max_h."""
    # One pixel of slack on every side: antialiased ink (and an italic
    # glyph's overhang past its advance) must stay inside the rect too.
    max_w, max_h = max(1.0, max_w - 2.0 - 0.08 * px), max(1.0, max_h - 2.0)
    px0 = float(px)
    chosen: Box | None = None
    stacked = False
    if _contains_div(node):
        size = px0
        while size >= px0 * STACK_MIN_SCALE - 1e-9:
            b = _Builder(_Style(size, compact=True)).build(node, size)
            if b.h <= max_h:
                chosen, stacked = b, True
                break
            size -= 1.0
    if chosen is None:
        size = px0
        while True:
            b = _Builder(_Style(size, inline_frac=True, compact=True)).build(node, size)
            if b.h <= max_h or size <= 6:
                chosen = b
                break
            size -= 1.0
    elided = False
    if chosen.w > max_w:
        chosen = elide_middle(chosen, max_w, size)
        elided = True
    if chosen.w > max_w + 1e-6 or chosen.h > max_h + 1e-6:
        s = min(max_w / max(chosen.w, 1e-9), max_h / max(chosen.h, 1e-9), 1.0)
        chosen = Scaled(chosen, s)
    return Compact(chosen, stacked, elided, size)


def elide_middle(box: Box, max_w: float, px: float) -> Box:
    """Keeps the first and last items of the top-level sequence and as many
    as fit from each end, with an ellipsis between. A stacked fraction is
    elided inside its numerator and denominator instead."""
    if isinstance(box, Frac):
        inner = max_w - 2 * box.pad
        num = elide_middle(box.num, inner, box.px) if box.num.w > inner else box.num
        den = elide_middle(box.den, inner, box.px) if box.den.w > inner else box.den
        return Frac(num, den, box.px, True)
    if box.w <= max_w:
        return box
    items = box.items()
    # A stacked fraction that dominates the row (z - p/p'): elide inside it,
    # keeping the rest of the row as is.
    widest = max(items, key=lambda b: b.w)
    rest = box.w - widest.w
    if isinstance(widest, Frac) and max_w - rest >= 4 * px:
        inner = elide_middle(widest, max_w - rest, px)
        if inner.w <= max_w - rest + 1e-6:
            return Row([inner if it is widest else it for it in items])
    if len(items) < 3:
        return Scaled(box, max_w / box.w)
    ell = Row([Space(px / 8), Glyph(_fonts().ellipsis, px), Space(px / 8)])
    budget = max_w - ell.w
    first, last = items[0], items[-1]
    if first.w + last.w > budget:
        return Scaled(box, max_w / box.w)
    # Balanced: each end gets half the budget, then whatever is left over is
    # filled from either end (left first).
    i, j = 1, len(items) - 2
    left_w, right_w = first.w, last.w
    half = budget / 2.0
    while i <= j and left_w + items[i].w <= half:
        left_w += items[i].w
        i += 1
    while j >= i and right_w + items[j].w <= half:
        right_w += items[j].w
        j -= 1
    progressed = True
    while i <= j and progressed:
        progressed = False
        if left_w + right_w + items[i].w <= budget:
            left_w += items[i].w
            i += 1
            progressed = True
        elif j >= i and left_w + right_w + items[j].w <= budget:
            right_w += items[j].w
            j -= 1
            progressed = True
    if i > j:   # everything fit after all
        return Row(items)
    left, right = items[:i], items[j + 1:]
    return Row(left + [ell] + right)


# =================================================================================================
# Painting helpers
# =================================================================================================
def paint_box(p: QPainter, box: Box, rect: QRectF, color: QColor, align_left: bool = True,
              clip: bool = True) -> None:
    p.save()
    if clip:
        p.setClipRect(rect)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    p.setPen(color)
    x = rect.left() + 1.0 if align_left else rect.left() + (rect.width() - box.w) / 2
    y = rect.top() + (rect.height() - box.h) / 2 + box.asc
    box.paint(p, x, y)
    p.restore()


def render_image(source_or_node, *, px: float = 16.0, compact_size: tuple[float, float] | None = None,
                 color: QColor | None = None, dpr: float = 1.0, margin: float = 2.0) -> QImage:
    """A transparent image of a formula: full size, or compact into
    compact_size = (max_w, max_h). Raises DisplayParseError for bad source."""
    node = typeset_source(source_or_node) if isinstance(source_or_node, str) else source_or_node
    if compact_size is None:
        box = layout_display(node, px)
        w, h = box.w + 2 * margin, box.h + 2 * margin
    else:
        box = layout_compact(node, px, compact_size[0], compact_size[1]).box
        w, h = compact_size
    img = QImage(max(1, math.ceil(w * dpr)), max(1, math.ceil(h * dpr)), QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    if compact_size is None:
        paint_box(p, box, QRectF(margin, margin, box.w, box.h), color or theme.color("text"))
    else:
        paint_box(p, box, QRectF(0, 0, w, h), color or theme.color("text"))
    p.end()
    return img


# =================================================================================================
# Widgets
# =================================================================================================
class FormulaView(QWidget):
    """One typeset formula.

    compact=True: fitted to the widget's own height and middle-elided to its
    width (map chip, library rows); the full formula shows in a tooltip.
    compact=False: full size; sizeHint is the natural size (put it in a
    scroll area when it may be wide).

    set_source(text) validates with cdx first. On failure the last good
    formula stays, greyed out, and set_source returns False."""

    def __init__(self, parent: QWidget | None = None, *, compact: bool = False, px: float = 16.0,
                 height: int | None = None, max_natural_width: int | None = None,
                 color_role: str = "text", tooltip: bool = True):
        super().__init__(parent)
        self._compact = compact
        self._px = float(px)
        self._height = height
        self._max_natural_width = max_natural_width
        self._color_role = color_role
        self._tooltip = tooltip
        self._source = ""
        self._node = None
        self._greyed = False
        self._empty_text = ""
        self._cache_key = None
        self._cache: Compact | None = None
        self._natural_key = None
        self._natural_box: Box | None = None
        self._caption = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if compact:
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            if height is not None:
                self.setFixedHeight(height)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    # ---- state ------------------------------------------------------------------
    def set_source(self, text: str) -> bool:
        if not text.strip():
            self._source, self._node, self._greyed = text, None, False
            self._changed()
            return True
        try:
            node = typeset_source(text)
        except DisplayParseError:
            self._greyed = self._node is not None
            self._changed()
            return False
        self._source, self._node, self._greyed = text, node, False
        self._changed()
        return True

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self._changed()

    def set_caption(self, text: str) -> None:
        """Extra lines under the formula in the tooltip."""
        self._caption = text

    def source(self) -> str:
        return self._source

    def node(self):
        return self._node

    def is_greyed(self) -> bool:
        return self._greyed

    def plain_text(self) -> str:
        return plain_text(self._node) if self._node is not None else self._empty_text

    def _changed(self) -> None:
        self._cache_key = None
        self._natural_key = None
        self.updateGeometry()
        self.update()

    # ---- layout -----------------------------------------------------------------
    def _natural(self) -> Box | None:
        if self._node is None:
            return None
        key = (self._source, self._px)
        if self._natural_key != key:
            self._natural_box = layout_display(self._node, self._px)
            self._natural_key = key
        return self._natural_box

    def layout_for(self, w: float, h: float) -> Compact | None:
        if self._node is None:
            return None
        key = (self._source, self._px, round(w, 2), round(h, 2))
        if self._cache_key != key:
            self._cache = layout_compact(self._node, self._px, w, h)
            self._cache_key = key
        return self._cache

    def is_elided(self) -> bool:
        if not self._compact or self._node is None:
            return False
        c = self.layout_for(self.width(), self.height())
        return bool(c and c.elided)

    def sizeHint(self) -> QSize:   # noqa: N802
        if self._compact:
            h = self._height or round(self._px * 1.9)
            if self._node is None:
                return QSize(round(self._px * 6), h)
            c = layout_compact(self._node, self._px, 1e9, h)
            w = math.ceil(c.box.w) + 2
            if self._max_natural_width:
                w = min(w, self._max_natural_width)
            return QSize(w, h)
        box = self._natural()
        if box is None:
            return QSize(round(self._px * 6), round(self._px * 1.6))
        return QSize(math.ceil(box.w) + 6, math.ceil(box.h) + 8)

    def minimumSizeHint(self) -> QSize:   # noqa: N802
        if self._compact:
            return QSize(round(self._px * 3), self.sizeHint().height())
        return self.sizeHint()

    # ---- painting -----------------------------------------------------------------
    def _color(self) -> QColor:
        c = theme.color(self._color_role)
        if self._greyed:
            c = QColor(theme.color("muted"))
            c.setAlphaF(0.55)
        return c

    def paintEvent(self, event) -> None:   # noqa: N802
        p = QPainter(self)
        rect = QRectF(self.rect())
        if self._node is None:
            if self._empty_text:
                p.setPen(theme.color("muted"))
                p.setFont(theme.ui_font())
                p.drawText(rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                           self._empty_text)
            p.end()
            return
        if self._compact:
            c = self.layout_for(rect.width(), rect.height())
            paint_box(p, c.box, rect, self._color())
        else:
            box = self._natural()
            paint_box(p, box, rect.adjusted(3, 0, 0, 0), self._color())
        p.end()

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip and self._tooltip and self._node is not None:
            FormulaToolTip.show_for(self, e.globalPos(), self._source, self._caption)
            return True
        if e.type() == QEvent.Type.Leave:
            FormulaToolTip.hide_for(self)
        return super().event(e)


class FormulaToolTip(QFrame):
    """A tooltip-style popup with the full typeset formula (Qt's own rich-
    text tooltips cannot show a painted formula)."""

    _instance: "FormulaToolTip | None" = None

    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("FormulaToolTip")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.view = FormulaView(self, compact=False, px=18, tooltip=False)
        self.caption = QLabel(self)
        self.caption.setProperty("role", "caption")
        layout.addWidget(self.view)
        layout.addWidget(self.caption)
        self.owner: QWidget | None = None
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    @classmethod
    def instance(cls) -> "FormulaToolTip":
        if cls._instance is None:
            cls._instance = FormulaToolTip()
        return cls._instance

    @classmethod
    def show_for(cls, owner: QWidget, global_pos: QPoint, source: str, caption: str = "") -> None:
        tip = cls.instance()
        tip.owner = owner
        tip.view.set_source(source)
        tip.caption.setText(caption)
        tip.caption.setVisible(bool(caption))
        tip.adjustSize()
        tip.move(global_pos + QPoint(12, 16))
        tip.show()
        tip.raise_()
        tip._hide_timer.start(12000)

    @classmethod
    def hide_for(cls, owner: QWidget) -> None:
        tip = cls._instance
        if tip is not None and tip.owner is owner:
            tip.hide()

    @classmethod
    def shown_source(cls) -> str | None:
        tip = cls._instance
        return tip.view.source() if tip is not None and tip.isVisible() else None


class LivePreview(QObject):
    """Debounces a text source into a FormulaView: on a parse error the view
    keeps its last good formula, greyed."""

    def __init__(self, view: FormulaView, delay_ms: int = 150, parent: QObject | None = None):
        super().__init__(parent or view)
        self.view = view
        self._pending = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self.flush)

    def text_changed(self, text: str) -> None:
        self._pending = text
        self._timer.start()

    def set_now(self, text: str) -> None:
        self._pending = text
        self._timer.stop()
        self.flush()

    def flush(self) -> None:
        self._timer.stop()
        self.view.set_source(self._pending)

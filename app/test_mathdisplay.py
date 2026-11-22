"""test_mathdisplay.py -- the formula typesetter (app/mathdisplay.py).

  - Parser parity: every expression in the C++ and Python test corpora (plus
    a hand-written list of edge cases) goes to both cdx and the display
    parser. Anything cdx accepts must display; anything cdx rejects must not
    (the display entry points gate on cdx); and the display parser on its
    own may accept a cdx-rejected expression ONLY for the algebraic reasons
    it cannot see by syntax (literal-zero denominator, a family parameter
    depending on z or dividing by a parameter).
  - Display conventions, via the plain-text rendering.
  - Golden images at fixed size for 12 representative formulas, compared
    with a small tolerance (blurred, then a bounded fraction of differing
    pixels) so macOS / Windows / Linux rasterizers all pass.
  - Compact layout never paints outside its rect; middle elision keeps both
    ends; the widgets (chip, Equation preview, Library, Info).

Run (from the repository root):

    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m app.test_mathdisplay
    ... python -m app.test_mathdisplay --regen    # rewrite the golden images
"""

from __future__ import annotations

import codecs
import re
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QHelpEvent, QImage, QPainter
from PySide6.QtWidgets import QApplication

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "app" / "test_data" / "mathdisplay"

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


KIM = "z^3*(2*(a-1)+a^2 +4*(a-1)*z+2*(a-1)*z^2)/((2*(a-1) + 4*(a-1)*z + (a^2+2*a-2)*z^2)^2)"
HALLEY_L4 = ("z - 2*laguerre(4, z)*d(laguerre(4, z))/(2*d(laguerre(4, z))^2 - "
             "laguerre(4, z)*d(d(laguerre(4, z))))")
NEWTON_H5 = "z - hermite(5, z)/d(hermite(5, z))"
GEN_LAGUERRE = "z - laguerre(4, alpha, z)/d(laguerre(4, alpha, z))"

GOLDEN = [
    ("01-quadratic", "z^2 + a"),
    ("02-kim", KIM),
    ("03-newton-hermite", NEWTON_H5),
    ("04-gen-laguerre", GEN_LAGUERRE),
    ("05-unary-minus-power", "-z^2 + a"),
    ("06-nested-fraction", "1/(1/(z+1) + a)"),
    ("07-poly-plus-pole", "z^5 + a/z^2 - 0.3"),
    ("08-jacobi", "jacobi(3, alpha, beta, z) + 2*3i"),
    ("09-derivative-quotient", "d((z^2+1)/(z-1))"),
    ("10-cyclotomic-greek", "z^(-2) + lam*cyclotomic(12, z) + a1*z"),
    ("11-halley-laguerre", HALLEY_L4),
    ("12-relaxed-newton-chebyshev", "z - a*chebyshev_t(6, z)/d(chebyshev_t(6, z))"),
]
GOLDEN_COMPACT = [("13-chip-kim", KIM), ("14-chip-halley-laguerre", HALLEY_L4)]


# ---------------------------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------------------------
_CPP_CALL = re.compile(r'\b(?:parse_rational|from_expression|parse_ok|poly_is|parse_error)\(\s*"((?:[^"\\]|\\.)*)"')
_PY_CALL = re.compile(r'''(?:_formula_edit\.setText|build_pq_map|from_expression|parse_rational_parameters|'''
                      r'''parse_rational_info|parse_rational_coefficients|cdx_coeffs)\(\s*(["'])((?:(?!\1)[^\\]|\\.)*)\1''')

EDGE_CASES = [
    "-z^2", "z^2^3", "2in", "2i", "3j*z", "0x1p3*z", "0x1A + z", "0x", "0xz", "1e-3*z", "1.e5*z",
    "1e", "z^ ( - 2 )", "z^(+3)", ".5*z", "5.", ".", "z_1", "a_1", "H(2,z)+H", "P*z + L + T + U + He",
    "hermite(2.0,z)", "hermite(007,z)", "hermite( 3 , z )", "d (z^2)", "d(z^2", "sin(z)", "foo(z)",
    "z (2)", "jacobi(2,1,2,z)", "P(2,1,z)", "P(2,1,2,z)", "L(3,a,z)", "", "   ", "z +", "(z", "z)",
    "z # a", "z^a", "2*-3", "--z", "+z", "z/0", "1/(2-2)", "laguerre(3,z-z,z)", "laguerre(3, 1/a, z)",
    "laguerre(3, d(z), z)", "laguerre(3, z/z, z)", "cyclotomic(0,z)", "cyclotomic(1,z)", "hermite(41,z)",
    "hermite(40,z)", "hermite(1000000000, z)", "hermite(,z)", "hermite(3,)", "hermite(3)",
    "d(z,1)", "d()", "hermite_e(3,z)", "chebyshev_u(4, 2*z - 1)", "gegenbauer(3, lam, z)",
    "bessel_poly(3, a*z)", "bernoulli(4, z) + euler_poly(3, z)", "z - a*hermite(5,z)/d(hermite(5,z))",
    "d(d(d(d(z^5))))", "d(hermite(3, 2*z))", "alpha*beta + gamma1", "ab*z", "pi*e",
    "(z^2 + 1)/(z^2 - 1)", "z^-1", "z^-(1)", "z ^ 2 * a", "a b", "2 3", "z^2 +- a",
    KIM, HALLEY_L4, NEWTON_H5, GEN_LAGUERRE,
]


def _cpp_unescape(s: str) -> str:
    return codecs.decode(s, "unicode_escape")


def corpus() -> list[str]:
    out: list[str] = []
    for path in sorted((REPO / "cdx" / "test").glob("*.cpp")):
        out += [_cpp_unescape(m.group(1)) for m in _CPP_CALL.finditer(path.read_text())]
    py_files = sorted((REPO / "app").glob("test_*.py")) + [REPO / "cdx" / "test" / "test_special_polys.py"]
    py_files += sorted((REPO / "modules").glob("*/test_*.py"))
    for path in py_files:
        if path.name == "test_mathdisplay.py":
            continue
        out += [_cpp_unescape(m.group(2)) for m in _PY_CALL.finditer(path.read_text())]
    # cdx/test/test_special_polys.py's family templates ("hermite({n}, z)", ...)
    templ = re.findall(r'\("([a-z_]+\(\{n\}[^"]*)"', (REPO / "cdx" / "test" / "test_special_polys.py").read_text())
    for t in templ:
        out += [t.format(n=n) for n in range(9)]
    out += EDGE_CASES
    seen, uniq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


ALGEBRAIC = ("literal zero denominator", "must not depend on z", "no division by a parameter",
             "has a zero denominator")


# ---------------------------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------------------------
def to_gray(img: QImage) -> np.ndarray:
    """Composite over white, to float gray in [0, 1] (1 = white)."""
    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
    arr = arr.reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4).astype(np.float64) / 255.0
    b, g, r, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    gray = (r + g + b) / 3.0
    return gray * a + (1.0 - a)   # premultiplied colour + white * (1 - alpha), approximately


def blur(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0


def compare(golden: QImage, got: QImage) -> tuple[bool, str]:
    g, c = to_gray(golden), to_gray(got)
    dh, dw = abs(g.shape[0] - c.shape[0]), abs(g.shape[1] - c.shape[1])
    if dh > 4 or dw > max(6, 0.03 * g.shape[1]):
        return False, f"size {c.shape[1]}x{c.shape[0]} vs golden {g.shape[1]}x{g.shape[0]}"
    h, w = min(g.shape[0], c.shape[0]), min(g.shape[1], c.shape[1])
    diff = np.abs(blur(g[:h, :w]) - blur(c[:h, :w]))
    ink = max(1, int(((g[:h, :w] < 0.8) | (c[:h, :w] < 0.8)).sum()))
    bad = int((diff > 0.30).sum())
    frac = bad / ink
    return frac < 0.06, f"{bad} differing px / {ink} ink px = {frac:.3f}"


def render_golden(md, source: str, compact: bool) -> QImage:
    if compact:
        return md.render_image(source, px=15, compact_size=(440.0, 30.0), color=QColor("black"), dpr=1.0)
    return md.render_image(source, px=20, color=QColor("black"), dpr=1.0)


def has_ellipsis(box) -> bool:
    if getattr(box, "source_text", "") in ("⋯", "…"):
        return True
    kids = list(getattr(box, "children", []))
    for attr in ("num", "den", "inner", "content", "base", "sup", "sub"):
        if getattr(box, attr, None) is not None:
            kids.append(getattr(box, attr))
    return any(has_ellipsis(k) for k in kids)


def ink_outside(md, node, px: float, w: float, h: float) -> int:
    """Pixels painted outside a w x h rect (compact layout, NO clipping)."""
    margin = 24
    img = QImage(int(w) + 2 * margin, int(h) + 2 * margin, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    c = md.layout_compact(node, px, w, h)
    md.paint_box(p, c.box, QRectF(margin, margin, w, h), QColor("black"), clip=False)
    p.end()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
    arr = arr.reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4].reshape(img.height(), img.width(), 4)
    alpha = arr[..., 3].copy()
    alpha[margin:margin + int(np.ceil(h)), margin:margin + int(np.ceil(w))] = 0
    return int((alpha > 8).sum())


def main() -> None:
    regen = "--regen" in sys.argv
    app = QApplication.instance() or QApplication([])
    from app import theme
    theme.apply(app)
    from app import mathdisplay as md

    print("=== app.mathdisplay tests ===")
    check(md.math_font_available(), "STIX Two Math is bundled and registers")

    # ---- parity -------------------------------------------------------------------------------
    print("\nparser parity with cdx (C++ + Python test corpora + edge cases):")
    exprs = corpus()
    accepted = rejected = 0
    missing, leaked, odd = [], [], []
    for e in exprs:
        err = md.cdx_error(e)
        try:
            md.parse_display(e)
            py_ok = True
        except md.DisplayParseError:
            py_ok = False
        if err is None:
            accepted += 1
            if not py_ok:
                missing.append(e)
                continue
            try:
                md.layout_display(md.typeset_source(e), 16)
            except Exception as ex:   # noqa: BLE001 -- any failure to lay out is a failure
                missing.append(f"{e}  [layout: {ex}]")
        else:
            rejected += 1
            try:
                md.typeset_source(e)
                leaked.append(e)
            except md.DisplayParseError:
                pass
            if py_ok and not any(k in err for k in ALGEBRAIC):
                odd.append(f"{e!r}: {err}")
    print(f"  ({len(exprs)} expressions: {accepted} accepted by cdx, {rejected} rejected)")
    check(len(exprs) > 150, f"the corpus is substantial ({len(exprs)} expressions)")
    check(not missing, "everything cdx accepts displays" + (f": {missing[:5]}" if missing else ""))
    check(not leaked, "nothing cdx rejects displays" + (f": {leaked[:5]}" if leaked else ""))
    check(not odd, "the display parser alone rejects everything cdx rejects except algebraic cases"
          + (f": {odd[:5]}" if odd else ""))

    # ---- conventions --------------------------------------------------------------------------
    print("\ndisplay conventions (plain-text rendering of the display AST):")

    def pt(s):
        return md.plain_text(md.parse_display(s))

    for src, want, what in [
        ("2*(a-1)*z", "2(a − 1)z", "implicit multiplication, real minus, thin-spaced binary minus"),
        ("2*3", "2·3", "a dot only between numbers"),
        ("-z^2 + a", "(−z)² + a", "unary minus binds tighter than ^ (as cdx computes it)"),
        ("hermite(5, z)", "H₅(z)", "hermite -> H₅"),
        ("H(5, z)", "H₅(z)", "alias H -> H₅"),
        ("laguerre(4, a, z)", "L₄^(a)(z)", "generalized Laguerre L₄^(a)"),
        ("P(3, alpha, beta, x0)", "P₃^(α,β)(x₀)", "P with 4 args is Jacobi; Greek names; digit subscripts"),
        ("gegenbauer(2, lam, z)", "C₂^(λ)(z)", "lam -> λ"),
        ("d(hermite(5, z))", "H₅′(z)", "d of a named polynomial of z -> p′(z)"),
        ("d(d(legendre(4, z)))", "P₄″(z)", "...twice -> p″(z)"),
        ("d(hermite(5, 2*z))", "d/dz[H₅(2z)]", "...but not when the argument isn't z (chain rule)"),
        ("d(d(z^3))", "d²/dz²[z³]", "nested d of an expression -> d²/dz²[...]"),
        ("cyclotomic(12, z)", "Φ₁₂(z)", "cyclotomic -> Φ"),
        ("a*-z", "a(−z)", "a signed factor is parenthesised"),
    ]:
        got = pt(src)
        check(got == want, f"{what}: {src!r} -> {got!r}")

    # ---- golden images ------------------------------------------------------------------------
    print("\ngolden images:")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in GOLDEN + GOLDEN_COMPACT:
        img = render_golden(md, src, name in dict(GOLDEN_COMPACT))
        path = GOLDEN_DIR / f"{name}.png"
        if regen or not path.exists():
            img.save(str(path))
            print(f"  (wrote {path.relative_to(REPO)})")
            continue
        ok, detail = compare(QImage(str(path)), img)
        check(ok, f"{name}: {detail}")

    # ---- compact: fit, stack/inline, elision --------------------------------------------------
    print("\ncompact layout:")
    sources = [KIM, HALLEY_L4, NEWTON_H5, GEN_LAGUERRE, "z^2 + a", "1/(1/(z+1) + a)",
               "d((z^2+1)/(z-1))", "jacobi(3, alpha, beta, z) + 2*3i"]
    sizes = [(440, 30), (260, 30), (160, 30), (90, 30), (300, 20), (60, 18), (440, 14)]
    outside = []
    for src in sources:
        node = md.typeset_source(src)
        for w, h in sizes:
            n = ink_outside(md, node, 15, w, h)
            if n:
                outside.append((src[:24], w, h, n))
    check(not outside, "layout never paints outside the rect (no clipping)"
          + (f": {outside[:4]}" if outside else ""))

    c = md.layout_compact(md.typeset_source(NEWTON_H5), 15, 440, 30)
    check(c.stacked and not c.elided, "Newton·H₅ stacks in a 30 px chip (fits at >= 70%)")
    c = md.layout_compact(md.typeset_source(KIM), 15, 440, 30)
    check(not c.stacked and c.px >= 15 * 0.7,
          f"the Kim operator falls back to inline (...)/(...) (px {c.px:.0f})")
    c = md.layout_compact(md.typeset_source("z^2 + a/z"), 15, 440, 30)
    check(c.stacked and c.px == 15, "a simple fraction stacks at full size")

    print("\nmiddle elision keeps both ends:")
    for src in (KIM, HALLEY_L4, "z^5 + z^4 + z^3 + z^2 + z + a*z^7 + 3*z^9 - z^11"):
        node = md.typeset_source(src)
        c = md.layout_compact(node, 15, 100, 30)
        full = md._Builder(md._Style(c.px, inline_frac=not c.stacked, compact=True)).build(node, c.px)
        box = c.box
        pairs = []
        if isinstance(full, md.Frac) and isinstance(box, md.Frac):
            pairs = [(full.num, box.num), (full.den, box.den)]
        else:
            pairs = [(full, box)]
        ok = c.elided and box.w <= 100 + 1e-6 and has_ellipsis(box)
        for f, b in pairs:
            fi, bi = [i.key() for i in f.items()], [i.key() for i in b.items()]
            ok = ok and fi[0] == bi[0] and fi[-1] == bi[-1]
        check(ok, f"{src[:30]}…: elided to {box.w:.0f} px, first and last items kept")

    # ---- fallback font --------------------------------------------------------------------------
    print("\nfallback (no STIX):")
    saved = md._FONTS
    md._FONTS = md._Fonts()
    md._FONTS.stix = False
    img = md.render_image(NEWTON_H5, px=18, color=QColor("black"))
    md._FONTS = saved
    check(img.width() > 50 and (to_gray(img) < 0.5).any(), "renders with Inter italic when STIX is absent")

    # ---- widgets --------------------------------------------------------------------------------
    print("\nwidgets:")
    view = md.FormulaView(compact=False, px=18)
    check(view.set_source("z^2 + a") and not view.is_greyed(), "FormulaView shows a valid formula")
    check(not view.set_source("z^2 +") and view.is_greyed() and view.source() == "z^2 + a",
          "a parse error keeps the last good formula, greyed")
    check(view.set_source("z^3") and not view.is_greyed(), "the next good formula clears the grey")
    check(not view.set_source("5/(z-z)") and view.source() == "z^3",
          "an expression only cdx rejects (zero denominator) is not displayed either")

    from app.widgets import MapChip
    chip = MapChip()
    chip.set_map("custom", KIM)
    chip.resize(chip.sizeHint())
    chip.show()
    app.processEvents()
    check(chip.formula_view.width() <= MapChip.MAX_FORMULA_WIDTH + 1, "the chip's formula width is capped")
    check(chip.formula_view.is_elided(), "the Kim operator is middle-elided in the chip")
    check(chip.text_shown().startswith("custom ") and "⋯" not in chip.text_shown(),
          f"text_shown: {chip.text_shown()[:40]}…")
    chip.setToolTip("custom: …")
    QApplication.sendEvent(chip, QHelpEvent(QEvent.Type.ToolTip, QPoint(5, 5), chip.mapToGlobal(QPoint(5, 5))))
    check(md.FormulaToolTip.shown_source() == KIM, "hovering the chip shows the full typeset formula")
    QApplication.sendEvent(chip, QEvent(QEvent.Type.Leave))
    check(md.FormulaToolTip.shown_source() is None, "...and leaving hides it")
    chip.hide()

    from app.info_panel import InfoPanel
    info = InfoPanel()
    info.resize(400, 600)
    info.set_map_formula(NEWTON_H5)
    info.set_this_map([("Map", "custom: " + NEWTON_H5), ("Degree", "5")])
    img = info.loadResource(2, __import__("PySide6.QtCore", fromlist=["QUrl"]).QUrl("formula:this-map"))
    check(isinstance(img, QImage) and not img.isNull(), "Info's \"This map\" has a typeset formula image")
    check("formula:this-map" in info.toHtml(), "...referenced from the document")

    from app.equation_panel import EquationPanel
    from app.session import Session
    panel = EquationPanel(Session(), lambda: None)
    panel._formula_edit.setText(NEWTON_H5)
    check(panel.preview().source() != NEWTON_H5, "the Equation preview is debounced (not yet updated)")
    panel.flush_preview()
    check(panel.preview().source() == NEWTON_H5 and not panel.preview().is_greyed(),
          "...and updates once the debounce fires")
    panel._formula_edit.setText(NEWTON_H5 + " +")
    panel.flush_preview()
    check(panel.preview().is_greyed() and panel.preview().source() == NEWTON_H5
          and panel._error_label.text() != "", "an error greys the last good preview and shows the message")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

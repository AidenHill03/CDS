"""test_theme.py -- app/theme.py: tokens, fonts, the generated style sheet,
and the rule that no other source file hard-codes a color.

Run with (from the repository root):

    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m app.test_theme
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (QApplication, QComboBox, QLineEdit, QPushButton, QSpinBox,
                               QToolButton, QWidget)

from app import theme

failures = 0
REPO = Path(__file__).resolve().parent.parent

# Hex color literals ("#abc", "#a1b2c3", "#aa112233") and CSS color functions.
_HEX = re.compile(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{2})?)?\b")
_CSS_FN = re.compile(r"\brgba?\(")


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def _luminance(c: QColor) -> float:
    def ch(v: int) -> float:
        s = v / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(QColor(a)), _luminance(QColor(b))), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def ui_sources() -> list[Path]:
    """Every non-test source that builds UI: app/, the module panels, the
    composition root and the dev tools."""
    files = [p for p in (REPO / "app").glob("*.py") if not p.name.startswith("test_")]
    files += [p for p in (REPO / "modules").glob("*/panel.py")]
    files += [REPO / "phoenix.py"]
    files += list((REPO / "tools").glob("*.py")) if (REPO / "tools").is_dir() else []
    return [p for p in files if p.name != "theme.py"]


def main() -> None:
    messages: list[str] = []

    def handler(mode, _context, message):
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg):
            messages.append(message)

    qInstallMessageHandler(handler)
    app = QApplication.instance() or QApplication(sys.argv)

    print("tokens:")
    for name in theme.token_names():
        check(QColor(getattr(theme.DARK, name)).isValid(), f"{name} is a valid color")
    check(contrast(theme.DARK.muted, theme.DARK.surface) >= 4.5,
          f"muted on surface >= 4.5:1 ({contrast(theme.DARK.muted, theme.DARK.surface):.2f})")
    check(contrast(theme.DARK.text, theme.DARK.raised) >= 4.5, "text on raised >= 4.5:1")
    check(contrast(theme.DARK.error, theme.DARK.surface) >= 4.5, "error text on surface >= 4.5:1")
    check(contrast(theme.DARK.on_amber, theme.DARK.amber) >= 4.5, "primary button text >= 4.5:1")
    check("dark" in theme.THEMES and theme.THEMES["dark"] is theme.DARK,
          "themes are a name -> token-set table (room for a light theme)")

    print("\nfonts:")
    loaded = theme.load_fonts()
    for family in (theme.FONT_UI, theme.FONT_HEADING, theme.FONT_MONO):
        check(family in loaded and family in QFontDatabase.families(), f"{family} is registered")
    for name in theme.FONT_FILES:
        check((theme.FONTS_DIR / name).is_file(), f"{name} is bundled")
    for lic in ("Inter-OFL.txt", "SpaceGrotesk-OFL.txt", "JetBrainsMono-OFL.txt"):
        text = (theme.FONTS_DIR / lic).read_text(encoding="utf-8") if (theme.FONTS_DIR / lic).is_file() else ""
        check("SIL Open Font License" in text, f"{lic} is the OFL")
    spec = (REPO / "complexdynamics.spec").read_text(encoding="utf-8")
    check('"app" / "resources" / "fonts"' in spec, "the PyInstaller spec bundles the fonts")
    check(theme.load_fonts() is loaded, "fonts are registered once")

    print("\nstyle sheet:")
    theme.apply(app)
    probe = QWidget()
    probe.ensurePolished()
    check(theme.active_style_name.lower() == "fusion", "Fusion style on every platform")
    check(not any("style" in m.lower() and "parse" in m.lower() for m in messages),
          "the generated style sheet parses without warnings")
    check(app.palette().color(app.palette().ColorRole.Highlight) == QColor(theme.DARK.teal),
          "palette highlight (Fusion focus frames, selections) is teal")
    check(theme.DARK.teal.lower() in theme.stylesheet().lower()
          and ":focus" in theme.stylesheet(), "style sheet has teal :focus rules")

    print("\nminimum control heights:")
    host = QWidget()
    for widget in (QPushButton("Button", host), QLineEdit(host), QComboBox(host), QSpinBox(host)):
        widget.ensurePolished()
        h = widget.sizeHint().height()
        check(h >= theme.CONTROL_HEIGHT, f"{type(widget).__name__} is >= {theme.CONTROL_HEIGHT} px ({h})")
    chip = QToolButton(host)
    chip.setText("Orbit")
    chip.setProperty("role", "chip")
    chip.ensurePolished()
    check(chip.sizeHint().height() >= theme.CONTROL_HEIGHT, f"a chip is >= {theme.CONTROL_HEIGHT} px")

    print("\ninvalid-field property:")
    edit = QLineEdit(host)
    theme.set_invalid(edit, True)
    check(edit.property("invalid") is True, "set_invalid marks a field")
    theme.set_invalid(edit, False)
    check(edit.property("invalid") is False, "and clears it")

    print("\nno colors outside app/theme.py:")
    offenders = []
    for path in ui_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
            if _HEX.search(code) or _CSS_FN.search(code):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    for o in offenders:
        print(f"      {o}")
    check(not offenders, f"no hex/rgb() color literals in {len(ui_sources())} UI source files")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

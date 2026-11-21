"""theme.py -- the ONE place colors, fonts and sizes are defined.

Every color in the UI comes from a token set here (mockup B, "Dusk"); the
Qt style sheet is generated from those tokens, never written by hand
elsewhere. app/test_theme.py enforces that no other app/ or module-panel
source file contains a hex color literal.

Structure for a second theme: a theme is just another `Tokens` instance in
THEMES (mockup C's paper palette would be `THEMES["light"]`); `stylesheet()`,
`palette()` and `color()` take whichever one is active, so nothing outside
this module needs to know which theme is showing.

Widgets opt into a look through dynamic properties, not object-specific
style sheets: `role` ("primary", "chip", "segment", "error", "muted",
"caption", "section", "planeTitle", ...), plus `plane` ("parameter" /
"dynamical") for the accent, and `invalid` for a field that failed
validation. `set_property()` re-polishes after a change so the look follows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class Tokens:
    bg: str              # window, plane area
    surface: str         # top bar, sidebar, dock
    raised: str          # chips, inputs, rows
    row: str             # list rows (mockup B's library rows: a step between surface and raised)
    control_on: str      # selected segment
    border: str          # dividers
    border_strong: str   # inputs, chip outlines
    text: str            # primary text
    text_soft: str       # secondary values inside readout chips
    muted: str           # labels, secondary text (>= 4.5:1 on surface)
    amber: str           # parameter plane, primary button, marker
    teal: str            # dynamical plane, focus ring, progress
    error: str           # inline validation text
    on_amber: str        # text on the amber primary button
    overlay: str         # readout-chip background painted over an image (#AARRGGBB)


DARK = Tokens(
    bg="#0F1418",
    surface="#131A1F",
    raised="#1A232A",
    row="#182128",
    control_on="#22303A",
    border="#26313A",
    border_strong="#2C3A44",
    text="#E6EEF0",
    text_soft="#B9C7CE",
    muted="#93A4AD",
    amber="#FFBE5A",
    teal="#5FD0D6",
    error="#FF7A6B",
    on_amber="#1A1206",
    overlay="#D10F1418",          # bg at 82 % opacity
)

THEMES: dict[str, Tokens] = {"dark": DARK}

# The active token set and Qt style. Only apply() changes them.
current: Tokens = DARK
active_style_name = ""

# ---- type -----------------------------------------------------------------------
FONT_UI = "Inter"
FONT_HEADING = "Space Grotesk"
FONT_MONO = "JetBrains Mono"

FONT_SIZE_UI = 13          # px
FONT_SIZE_SMALL = 12
FONT_SIZE_CAPTION = 11
FONT_SIZE_MONO = 12
FONT_SIZE_HEADING = 15
FONT_SIZE_WORDMARK = 17

def resources_root() -> Path:
    """Where bundled resources live, frozen (PyInstaller) or not."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) / "app" / "resources" if base else Path(__file__).resolve().parent / "resources"


FONTS_DIR = resources_root() / "fonts"
FONT_FILES = (
    "Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf",
    "SpaceGrotesk-Medium.ttf", "SpaceGrotesk-Bold.ttf",
    "JetBrainsMono-Regular.ttf", "JetBrainsMono-Medium.ttf",
)

# ---- geometry (4 px grid) -----------------------------------------------------------
GRID = 4
GUTTER = 14                 # around the planes
RADIUS_CONTROL = 8
RADIUS_FRAME = 10
RADIUS_CHIP = 999           # "fully round"; QSS gets CHIP_HEIGHT / 2 (Qt mis-draws larger radii)
CONTROL_HEIGHT = 36         # every interactive control
SIDEBAR_CONTROL_HEIGHT = 44
CHIP_HEIGHT = 36
SEGMENT_HEIGHT = 36


def pill_radius(height: int = CHIP_HEIGHT) -> int:
    return min(RADIUS_CHIP, height // 2)


# ---- color helpers ----------------------------------------------------------------
def color(name: str, tokens: Tokens | None = None) -> QColor:
    """A token as a QColor, for code that paints (ImageView, chips' swatches)."""
    value = getattr(tokens or current, name)
    return QColor(value)


def mix(a: str, b: str, t: float) -> QColor:
    """a blended toward b by t (0..1) -- for derived shades (disabled text)
    rather than adding a hand-picked token for every state."""
    ca, cb = QColor(a), QColor(b)
    return QColor(round(ca.red() + (cb.red() - ca.red()) * t),
                  round(ca.green() + (cb.green() - ca.green()) * t),
                  round(ca.blue() + (cb.blue() - ca.blue()) * t))


def _css(name_or_color, tokens: Tokens) -> str:
    c = name_or_color if isinstance(name_or_color, QColor) else QColor(getattr(tokens, name_or_color))
    if c.alpha() == 255:
        return c.name(QColor.NameFormat.HexRgb)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})"


# ---- fonts --------------------------------------------------------------------------
_loaded_families: set[str] | None = None


def load_fonts() -> set[str]:
    """Registers the bundled fonts once. Returns the families that loaded;
    a file that is missing or fails to load is skipped silently, and Qt
    then falls back to a system font for that family."""
    global _loaded_families
    if _loaded_families is not None:
        return _loaded_families
    families: set[str] = set()
    for name in FONT_FILES:
        path = FONTS_DIR / name
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            families.update(QFontDatabase.applicationFontFamilies(font_id))
    _loaded_families = families
    return families


def ui_font(size: int = FONT_SIZE_UI, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies([FONT_UI])
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


def mono_font(size: int = FONT_SIZE_MONO, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies([FONT_MONO, "Menlo", "Consolas", "monospace"])
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


def heading_font(size: int = FONT_SIZE_HEADING, weight: QFont.Weight = QFont.Weight.DemiBold) -> QFont:
    font = QFont()
    font.setFamilies([FONT_HEADING, FONT_UI])
    font.setPixelSize(size)
    font.setWeight(weight)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 104)
    return font


# ---- palette and style sheet -----------------------------------------------------------
def palette(tokens: Tokens | None = None) -> QPalette:
    """Fusion draws whatever the style sheet doesn't cover from this palette,
    so it gets the same tokens (Highlight is teal: Fusion's own focus frames
    and selections come out in the focus color)."""
    t = tokens or current
    p = QPalette()
    R = QPalette.ColorRole
    disabled = mix(t.muted, t.surface, 0.45)
    for role, value in (
        (R.Window, t.bg), (R.WindowText, t.text), (R.Base, t.raised), (R.AlternateBase, t.surface),
        (R.ToolTipBase, t.raised), (R.ToolTipText, t.text), (R.PlaceholderText, t.muted),
        (R.Text, t.text), (R.Button, t.raised), (R.ButtonText, t.text), (R.BrightText, t.amber),
        (R.Light, t.border_strong), (R.Midlight, t.border_strong), (R.Mid, t.border),
        (R.Dark, t.bg), (R.Shadow, t.bg), (R.Highlight, t.teal), (R.HighlightedText, t.bg),
        (R.Link, t.teal), (R.LinkVisited, t.teal),
    ):
        p.setColor(role, QColor(value))
    for role in (R.WindowText, R.Text, R.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


def stylesheet(tokens: Tokens | None = None) -> str:
    t = tokens or current
    c = lambda name: _css(name, t)   # noqa: E731
    disabled = _css(mix(t.muted, t.surface, 0.45), t)
    pad_v = (CONTROL_HEIGHT - 2 - 20) // 2            # 1 px border, ~20 px line
    pill = pill_radius()
    return f"""
* {{
    font-family: "{FONT_UI}";
    font-size: {FONT_SIZE_UI}px;
}}
QMainWindow, QDialog {{ background: {c('bg')}; color: {c('text')}; }}
QWidget {{ color: {c('text')}; }}
QToolTip {{
    background: {c('raised')}; color: {c('text')};
    border: 1px solid {c('border_strong')}; padding: 4px 8px;
}}

/* ---- surfaces ---- */
QWidget[role="surface"], QDockWidget > QWidget, QStatusBar {{ background: {c('surface')}; }}
QWidget#TopBar {{ background: {c('surface')}; border-bottom: 1px solid {c('border')}; }}
QStatusBar {{ border-top: 1px solid {c('border')}; color: {c('muted')}; }}
QStatusBar QLabel {{ color: {c('muted')}; font-size: {FONT_SIZE_SMALL}px; }}
QSplitter::handle {{ background: {c('bg')}; }}
QMainWindow::separator {{ background: {c('border')}; width: 1px; height: 1px; }}
QMenuBar {{ background: {c('surface')}; color: {c('text')}; }}
QMenuBar::item:selected {{ background: {c('control_on')}; }}
QMenu {{ background: {c('raised')}; color: {c('text')}; border: 1px solid {c('border_strong')}; padding: 4px; }}
QMenu::item {{ padding: 6px 20px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c('control_on')}; }}
QMenu::item:disabled {{ color: {disabled}; }}
QMenu::separator {{ height: 1px; background: {c('border')}; margin: 4px 6px; }}

/* ---- docks ---- */
QDockWidget {{ color: {c('muted')}; titlebar-close-icon: none; }}
QDockWidget::title {{
    background: {c('surface')}; padding: 8px 14px; border-bottom: 1px solid {c('border')};
    font-family: "{FONT_HEADING}"; font-weight: 600;
}}

/* ---- planes ---- */
QWidget[role="paneColumn"] {{ border: 1px solid transparent; border-radius: {RADIUS_FRAME}px; }}
QWidget[role="paneColumn"][focused="true"] {{ border: 1px solid {c('teal')}; }}

/* ---- text roles ---- */
QLabel[role="muted"], QLabel[role="caption"] {{ color: {c('muted')}; }}
QLabel[role="caption"] {{ font-size: {FONT_SIZE_SMALL}px; }}
QLabel[role="section"] {{
    color: {c('muted')}; font-size: {FONT_SIZE_CAPTION}px; letter-spacing: 1px; padding-top: 4px;
}}
QLabel[role="error"] {{ color: {c('error')}; font-size: {FONT_SIZE_SMALL}px; }}
QLabel[role="wordmark"] {{
    font-family: "{FONT_HEADING}"; font-weight: 700; font-size: {FONT_SIZE_WORDMARK}px;
    letter-spacing: 1px;
}}
QLabel[role="dockTitle"] {{
    font-family: "{FONT_HEADING}"; font-weight: 600; color: {c('muted')}; letter-spacing: 1px;
}}
QLabel[role="planeTitle"] {{
    font-family: "{FONT_HEADING}"; font-weight: 600; font-size: {FONT_SIZE_HEADING}px;
    letter-spacing: 1px;
}}
QLabel[role="planeTitle"][plane="parameter"] {{ color: {c('amber')}; }}
QLabel[role="planeTitle"][plane="dynamical"] {{ color: {c('teal')}; }}
QLabel[role="status"] {{ color: {c('muted')}; font-size: {FONT_SIZE_SMALL}px; }}
QLabel[mono="true"], QLineEdit[mono="true"], QTableWidget[mono="true"] {{
    font-family: "{FONT_MONO}";
}}
QLabel[role="empty"] {{ color: {c('muted')}; }}

/* ---- inputs ---- */
QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {c('raised')}; color: {c('text')};
    border: 1px solid {c('border_strong')}; border-radius: {RADIUS_CONTROL}px;
    selection-background-color: {c('teal')}; selection-color: {c('bg')};
}}
QLineEdit, QAbstractSpinBox, QComboBox {{ padding: {pad_v}px 10px; min-height: 20px; }}
QLineEdit[invalid="true"], QAbstractSpinBox[invalid="true"] {{ border-color: {c('error')}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {c('teal')};
}}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {disabled}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {c('raised')}; border: 1px solid {c('border_strong')};
    selection-background-color: {c('control_on')}; selection-color: {c('text')};
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 18px; border: none; }}
QTextBrowser {{ background: {c('surface')}; border: none; }}

/* ---- buttons ---- */
QPushButton {{
    background: {c('raised')}; color: {c('text')};
    border: 1px solid {c('border_strong')}; border-radius: {RADIUS_CONTROL}px;
    padding: {pad_v}px 14px; min-height: 20px;
}}
QPushButton:hover {{ background: {c('control_on')}; }}
QPushButton:pressed {{ background: {c('border_strong')}; }}
QPushButton:disabled {{ color: {disabled}; }}
QPushButton:focus, QToolButton:focus {{ border: 1px solid {c('teal')}; }}
QPushButton[role="primary"] {{
    background: {c('amber')}; color: {c('on_amber')}; border: 1px solid {c('amber')}; font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {c('amber')}; border-color: {c('text')}; }}
QPushButton[role="primary"]:focus {{ border: 2px solid {c('teal')}; }}
QPushButton[role="link"] {{
    background: transparent; border: 1px solid transparent; color: {c('teal')}; padding-left: 0;
}}
QPushButton[role="link"]:focus {{ border: 1px solid {c('teal')}; }}
QToolButton {{
    background: {c('raised')}; color: {c('text')};
    border: 1px solid {c('border_strong')}; border-radius: {RADIUS_CONTROL}px;
    min-height: {CONTROL_HEIGHT - 2}px; min-width: {CONTROL_HEIGHT - 2}px;
}}
QToolButton:hover {{ background: {c('control_on')}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}

QToolButton[role="icon"] {{ background: transparent; border: 1px solid transparent; }}
QToolButton[role="icon"]:hover {{ background: {c('control_on')}; }}
QToolButton[role="icon"]:focus {{ border: 1px solid {c('teal')}; }}
QToolBar {{ background: {c('surface')}; border: none; padding: 0; spacing: 0; }}

/* map chip (top bar) */
QPushButton[role="mapChip"] {{
    border-radius: {pill}px; font-family: "{FONT_MONO}"; font-size: {FONT_SIZE_MONO}px;
    padding: {pad_v}px 14px;
}}

/* segmented controls: a frame holding exclusive, checkable tool buttons */
QFrame[role="segmented"] {{
    background: {c('bg')}; border: 1px solid {c('border')}; border-radius: {RADIUS_CONTROL}px;
}}
QToolButton[role="segment"] {{
    background: transparent; color: {c('muted')}; border: 1px solid transparent;
    border-radius: 6px; padding: 0 10px; min-height: {SEGMENT_HEIGHT - 2}px;
    font-size: {FONT_SIZE_SMALL + 1}px;
}}
QToolButton[role="segment"]:hover {{ color: {c('text')}; }}
QToolButton[role="segment"]:checked {{ background: {c('control_on')}; color: {c('text')}; font-weight: 600; }}
QToolButton[role="segment"]:focus {{ border: 1px solid {c('teal')}; }}
QFrame[role="segmentDivider"] {{ background: {c('border')}; }}

/* overlay chips */
QToolButton[role="chip"] {{
    background: transparent; color: {c('muted')}; border: 1px solid {c('border_strong')};
    border-radius: {pill}px; padding: 0 12px; min-height: {CHIP_HEIGHT - 2}px;
    font-size: {FONT_SIZE_SMALL}px;
}}
QToolButton[role="chip"]:checked {{ background: {c('raised')}; color: {c('text')}; border-color: {c('text_soft')}; }}
QToolButton[role="chip"][secondary="true"] {{ padding: 0 10px; font-size: {FONT_SIZE_CAPTION}px; }}
QToolButton[role="chip"]:disabled {{ color: {disabled}; border-color: {c('border')}; }}
QToolButton[role="chip"]:focus {{ border: 1px solid {c('teal')}; }}

/* ---- tabs: underline style (dock tabs: amber, sidebar tabs: teal) ---- */
QTabWidget::pane {{ border: none; background: {c('surface')}; }}
QTabBar {{ background: {c('surface')}; }}
QTabBar::tab {{
    background: transparent; color: {c('muted')}; border: none;
    border-bottom: 2px solid transparent; padding: 10px 12px; min-height: 20px;
}}
QTabBar::tab:selected {{ color: {c('text')}; font-weight: 600; border-bottom: 2px solid {c('amber')}; }}
QTabBar::tab:hover {{ color: {c('text')}; }}
QTabBar[role="sidebarTabs"] {{ border-bottom: 1px solid {c('border')}; }}
QTabBar[role="sidebarTabs"]::tab {{ min-height: {SIDEBAR_CONTROL_HEIGHT - 2 - 20}px; padding: 10px 8px; }}
QTabBar[role="sidebarTabs"]::tab:selected {{ border-bottom: 2px solid {c('teal')}; }}

/* ---- lists and tables ---- */
QListView, QListWidget, QTableView, QTableWidget, QTreeView {{
    background: {c('surface')}; color: {c('text')}; border: none;
    selection-background-color: {c('control_on')}; selection-color: {c('text')};
    gridline-color: {c('border')}; outline: 0;
}}
QListView:focus, QTableView:focus, QTreeView:focus {{ border: 1px solid {c('teal')}; }}
QHeaderView::section {{
    background: {c('surface')}; color: {c('muted')}; border: none;
    border-bottom: 1px solid {c('border')}; padding: 4px 6px; font-size: {FONT_SIZE_SMALL}px;
}}
QTableCornerButton::section {{ background: {c('surface')}; border: none; }}

/* ---- group boxes and check boxes ---- */
QGroupBox {{
    border: 1px solid {c('border')}; border-radius: {RADIUS_CONTROL}px;
    margin-top: 18px; padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c('muted')};
}}
QCheckBox, QRadioButton {{ spacing: 8px; min-height: {CONTROL_HEIGHT}px; }}
QCheckBox:focus, QRadioButton:focus {{ color: {c('teal')}; }}

/* ---- progress ---- */
QProgressBar {{
    background: {c('control_on')}; border: none; border-radius: 3px;
    max-height: 6px; min-height: 6px; text-align: center;
}}
QProgressBar::chunk {{ background: {c('teal')}; border-radius: 3px; }}

/* ---- scroll bars ---- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {c('border_strong')}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
"""


def set_property(widget: QWidget, name: str, value) -> None:
    """Sets a style-driving dynamic property and re-polishes, so a change
    (e.g. `invalid`) takes effect immediately."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_invalid(widget: QWidget, invalid: bool) -> None:
    set_property(widget, "invalid", bool(invalid))


def apply(app: QApplication, name: str = "dark") -> None:
    """Fusion on every platform + bundled fonts + palette + generated QSS.
    The menu bar stays native on macOS (Fusion doesn't change that)."""
    global current, active_style_name
    current = THEMES[name]
    app.setStyle("Fusion")
    # Read back before the style sheet wraps the style (QStyleSheetStyle has no name).
    active_style_name = app.style().name()
    load_fonts()
    app.setFont(ui_font())
    app.setPalette(palette(current))
    app.setStyleSheet(stylesheet(current))


def token_names() -> list[str]:
    return [f.name for f in fields(Tokens)]

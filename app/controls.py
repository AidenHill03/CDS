"""controls.py -- the single registry of Phoenix's input controls.

Every mouse gesture, plot key and keyboard shortcut the app responds to is
described ONCE here, as data. Three consumers read it, so documentation
cannot drift from behavior (the same single-source pattern OVERLAY_LAYERS
uses for the View menu):

  * SandboxWindow's QActions take their shortcuts from it (key_sequences).
  * ImageView's key handler takes its key sets from it (plot_keys).
  * The Info tab and the status-bar pane hints are generated from it
    (info_html, binding_label).

Where a handler cannot reasonably read from here (the mouse gestures, whose
logic is a state machine rather than a lookup), app/test_controls.py
asserts that the registry and the handler agree, entry by entry.

Labels are platform-aware: key sequences come from Qt's own
QKeySequence.toString(NativeText) on the running platform ("Ctrl" in a
QKeySequence IS Command on macOS); mouse gestures and plot keys use
binding_label's own formatter ("⇧ Drag" / "⌥ ←" on macOS, "Shift+Drag" /
"Alt+←" elsewhere). Every formatter takes an injectable `platform` (a
sys.platform string) so tests can check both without depending on the host.

Research modules contribute their own entries (modules may import app/, never
the reverse): a panel widget with a `controls` attribute passed to
SandboxWindow.add_panel has those entries added to the Info tab.
"""

from __future__ import annotations

import html
import sys
from dataclasses import dataclass, field
from typing import Iterable, Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

# ---- binding kinds -------------------------------------------------------------------


@dataclass(frozen=True)
class KeyBinding:
    """An application shortcut on a QAction (fires from anywhere in the
    window). `standard` is used wherever Qt has one, so each platform gets
    its native sequence; `sequences` are extra portable-text sequences
    ("Ctrl+E"); `fallback` is used only when everything else is empty (e.g.
    StandardKey.Preferences has no binding on Windows). `is_range` marks a
    two-sequence first...last family (Ctrl+1 ... Ctrl+9)."""
    standard: QKeySequence.StandardKey | None = None
    sequences: tuple[str, ...] = ()
    fallback: tuple[str, ...] = ()
    is_range: bool = False


@dataclass(frozen=True)
class MouseBinding:
    """A mouse/trackpad gesture on a widget. `gesture` is one of click,
    double-click, drag, wheel, pinch, hover; `button` left or middle;
    `modifiers` a tuple of "shift"/"alt"; `target` names what is clicked
    when it is not the plot itself ("row", "entry")."""
    gesture: str
    button: str = "left"
    modifiers: tuple[str, ...] = ()
    target: str = ""


@dataclass(frozen=True)
class PlotKey:
    """A key handled by a focused plot pane itself (ImageView.keyPressEvent),
    NOT an application shortcut -- it never fires while typing in a text
    field. `keys` are Qt.Key values; `modifiers` "shift"/"alt"."""
    keys: tuple[Qt.Key, ...]
    modifiers: tuple[str, ...] = ()


Binding = Union[KeyBinding, MouseBinding, PlotKey, None]


@dataclass(frozen=True)
class ControlEntry:
    """One documented control. `binding` None makes it a plain note (Info
    tab prose with no input of its own). `hint`, if set, is the short text
    the status-bar pane hint shows for it ("pan" -> "Drag: pan")."""
    id: str
    section: str
    action: str
    context: str
    binding: Binding = None
    hint: str | None = None


# ---- sections and contexts -------------------------------------------------------------

NAVIGATING = "Navigating"
PLOT = "Working in the plot"
ORBITS = "Orbits"
PANELS = "Panels"
SHORTCUTS = "Keyboard shortcuts"

ANY_PANE = "Any pane"
DYNAMICAL = "Dynamical plane"
PARAMETER = "Parameter plane"
ANYWHERE = "Anywhere in the window"

_ARROWS = (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)

# ---- the registry ------------------------------------------------------------------------

CONTROLS: tuple[ControlEntry, ...] = (
    # Navigating: every pane
    ControlEntry("plot.drag_pan", NAVIGATING, "Pan the view", ANY_PANE,
                 MouseBinding("drag"), hint="pan"),
    ControlEntry("plot.middle_drag_pan", NAVIGATING, "Pan the view", ANY_PANE,
                 MouseBinding("drag", button="middle")),
    ControlEntry("plot.shift_drag_box_zoom", NAVIGATING, "Zoom into the dragged box", ANY_PANE,
                 MouseBinding("drag", modifiers=("shift",)), hint="box zoom"),
    ControlEntry("plot.wheel_zoom", NAVIGATING, "Zoom in/out at the cursor", ANY_PANE,
                 MouseBinding("wheel"), hint="zoom (pinch too)"),
    ControlEntry("plot.pinch_zoom", NAVIGATING, "Zoom in/out at the cursor", ANY_PANE,
                 MouseBinding("pinch")),
    ControlEntry("plot.esc_cancel", NAVIGATING,
                 "Cancel the drag in progress (a pan snaps back; a box is discarded)", ANY_PANE,
                 PlotKey((Qt.Key.Key_Escape,))),
    ControlEntry("plot.key_zoom_in", NAVIGATING, "Zoom in at the center (numpad + too)", ANY_PANE,
                 PlotKey((Qt.Key.Key_Plus, Qt.Key.Key_Equal))),
    ControlEntry("plot.key_zoom_out", NAVIGATING, "Zoom out at the center (numpad − too)",
                 ANY_PANE, PlotKey((Qt.Key.Key_Minus,))),
    ControlEntry("app.zoom_in", NAVIGATING, "Zoom the focused pane in at its center", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.ZoomIn, sequences=("Ctrl+=",))),
    ControlEntry("app.zoom_out", NAVIGATING, "Zoom the focused pane out at its center", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.ZoomOut)),
    ControlEntry("app.reset_view", NAVIGATING, "Reset the focused pane's view", ANYWHERE,
                 KeyBinding(sequences=("Ctrl+0",))),

    # Working in the plot, by plane
    ControlEntry("plot.hover_readout", PLOT, "Show the point under the cursor in the status bar",
                 ANY_PANE, MouseBinding("hover")),
    ControlEntry("plot.click_seed", PLOT, "Seed an orbit at the clicked point (sets z0)",
                 DYNAMICAL, MouseBinding("click"), hint="seed orbit"),
    ControlEntry("plot.arrows_pan", PLOT,
                 "Pan by 10% of the view per press; hold to keep panning", DYNAMICAL,
                 PlotKey(_ARROWS), hint="pan"),
    ControlEntry("plot.click_set_a", PLOT, "Set the parameter a to the clicked point", PARAMETER,
                 MouseBinding("click"), hint="set a"),
    ControlEntry("plot.arrows_nudge", PLOT,
                 "Nudge a (the marker) by a few pixels; hold to keep moving", PARAMETER,
                 PlotKey(_ARROWS), hint="nudge a"),
    ControlEntry("plot.alt_arrows_pan", PLOT,
                 "Pan by 10% of the view per press (on the dynamical plane too)", PARAMETER,
                 PlotKey(_ARROWS, modifiers=("alt",)), hint="pan"),
    ControlEntry("note.coupled_off", PLOT,
                 "In Single view, clicking the parameter plane also switches that pane to "
                 "the dynamical plane (Escape) for the new a.", PARAMETER),
    ControlEntry("note.undo_a", PLOT,
                 "Undo / Redo with a plane focused step back and forth through the changes "
                 "to a (clicks, typed values; a held arrow nudge is one step).", PARAMETER),
    ControlEntry("note.offscreen_marker", PLOT,
                 "When the marker is outside the view, an arrow on the edge points to it; "
                 "click the arrow to re-center on it.", PARAMETER),

    # Orbits (dynamical plane)
    ControlEntry("plot.space_step", ORBITS, "Step the orbit once", DYNAMICAL,
                 PlotKey((Qt.Key.Key_Space,)), hint="step orbit"),
    ControlEntry("plot.shift_space_run", ORBITS,
                 "Run N steps (N is the number next to Run N on the Orbit tab)", DYNAMICAL,
                 PlotKey((Qt.Key.Key_Space,), modifiers=("shift",))),
    ControlEntry("note.orbit_strip", ORBITS,
                 "The chips on the image show z₀, n and the orbit's fate; the Orbit tab "
                 "(Equations & Settings) has Step, Run N and Clear, and typing a value into "
                 "z₀ there seeds an orbit too.", DYNAMICAL),

    # Panels (one or two lines per tab)
    ControlEntry("note.view_tab", PANELS,
                 "Each plane has its mode buttons in its header and overlay chips under it; "
                 "Coupled | Single, Reset view and Export are in the top bar. Click a plane to "
                 "make it the focused one.", "Planes"),
    ControlEntry("note.equation_tab", PANELS,
                 "Type a formula and press Return; the 'on plane' radio picks the parameter "
                 "the parameter plane sweeps; Add pole / Add zero; Undo / Redo (also in the "
                 "Edit menu).", "Equation tab"),
    ControlEntry("note.settings_tabs", PANELS,
                 "Rendering and Coloring apply as you edit, after a short pause; an invalid "
                 "value shows its error under the field and is not applied.",
                 "Rendering & Coloring tabs"),
    ControlEntry("panel.facts_seed", PANELS,
                 "Critical- or fixed-point row: seed an orbit there", "Facts tab",
                 MouseBinding("click", target="row")),
    ControlEntry("panel.facts_center", PANELS,
                 "Attracting-cycle or pole row: center the view there", "Facts tab",
                 MouseBinding("click", target="row")),
    ControlEntry("panel.library_load", PANELS, "Load that family (or select it and press Return)",
                 "Library tab",
                 MouseBinding("double-click", target="entry")),
    ControlEntry("note.library_tab", PANELS,
                 "Search by name or formula. Each entry's ⋯ menu has Load (and, for your own "
                 "entries, Rename / Delete); Save current as… adds the current map. Presets "
                 "are read-only; notes are saved as you type.", "Library tab"),

    # Keyboard shortcuts (application-wide QActions)
    ControlEntry("app.save", SHORTCUTS, "Save Experiment…", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.Save)),
    ControlEntry("app.open", SHORTCUTS, "Open Experiment…", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.Open)),
    ControlEntry("app.export_image", SHORTCUTS, "Export Image (PNG)…", ANYWHERE,
                 KeyBinding(sequences=("Ctrl+E",))),
    ControlEntry("app.settings", SHORTCUTS, "Settings… (the Rendering tab)", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.Preferences, fallback=("Ctrl+,",))),
    ControlEntry("app.undo", SHORTCUTS,
                 "Undo an Equation edit; in a plane, the last change to a (in a text field: "
                 "undo typing)", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.Undo)),
    ControlEntry("app.redo", SHORTCUTS,
                 "Redo an Equation edit; in a plane, a change to a (in a text field: redo "
                 "typing)", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.Redo)),
    ControlEntry("app.go_to_tab", SHORTCUTS,
                 "Go to a tab: Info, Library, Facts, then Equation, Rendering, Coloring, Orbit "
                 "(then any module tabs, and the next workspace)", ANYWHERE,
                 KeyBinding(sequences=("Ctrl+1", "Ctrl+9"), is_range=True)),
    ControlEntry("app.info", SHORTCUTS, "Open this Info tab", ANYWHERE,
                 KeyBinding(QKeySequence.StandardKey.HelpContents, sequences=("F1",))),
)

_BY_ID = {entry.id: entry for entry in CONTROLS}
assert len(_BY_ID) == len(CONTROLS), "duplicate control ids"


def entry(control_id: str) -> ControlEntry:
    return _BY_ID[control_id]


# ---- bindings for the code that implements them ---------------------------------------


def key_sequences(control_id: str) -> list[QKeySequence]:
    """The QAction shortcuts for a KeyBinding entry, on the RUNNING platform:
    the StandardKey's own bindings, then the extra sequences (deduplicated),
    or the fallback if both are empty. A range entry yields its first and
    last sequence only -- the window builds the members itself."""
    binding = entry(control_id).binding
    assert isinstance(binding, KeyBinding), control_id
    return _key_binding_sequences(binding)


_QT_MODIFIERS = {"shift": Qt.KeyboardModifier.ShiftModifier,
                 "alt": Qt.KeyboardModifier.AltModifier,
                 "ctrl": Qt.KeyboardModifier.ControlModifier,
                 "meta": Qt.KeyboardModifier.MetaModifier}


def plot_modifiers(control_id: str) -> Qt.KeyboardModifier:
    """The Qt modifier flags a PlotKey entry requires."""
    binding = entry(control_id).binding
    assert isinstance(binding, PlotKey), control_id
    flags = Qt.KeyboardModifier.NoModifier
    for name in binding.modifiers:
        flags |= _QT_MODIFIERS[name]
    return flags


def plot_keys(control_id: str) -> frozenset[Qt.Key]:
    binding = entry(control_id).binding
    assert isinstance(binding, PlotKey), control_id
    return frozenset(binding.keys)


# ---- platform-aware labels ---------------------------------------------------------------

_MAC_MODIFIER_SYMBOLS = {"ctrl": "⌘", "meta": "⌃", "alt": "⌥", "shift": "⇧"}
_MAC_MODIFIER_ORDER = ("meta", "alt", "shift", "ctrl")   # Apple's ⌃⌥⇧⌘ order
_OTHER_MODIFIER_NAMES = {"ctrl": "Ctrl", "meta": "Meta", "alt": "Alt", "shift": "Shift"}
_OTHER_MODIFIER_ORDER = ("ctrl", "alt", "shift", "meta")

_KEY_NAMES = {
    Qt.Key.Key_Left: "←", Qt.Key.Key_Right: "→", Qt.Key.Key_Up: "↑", Qt.Key.Key_Down: "↓",
    Qt.Key.Key_Plus: "+", Qt.Key.Key_Equal: "=", Qt.Key.Key_Minus: "−",
    Qt.Key.Key_Space: "Space", Qt.Key.Key_Escape: "Esc",
}
_PORTABLE_KEY_SYMBOLS_MAC = {"Left": "←", "Right": "→", "Up": "↑", "Down": "↓",
                             "Esc": "⎋", "Backspace": "⌫", "Del": "⌦", "Return": "↩"}


def _platform(platform: str | None) -> str:
    return sys.platform if platform is None else platform


def is_mac(platform: str | None = None) -> bool:
    return _platform(platform) == "darwin"


def _modifier_prefix(modifiers: Iterable[str], platform: str | None) -> str:
    mods = set(modifiers)
    if not mods:
        return ""
    if is_mac(platform):
        return "".join(_MAC_MODIFIER_SYMBOLS[m] for m in _MAC_MODIFIER_ORDER if m in mods) + " "
    return "".join(_OTHER_MODIFIER_NAMES[m] + "+" for m in _OTHER_MODIFIER_ORDER if m in mods)


def _portable_chord_to_label(chord: str, platform: str | None) -> str:
    """'Shift+Ctrl+Z' (QKeySequence PortableText) -> '⇧⌘Z' / 'Ctrl+Shift+Z'.
    Splits on '+' but keeps a trailing '+' key ('Ctrl++')."""
    parts = chord.split("+")
    if chord.endswith("++"):
        parts = parts[:-2] + ["+"]
    *mods, key = parts
    mod_keys = {m.lower() for m in mods}
    if is_mac(platform):
        key = _PORTABLE_KEY_SYMBOLS_MAC.get(key, key)
        return "".join(_MAC_MODIFIER_SYMBOLS[m] for m in _MAC_MODIFIER_ORDER if m in mod_keys) + key
    return "".join(_OTHER_MODIFIER_NAMES[m] + "+" for m in _OTHER_MODIFIER_ORDER
                   if m in mod_keys) + key


def key_sequence_label(seq: QKeySequence, platform: str | None = None) -> str:
    """Qt's own native text on the running platform; for an injected OTHER
    platform (tests), the same conventions applied to the portable text.
    Note that WHICH sequences a StandardKey means always comes from the
    running platform (Qt has no API for another platform's bindings), so an
    injected platform only changes how a sequence is written."""
    if _platform(platform) == sys.platform:
        return seq.toString(QKeySequence.SequenceFormat.NativeText)
    portable = seq.toString(QKeySequence.SequenceFormat.PortableText)
    return ", ".join(_portable_chord_to_label(chord, platform) for chord in portable.split(", "))


def _mouse_label(binding: MouseBinding, platform: str | None) -> str:
    if binding.gesture == "pinch":
        if is_mac(platform):
            return "Pinch"
        # A Windows precision touchpad delivers pinch as Ctrl+wheel events.
        return "Pinch (precision touchpad) / " + _modifier_prefix(("ctrl",), platform) + "Scroll"
    base = {"click": "Click", "double-click": "Double-click", "drag": "Drag",
            "wheel": "Scroll", "hover": "Hover"}[binding.gesture]
    if binding.button == "middle":
        base = "Middle-" + base.lower()
    if binding.target:
        base += " " + binding.target
    return _modifier_prefix(binding.modifiers, platform) + base


def _plot_key_label(binding: PlotKey, platform: str | None) -> str:
    if set(binding.keys) == set(_ARROWS):
        keys = "←↑→↓"
    else:
        keys = " / ".join(_KEY_NAMES[k] for k in binding.keys)
    return _modifier_prefix(binding.modifiers, platform) + keys


def binding_label(binding: Binding, platform: str | None = None) -> str:
    """The human label for any binding, on `platform` (default: running)."""
    if binding is None:
        return ""
    if isinstance(binding, MouseBinding):
        return _mouse_label(binding, platform)
    if isinstance(binding, PlotKey):
        return _plot_key_label(binding, platform)
    seqs = _key_binding_sequences(binding)
    labels = [key_sequence_label(s, platform) for s in seqs]
    return " … ".join(labels) if binding.is_range else " / ".join(labels)


def _is_usable(seq: QKeySequence) -> bool:
    """False for a bare hardware/media key (Qt lists e.g. the keyboard's own
    "Save" or "Settings" key among some StandardKey bindings): a sequence
    with no modifier is kept only if it is a function key (F1)."""
    if seq.isEmpty():
        return False
    combo = seq[0]
    if combo.keyboardModifiers() != Qt.KeyboardModifier.NoModifier:
        return True
    return Qt.Key.Key_F1.value <= combo.key().value <= Qt.Key.Key_F35.value


def _key_binding_sequences(binding: KeyBinding) -> list[QKeySequence]:
    result: list[QKeySequence] = []
    if binding.standard is not None:
        result.extend(s for s in QKeySequence.keyBindings(binding.standard) if _is_usable(s))
    for text in binding.sequences:
        seq = QKeySequence(text)
        if seq not in result:
            result.append(seq)
    return result or [QKeySequence(text) for text in binding.fallback]


# ---- generated documentation -----------------------------------------------------------


def hint_parts(contexts: Iterable[str], platform: str | None = None) -> list[str]:
    """'Label: hint' for every entry with a hint in one of `contexts`, in
    registry order -- the building blocks of the status-bar pane hint."""
    wanted = set(contexts)
    return [f"{binding_label(e.binding, platform)}: {e.hint}"
            for e in CONTROLS if e.hint and e.context in wanted]


@dataclass
class _Section:
    title: str
    groups: dict[str, list[ControlEntry]] = field(default_factory=dict)


def _row(label: str, action: str, context: str | None = None) -> str:
    cells = f"<td style='padding:2px 12px 2px 0; white-space:nowrap'><b>{html.escape(label)}</b></td>"
    cells += f"<td style='padding:2px 12px 2px 0'>{html.escape(action)}</td>"
    if context is not None:
        cells += f"<td style='padding:2px 0; color:gray'>{html.escape(context)}</td>"
    return f"<tr>{cells}</tr>"


def _group_html(entries: list[ControlEntry], platform: str | None) -> str:
    rows = [_row(binding_label(e.binding, platform), e.action)
            for e in entries if e.binding is not None]
    notes = [f"<p>{html.escape(e.action)}</p>" for e in entries if e.binding is None]
    table = f"<table>{''.join(rows)}</table>" if rows else ""
    return table + "".join(notes)


def info_html(extra: Iterable[ControlEntry] = (), platform: str | None = None) -> str:
    """The Info tab's whole document. `extra` are module-contributed entries
    (normally section PANELS, context '<Title> tab')."""
    entries = list(CONTROLS) + list(extra)
    out = [f"<h2>Controls</h2><p>Every control in the app. Shortcuts shown for "
           f"{'macOS' if is_mac(platform) else 'this platform'}.</p>"]
    for title in (NAVIGATING, PLOT, ORBITS, PANELS):
        groups: dict[str, list[ControlEntry]] = {}
        for e in entries:
            if e.section == title:
                groups.setdefault(e.context, []).append(e)
        out.append(f"<h3>{html.escape(title)}</h3>")
        for context, members in groups.items():
            out.append(f"<h4>{html.escape(context)}</h4>")
            out.append(_group_html(members, platform))
    rows = [_row(binding_label(e.binding, platform), e.action, e.context)
            for e in entries if isinstance(e.binding, (KeyBinding, PlotKey))]
    out.append(f"<h3>{SHORTCUTS}</h3><p>Plot keys work while a plot pane has focus (click it "
               f"first); the rest work anywhere in the window.</p>")
    out.append(f"<table>{''.join(rows)}</table>")
    return "\n".join(out)

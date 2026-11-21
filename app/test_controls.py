"""test_controls.py -- the controls registry (app/controls.py) and its labels.

Registry <-> handler agreement for the plot and the window lives in
app/test_sandbox.py (it needs a real ImageView and SandboxWindow); this file
covers the registry's own integrity, the platform-aware label formatter
(with the platform INJECTED, so both "darwin" and "win32" are checked on any
host), and the generated Info document.

Run with (from the repository root):

    QT_QPA_PLATFORM=offscreen PYTHONPATH=cdx/build python -m app.test_controls
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from app import controls
from app.controls import (CONTROLS, ControlEntry, KeyBinding, MouseBinding, PlotKey,
                          binding_label, key_sequence_label)

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)   # noqa: F841 (QKeySequence needs one)

    print("registry integrity:")
    ids = [e.id for e in CONTROLS]
    check(len(ids) == len(set(ids)), "every control id is unique")
    sections = {controls.NAVIGATING, controls.PLOT, controls.ORBITS, controls.PANELS,
                controls.SHORTCUTS}
    check(all(e.section in sections for e in CONTROLS), "every entry is in a known section")
    check(all(e.binding is not None for e in CONTROLS if e.hint),
          "every entry with a status-bar hint has a binding to label it with")
    check(all(e.binding is None for e in CONTROLS if e.id.startswith("note.")),
          "note.* entries are prose only")
    check(all(isinstance(e.binding, KeyBinding) for e in CONTROLS if e.id.startswith("app.")),
          "app.* entries are application shortcuts (KeyBindings)")
    check(all(isinstance(e.binding, (MouseBinding, PlotKey)) for e in CONTROLS
              if e.id.startswith("plot.")),
          "plot.* entries are mouse gestures or focus-scoped plot keys")
    check(all(controls.key_sequences(e.id) for e in CONTROLS if isinstance(e.binding, KeyBinding)),
          "every application shortcut resolves to at least one key sequence on this platform")

    print("\nkey sequences:")
    check(controls.key_sequences("app.export_image") == [QKeySequence("Ctrl+E")],
          "an explicit sequence is used as-is")
    check(controls._key_binding_sequences(KeyBinding(fallback=("Ctrl+,",))) == [QKeySequence("Ctrl+,")],
          "a binding with nothing else uses its fallback (Settings on Windows)")
    check(not controls._is_usable(QKeySequence(Qt.Key.Key_Save)),
          "a bare hardware/media key is not a usable shortcut")
    check(controls._is_usable(QKeySequence("F1")) and controls._is_usable(QKeySequence("Ctrl+S")),
          "function keys and modified keys are")
    info = controls.key_sequences("app.info")
    check(QKeySequence("F1") in info and len(info) == len(set(s.toString() for s in info)),
          "Info is bound to F1 on every platform, without duplicates (F1 IS HelpContents on Windows)")
    check(controls.plot_modifiers("plot.alt_arrows_pan") == Qt.KeyboardModifier.AltModifier
          and controls.plot_modifiers("plot.shift_space_run") == Qt.KeyboardModifier.ShiftModifier
          and controls.plot_modifiers("plot.space_step") == Qt.KeyboardModifier.NoModifier,
          "plot_modifiers maps registry modifier names to Qt flags")

    print("\nlabels, macOS (platform injected as 'darwin'):")
    mac = "darwin"
    check(binding_label(MouseBinding("drag", modifiers=("shift",)), mac) == "⇧ Drag", "⇧ Drag")
    check(binding_label(PlotKey((Qt.Key.Key_Left,), modifiers=("alt",)), mac) == "⌥ ←", "⌥ ←")
    check(binding_label(controls.entry("plot.alt_arrows_pan").binding, mac) == "⌥ ←↑→↓",
          "⌥ ←↑→↓ for the four arrows")
    check(binding_label(controls.entry("plot.pinch_zoom").binding, mac) == "Pinch", "Pinch")
    check(binding_label(controls.entry("plot.middle_drag_pan").binding, mac) == "Middle-drag",
          "Middle-drag")
    check(binding_label(controls.entry("panel.library_load").binding, mac) == "Double-click entry",
          "Double-click entry")
    check(binding_label(controls.entry("plot.shift_space_run").binding, mac) == "⇧ Space", "⇧ Space")
    check(key_sequence_label(QKeySequence("Ctrl+E"), mac) == "⌘E", "Ctrl+E is written ⌘E")
    check(key_sequence_label(QKeySequence("Shift+Ctrl+Z"), mac) == "⇧⌘Z",
          "Shift+Ctrl+Z is written ⇧⌘Z (Apple's modifier order)")
    check(key_sequence_label(QKeySequence("Meta+Alt+Left"), mac) == "⌃⌥←",
          "Meta is the Control key (⌃), arrows are symbols")
    check(key_sequence_label(QKeySequence("Ctrl++"), mac) == "⌘+", "a '+' key survives: ⌘+")
    check(binding_label(controls.entry("app.go_to_tab").binding, mac) == "⌘1 … ⌘9",
          "a range is written first … last")

    print("\nlabels, Windows (platform injected as 'win32'):")
    win = "win32"
    check(binding_label(MouseBinding("drag", modifiers=("shift",)), win) == "Shift+Drag", "Shift+Drag")
    check(binding_label(PlotKey((Qt.Key.Key_Left,), modifiers=("alt",)), win) == "Alt+←", "Alt+←")
    check(binding_label(controls.entry("plot.pinch_zoom").binding, win)
          == "Pinch (precision touchpad) / Ctrl+Scroll",
          "pinch is phrased for precision touchpads, which send Ctrl+wheel")
    check(binding_label(controls.entry("plot.shift_space_run").binding, win) == "Shift+Space",
          "Shift+Space")
    check(key_sequence_label(QKeySequence("Ctrl+E"), win) == "Ctrl+E", "Ctrl+E")
    check(key_sequence_label(QKeySequence("Shift+Ctrl+Z"), win) == "Ctrl+Shift+Z",
          "Ctrl+Shift+Z (Ctrl first)")
    check(binding_label(controls.entry("app.go_to_tab").binding, win) == "Ctrl+1 … Ctrl+9",
          "Ctrl+1 … Ctrl+9")

    print("\nlabels, running platform:")
    seq = QKeySequence("Shift+Ctrl+Z")
    check(key_sequence_label(seq) == seq.toString(QKeySequence.SequenceFormat.NativeText),
          "on the running platform, key labels are Qt's own NativeText")
    host = sys.platform
    check(key_sequence_label(seq, host) == key_sequence_label(seq),
          "injecting the running platform is the same as not injecting one")

    print("\nInfo document:")
    doc = controls.info_html(platform=win)
    for title in (controls.NAVIGATING, controls.PLOT, controls.ORBITS, controls.PANELS,
                  controls.SHORTCUTS):
        check(f"<h3>{title}</h3>" in doc, f"has the '{title}' section")
    import html
    check(all(html.escape(e.action) in doc for e in CONTROLS),
          "every registry entry's text appears in the document")
    check(html.escape(controls.entry("note.coupled_off").action) in doc,
          "the Coupled-View-off click note is included")
    extra = ControlEntry("x.demo", controls.PANELS, "Demo module control", "Demo tab",
                         MouseBinding("wheel"))
    doc2 = controls.info_html([extra], platform=win)
    check("Demo tab" in doc2 and "Demo module control" in doc2,
          "module-contributed entries get their own Panels group")
    shortcut_table = doc.split(f"<h3>{controls.SHORTCUTS}</h3>")[1]
    check(all(html.escape(e.action) in shortcut_table for e in CONTROLS
              if isinstance(e.binding, (KeyBinding, PlotKey))),
          "the Keyboard shortcuts table lists every key binding, app-wide and plot-scoped")
    check("Shift+Drag" in doc and "⇧" not in doc, "the document uses the injected platform's labels")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

"""test_about_dialog.py -- property-based checks for app/about_dialog.

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_about_dialog

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from app.about_dialog import AboutDialog, about_text
from app.version import AUTHOR, PRODUCT_NAME, VERSION

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(f"  [{'PASS' if cond else 'FAIL'}] {what}")
    if not cond:
        failures += 1


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.about_dialog tests ===")

    text = about_text()
    check(PRODUCT_NAME in text, "the product name appears")
    check(VERSION in text, "the version string appears")
    check(AUTHOR in text, "the author appears")
    check("Capabilities:" in text, "a capabilities section is present")
    check("Roadmap:" in text, "a roadmap section is present")
    # The roadmap is meant to be an ACCURATE, current summary, not a copy
    # of CLAUDE.md's own "Current state" section, which describes an
    # earlier (pre-P5b/P5c) snapshot -- this checks it doesn't claim
    # things that are actually already done (e.g. RationalMap wiring,
    # explicitly listed as still-open in that stale section) are still
    # open.
    check("wiring `RationalMap` into" not in text,
          "the roadmap does not repeat CLAUDE.md's stale 'still open' items that this "
          "project's own P5b/P5c work has since completed")

    dialog = AboutDialog()
    check(dialog.windowTitle() == f"About {PRODUCT_NAME}",
          "the dialog's window title names the product")
    check(dialog._label.text() == about_text(),
          "the dialog's displayed label text matches about_text() exactly -- one source of "
          "truth for the content")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else 'SOME CHECKS FAILED'} "
          f"({failures} failure{'' if failures == 1 else 's'})")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

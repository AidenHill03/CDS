"""test_complex_field.py -- property-based checks for app/complex_field.

Run with:

    QT_QPA_PLATFORM=offscreen \\
    QT_QPA_PLATFORM_PLUGIN_PATH=$(python3 -c "import PySide6,os;print(os.path.join(os.path.dirname(PySide6.__file__),'Qt','plugins','platforms'))") \\
    PYTHONPATH=cdx/build python3 -m app.test_complex_field

(from the repository root).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from app.complex_field import ComplexField, format_complex_literal, parse_complex_literal

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    status = "PASS" if cond else "FAIL"
    if not cond:
        failures += 1
    print(f"  [{status}] {what}")


def main() -> None:
    app = QApplication.instance() or QApplication([])

    print("=== app.complex_field tests ===")

    # ---- parse_complex_literal: every form P6 explicitly requires ---------------
    print("\nparse_complex_literal (required forms):")
    cases = {
        "-0.7269+0.1889i": complex(-0.7269, 0.1889),
        "2-3i": complex(2, -3),
        "i": complex(0, 1),
        "-i": complex(0, -1),
        "1": complex(1, 0),
        "3.2e-1+0i": complex(0.32, 0),
    }
    for text, expected in cases.items():
        got = parse_complex_literal(text)
        check(abs(got - expected) < 1e-12, f"{text!r} parses to {expected!r}")

    print("\nparse_complex_literal (rejections):")
    for bad in ["", "   ", "not a number", "z", "a", "a+1", "z*2", "1/0", "(("]:
        try:
            parse_complex_literal(bad)
            check(False, f"{bad!r} should have raised ValueError")
        except ValueError:
            check(True, f"{bad!r} raises ValueError, not a crash")

    # ---- format_complex_literal: round-trips through parse ----------------------
    print("\nformat_complex_literal (round-trip through parse):")
    for z in [complex(-0.7269, 0.1889), complex(2, -3), complex(0, 1), complex(0, -1),
             complex(1, 0), complex(0.32, 0), complex(0, 0)]:
        text = format_complex_literal(z)
        back = parse_complex_literal(text)
        check(abs(back - z) < 1e-9, f"format({z!r}) = {text!r} parses back to the same value")
    check(format_complex_literal(complex(0, 1)) == "i", "pure i formats as 'i', not '1i'")
    check(format_complex_literal(complex(0, -1)) == "-i", "pure -i formats as '-i', not '-1i'")

    # ---- ComplexField widget: commit, invalid input, set_value -------------------
    print("\nComplexField (commit on valid input):")
    committed = []
    field = ComplexField("a", complex(-0.7269, 0.1889))
    field.committed.connect(committed.append)
    check(field.value == complex(-0.7269, 0.1889), "starts at the constructor's initial value")

    field._line_edit.setText("2-3i")
    field._on_editing_finished()
    check(field.value == complex(2, -3), "a new valid value updates .value")
    check(committed == [complex(2, -3)], "committed fired exactly once with the new value")
    check(field._line_edit.styleSheet() == "", "no error styling after a valid commit")

    print("\nComplexField (invalid input -- no crash, model untouched):")
    committed.clear()
    field._line_edit.setText("not a number")
    field._on_editing_finished()   # must not raise
    check(field.value == complex(2, -3), "the last VALID value is kept, not zeroed")
    check(committed == [], "committed does NOT fire on invalid input")
    check(field._line_edit.styleSheet() != "", "invalid input marks the field with error styling")
    check(field._line_edit.toolTip() != "", "invalid input sets a tooltip explaining why")

    print("\nComplexField (recommitting the SAME text after an error clears it):")
    field._line_edit.setText("2-3i")
    field._on_editing_finished()
    check(field._line_edit.styleSheet() == "", "a subsequent valid commit clears the error styling")

    print("\nComplexField (set_value: programmatic sync, no feedback loop):")
    committed.clear()
    field.set_value(complex(1, 1))
    check(field.value == complex(1, 1), "set_value updates .value")
    check(field._line_edit.text() == format_complex_literal(complex(1, 1)),
          "set_value updates the displayed text")
    check(committed == [], "set_value does NOT emit committed -- no feedback loop back to itself")

    print("\nComplexField (re-typing the SAME value does not re-commit):")
    committed.clear()
    field._line_edit.setText(format_complex_literal(complex(1, 1)))
    field._on_editing_finished()
    check(committed == [], "typing back the value already held does not emit committed again")

    if failures:
        print(f"\n{failures} check(s) FAILED")
        raise SystemExit(1)
    print("\nALL CHECKS PASSED (0 failures)")


if __name__ == "__main__":
    main()

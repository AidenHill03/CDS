"""app/orbit_panel.py -- the orbit-tracking control strip: readout of the
current n/z and its classification, plus Step/Run N/Clear controls.

Sits directly BELOW the View tab's image (see app/sandbox.py's
SandboxWindow._build_ui), not a separate tab of its own -- it is
meaningless without the image it overlays (see app.sandbox.ImageView's
own orbit painting) and needs to stay visible while the user clicks the
image to seed a new orbit.

This widget owns no orbit state itself -- app.orbit_tracker.OrbitTracker
(held by ImageView, since painting the traced orbit and handling the
click-to-seed gesture both live there) is the single source of truth.
This is purely a thin readout + button strip, driven by ImageView's own
orbit_changed signal, matching the "panel talks to session-owned state
through the widget that actually owns it" shape every other panel in
this app already uses.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget


def _format_complex(z: complex) -> str:
    return f"{z.real:.6g}{'+' if z.imag >= 0 else ''}{z.imag:.6g}j"


class OrbitPanel(QWidget):
    def __init__(self, session, image_view, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.image_view = image_view

        self._build_ui()
        self.image_view.orbit_changed.connect(self._refresh)
        self._refresh()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        layout = QHBoxLayout()
        layout.setSpacing(6)
        outer.addLayout(layout)

        # The orbit readout line, under the buttons.
        self._readout_label = QLabel()
        self._readout_label.setWordWrap(True)
        self._readout_label.setProperty("mono", True)
        self._readout_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        step_button = QPushButton("Step")
        step_button.clicked.connect(lambda: self.image_view.step_orbit(1))
        layout.addWidget(step_button)

        self._run_n_spin = QSpinBox()
        self._run_n_spin.setRange(1, 1_000_000)
        self._run_n_spin.setValue(10)
        layout.addWidget(self._run_n_spin)

        run_button = QPushButton("Run N")
        run_button.clicked.connect(lambda: self.image_view.step_orbit(self._run_n_spin.value()))
        layout.addWidget(run_button)

        clear_button = QPushButton("Clear")
        # A lambda, like the other two buttons above -- NOT
        # self.image_view.clear_orbit directly, which would bind to
        # whichever ImageView is current AT CONNECT TIME and keep calling
        # that one even after set_image_view below repoints self.image_view
        # to a different pane's view.
        clear_button.clicked.connect(lambda: self.image_view.clear_orbit())
        layout.addWidget(clear_button)
        layout.addStretch(1)
        outer.addWidget(self._readout_label)

    def run_n(self) -> int:
        """The current "Run N" count (Shift+Space in a dynamical pane uses it)."""
        return self._run_n_spin.value()

    def set_image_view(self, image_view) -> None:
        """Repoints this panel at a DIFFERENT pane's ImageView -- Stage 2's
        "the orbit strip follows the dynamical pane": SandboxWindow calls
        this whenever which pane counts as "the" dynamical one changes
        (a mode switch on either pane, see its own _sync_orbit_panel).
        A no-op if already pointed here.
        """
        if image_view is self.image_view:
            return
        self.image_view.orbit_changed.disconnect(self._refresh)
        self.image_view = image_view
        self.image_view.orbit_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        tracker = self.image_view.orbit_tracker
        if tracker.state is None:
            # No orbit yet: say nothing here -- the status bar's pane hint
            # already says what a click on each plane does.
            self._readout_label.setText("")
            return
        classification = tracker.classify(self.session.map, self.session.param,
                                          self.session.render_settings.max_iter,
                                          self.session.render_settings.tol)
        self._readout_label.setText(
            f"n = {tracker.state.n}   z = {_format_complex(tracker.state.z)}   "
            f"{classification.text}")

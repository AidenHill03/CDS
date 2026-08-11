"""app/pane.py -- a Pane owns everything ONE image view needs that isn't
shared across the whole session: which plane it's looking at (render_mode),
where (viewport), its ImageView widget, and its own render-dispatch state.

WHY THIS SPLIT (the coupled-viewer refactor's own reason for existing):
two panes showing DIFFERENT planes of the SAME map at once (a parameter
pane and a dynamical pane, side by side) need two independent
viewports/modes -- there is no longer one authoritative "the" viewport.
What genuinely IS shared across every pane looking at the same experiment
stays on Session: map, param, render_settings, cache. See
app.session.Session's own docstring for the shared side of this split.

PER-PANE RENDER DISPATCH. Each pane gets its own monotonically-increasing
request_id counter and its own table of in-flight RenderTasks -- a stale
result for THIS pane must never paint a DIFFERENT pane, and a superseded
request for this same pane must be dropped while the current one applies.
Deliberately holds only the STATE (the counter, the pending-task table);
the dispatch LOGIC (building a RenderTask, connecting its signals, routing
a result into image_view.set_image) stays in app.sandbox.SandboxWindow,
which already owns the one shared QThreadPool every pane's renders run on
-- see its _start_render(pane)/_on_partial_ready(pane, ...)/etc. Kept
deliberately untyped on request/task types here (see the module-level note
below) so this file never needs to import app.sandbox, which imports THIS
module to construct panes -- that would be circular.

Plain Python plus one cdx type -- no Qt import here at all except the
image_view attribute's eventual value -- so a Pane's viewport/render_mode/
request-id bookkeeping is testable offscreen without constructing a real
QWidget.
"""

from __future__ import annotations

import cdx
from app.session import RENDER_MODES


class Pane:
    """`image_view` is None until the caller constructs the matching
    ImageView(pane, session, ...) and assigns it back -- a Pane must exist
    FIRST (with a real viewport/render_mode) for ImageView's own
    constructor to read an initial viewport/mode from, so the two can't be
    built in one step.
    """

    def __init__(self, viewport: cdx.Viewport, render_mode: str):
        self.viewport = viewport
        self.render_mode = render_mode
        self.image_view = None

        # Per-pane render supersession: bumped on every render this pane
        # starts; a result tagged with anything else is stale and dropped.
        # Values are RenderTask instances (app.sandbox) -- left untyped
        # rather than imported, to avoid a circular import (app.sandbox
        # imports Pane to construct one; RenderTask importing back here
        # would close the loop).
        self.request_id: int = 0
        self.pending_tasks: dict[int, object] = {}

    def set_render_mode(self, mode: str) -> None:
        """Same validation Session.set_render_mode used to do before
        render_mode moved here -- kept as a method (not left to callers to
        check RENDER_MODES themselves) so every path that sets a pane's
        mode (the toolbar combo, snapshot restore) rejects a bad string
        the same way.
        """
        if mode not in RENDER_MODES:
            raise ValueError(f"unknown render mode {mode!r}; must be one of {RENDER_MODES}")
        self.render_mode = mode

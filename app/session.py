"""app/session.py -- sandbox session state.

Owns the mutable state a user is actually operating on: the current map,
its bound parameter, the view window, which render mode is selected, and a
library of saved maps. The engine (cdx/) stays stateless apart from
Renderer's own configuration -- see ARCHITECTURE.md's layering rules. This
module holds no mathematics of its own; every operation here is a thin
wrapper delegating straight to a cdx call.

Requires the cdx extension module to be importable, and (since the
render-cache import below) must be imported as `app.session` -- the
package-qualified form, with the REPOSITORY ROOT on sys.path, not `session`
run standalone from inside this directory. See app/test_session.py or
app/sandbox.py for the actual invocation:

    PYTHONPATH=cdx/build python -m app.test_session

(run from the repository root).
"""

from __future__ import annotations

import json

import numpy as np

import cdx
from app.render_cache import RenderCache, make_key
from app.settings import Settings
from app.version import VERSION

RENDER_MODES = ("julia", "parameter", "basin", "greens", "parameter_greens", "parameter_basin",
                "parameter_period")

# Which modes render the PARAMETER plane (pixel = a value of the map's
# own free parameter `a`, orbit seeded at that parameter's critical
# point) rather than the DYNAMICAL plane (pixel = an initial condition,
# under whatever `a` is currently bound). Consulted wherever "which plane
# is this" matters beyond just dispatching a render call -- e.g. the
# critical-point overlay and cursor readout are dynamical-plane-only
# concepts (see P5c's own spec: critical points belong to the dynamical
# plane, not the parameter plane). "parameter_basin" (number of
# attracting cycles per parameter) and "parameter_period" (period of the
# attracting cycle a tracked critical orbit converges to) both participate
# in this SAME set, and so get the marker/arrow-nudging/click-sets-`a`
# machinery every other parameter-plane mode already has "for free" -- see
# app/sandbox.py's own gates, all keyed off membership here rather than a
# per-mode name list.
PARAMETER_PLANE_MODES = frozenset({"parameter", "parameter_greens", "parameter_basin",
                                   "parameter_period"})

# The six built-in families are read-only: save_to_library/rename_in_library/
# delete_from_library/set_library_notes below all refuse to touch a name in
# this set. Computed from FamilyLibrary.with_defaults() itself, not hand-
# copied, so this can never drift out of sync with whatever that actually
# ships (see cdx/src/rational.cpp's own with_defaults()).
PRESET_FAMILY_NAMES = frozenset(cdx.FamilyLibrary.with_defaults().names())

# Bumped whenever snapshot_to_dict's own dict shape changes in a way
# restore_from_snapshot can't parse forward-compatibly -- checked FIRST in
# restore_from_snapshot, before touching any other field, so a snapshot
# from an incompatible future (or unrelated) version is rejected cleanly
# rather than half-parsed.
#
# 1 -> 2 (coupled viewer Stage 4): a single top-level viewport/render_mode
# became a "layout" section (coupled flag, focused pane index, one
# viewport+render_mode per pane) -- an old version-1 file has neither the
# shape restore_from_snapshot now expects nor any way to guess a second
# pane's mode/viewport that never existed, so it is REJECTED by the
# version check below, not migrated. A straight reject also matches every
# other malformed-input case restore_from_snapshot already gives (raise,
# leave the session untouched) -- migrating in place would be the one
# exception to that rule, for a format only ever shipped internally
# between P7 and Stage 4, never in a tagged release.
#
# 2 -> 3 (embedded preview, Stage C): added one optional "preview" key
# (a base64 PNG thumbnail, or absent/None). Unlike 1 -> 2 above, this is
# PURELY ADDITIVE -- nothing existing changed shape, so a version-2 file
# still parses correctly through this exact same code path; it simply has
# no "preview" key for restore_from_snapshot to find (data.get returns
# None). MIN_SNAPSHOT_SCHEMA_VERSION below is what lets a version-2 file
# keep loading instead of being rejected outright the way version-1 is --
# raise it (to the same value as SNAPSHOT_SCHEMA_VERSION) if a future bump
# ever becomes a real structural break like 1 -> 2 was.
SNAPSHOT_SCHEMA_VERSION = 3
MIN_SNAPSHOT_SCHEMA_VERSION = 2

# Startup parameter for the DYNAMICAL plane: a filled, dendritic quadratic
# Julia set, not the origin (which gives the plain filled unit disc --
# correct but uninstructive). This is NOT what the startup view shows --
# Session starts in "parameter" mode (see render_mode below), which ignores
# the bound parameter entirely (see render_map) -- it's what the dynamical
# plane shows once the user switches to it.
DEFAULT_JULIA_PARAM = complex(-0.7269, 0.1889)

# Classic Mandelbrot framing: centered on -0.5, not 0, so the view spans
# roughly [-2, 1] on the real axis instead of cropping the cardioid's tail.
DEFAULT_PARAMETER_VIEW_CENTER = complex(-0.5, 0.0)
DEFAULT_PARAMETER_VIEW_SCALE = 1.5

# NOTE on Settings.resolution: see app/settings.py's DEFAULT_RESOLUTION
# comment for how that number (120, not the ~1100 a first, flawed
# measurement suggested) was actually measured, and the P5a-final commit
# message for the full story -- render_settings/viewport.resolution below
# both come from whatever Settings this Session was constructed with.

_POLY_TERM_FIELDS = ("coeff", "exponent", "param_power", "enabled", "label")
_POLE_TERM_FIELDS = ("location", "strength", "order", "param_power", "enabled",
                     "location_is_param", "label")


def _swap_terms(terms, i: int, j: int, fields: tuple[str, ...]) -> bool:
    """Swaps terms[i] and terms[j] field-by-field. False (no-op) if either
    index is out of range -- the caller uses this for "move up"/"move
    down", where one end of the list has no such neighbor. See
    Session.move_poly_term's docstring for why this is NOT
    `terms[i], terms[j] = terms[j], terms[i]`.
    """
    if not (0 <= i < len(terms) and 0 <= j < len(terms)):
        return False
    a, b = terms[i], terms[j]
    a_vals = tuple(getattr(a, f) for f in fields)
    b_vals = tuple(getattr(b, f) for f in fields)
    for f, v in zip(fields, b_vals):
        setattr(a, f, v)
    for f, v in zip(fields, a_vals):
        setattr(b, f, v)
    return True


def _greens_potential(potential: str) -> cdx.GreensPotential:
    """Translates app.settings.Settings.greens_potential's plain string
    (see its own FieldSpec -- "pragmatic"/"conformal", the same
    string-typed-setting convention color_palette/color_scaling already
    use) into the cdx enum render_greens/render_parameter_greens actually
    take. Kept a free function rather than inlined at each call site since
    both "greens" and "parameter_greens" need it.
    """
    return (cdx.GreensPotential.Conformal if potential == "conformal"
            else cdx.GreensPotential.Pragmatic)


def render_map(rational_map: cdx.RationalMap, param: complex, viewport: cdx.Viewport,
               settings: cdx.RenderSettings, mode: str,
               cancel: cdx.CancelToken | None = None,
               cache: RenderCache | None = None,
               potential: str = "pragmatic"):
    """Renders `rational_map` at `param` over `viewport`/`settings`, in the
    given mode. A free function rather than a Session method, and taking
    every value explicitly rather than reading them off a Session, so a
    background render thread (see app/sandbox.py) can call it on a snapshot
    of session state without touching the Session object itself -- Session
    is owned and mutated by the GUI thread, and has no thread-safety of its
    own to rely on.

    `cancel`, if given, is checked by the underlying render loop once per
    column (see cdx.Renderer's docs) and makes this interruptible from
    another thread; on cancellation the returned array is partial and
    should be discarded, not displayed. NOTE: find_attractors (used for
    "basin" mode, to discover the cycles being classified against) does not
    itself check `cancel` -- it has no per-column notion to check at, being
    an iterate-every-critical-orbit computation rather than a per-pixel one.
    A slow find_attractors call (a root-finding-heavy custom map) is not
    interruptible by this yet.

    `potential` (Stage 3) selects which of the two rational Green's
    potentials "greens"/"parameter_greens" compute for a RATIONAL map --
    "pragmatic" (smooth chordal approach-rate, the default) or "conformal"
    (log|phi(z)| for the Boettcher/Koenigs coordinate, estimated
    numerically -- see cdx.GreensPotential's own doc comment). Ignored
    entirely for a certified polynomial (the two already coincide there)
    and for every other mode.

    `cache`, if given, is consulted BEFORE rendering anything (a hit skips
    computation entirely, including find_attractors for basin mode) and
    populated AFTER, but only with a result that completed -- a cancelled
    render's partial array is never stored, since `cancel` itself (not some
    separate flag the caller has to remember to pass) is what render_map
    checks to decide that. RenderCache.make_key deliberately excludes
    `settings.threads`: it changes how fast this runs, never what it
    produces, so two requests differing only in thread count share a hit.

    Returns a NumPy array (row 0 at the bottom -- see cdx's own orientation
    convention; plot with origin='lower'), except:
      - "julia" returns a STACKED 3D array, shape (2, height, width), for a
        RATIONAL map (Stage 2: has poles, not cdx.polynomial_escape_
        certified) -- index 0 is the smooth chordal approach-rate value (0
        = unresolved), index 1 is which attractor (Cycle.id) each pixel
        reached (0 = unresolved). ONLY index 0 is actually colored
        (through color_escape_time, the SAME palette/scaling pipeline a
        certified polynomial's plain array already goes through -- see
        app/sandbox.py's array_to_qimage, "GOVERNING PRINCIPLE"); index 1
        is kept only for the cursor readout to report which basin a pixel
        is in (see ImageView._sample_at_pixel) -- which basin was reached
        is BASIN mode's own question, not something Julia colors by. For
        a CERTIFIED polynomial (the common case), still a plain 2D array
        -- today's escape-time values, UNCHANGED -- since there is no
        label concept on that path at all (see cdx::Renderer::
        render_julia's own doc comment).
      - "basin" returns a STACKED 3D array, shape (2, height, width):
        index 0 is the label (0 = unresolved, else the basin/cycle id, per
        cdx::Renderer::render_basin), index 1 is the iteration count each
        pixel took to resolve -- for basin SHADING (see app/color.py's
        color_basin). Stacked into one array rather than returned as a
        tuple so byte-accounting stays simple (a single array's .nbytes).
      - "greens"/"parameter_greens" return a plain 2D array -- the
        potential G(z) (or G_M(c) on the parameter plane) -- for a
        CERTIFIED polynomial, UNCHANGED; for a RATIONAL map (Stage 3), a
        STACKED 3D array (2, height, width): index 0 is the potential
        value (selected by `potential`), index 1 is `exact` (1.0 where
        CONFORMAL was genuinely computed, 0.0 where it fell back to
        Pragmatic -- meaningless, always 0, for Pragmatic itself). See
        cdx.Renderer.render_greens/render_parameter_greens's own doc
        comments. Normalized per-pixel at each pixel's own escape/
        attractor-crossing iteration, so unlike the old accumulate/
        degree^max_iter form there is no overflow case and nothing to
        warn about.
      - "parameter_basin" always returns a STACKED 3D array, shape (3,
        height, width): index 0 is the NUMBER OF DISTINCT ATTRACTING
        CYCLES at that parameter (infinity counts as one when a critical
        orbit's limit, and so does any algebraically-attracting fixed
        point complete_attractors' own union recovers -- see its doc
        comment -- even one no critical orbit happened to settle on),
        index 1 is how many of that pixel's critical orbits are explained
        by NO attractor in the COMPLETE set (Siegel/Herman/parabolic --
        tracked separately, never folded into index 0, and RECONCILED
        against the union: an orbit that failed to close numerically but
        actually landed on an injected fixed point is not counted here --
        see cdx.complete_attractors_from_seeds' own doc comment for why
        this reconciliation is what makes index 1 honest rather than the
        critical-seeded pass's raw, pre-union tally), index 2 is how many
        of index 0's cycles are CERTAIN -- algebraically injected fixed
        points, not numerically-discovered closures still subject to a
        finite search budget/tolerance (app.color.color_parameter_basin's
        own coloring policy reads this: a certain attractor is shown as
        confirmed even when index 1 > index 0 overall, since a genuinely
        unrelated residual elsewhere in the same pixel shouldn't make an
        exact, algebraically-known attractor look unconfirmed). Escape-
        radius-free. See cdx.Renderer.render_parameter_basin's own doc
        comment for the method (critical-orbit convergence + chordal
        clustering, reusing complete_attractors_from_seeds -- no per-pixel
        find_attractors/complete_attractors that would redundantly
        re-root-find the same critical points).
      - "parameter_period" always returns a STACKED 3D array, shape (3,
        height, width): index 0 is the PERIOD (measured in the raw
        z-plane, never a symmetry quotient -- see cdx.per_seed_outcomes'
        own doc comment) of the attracting cycle a TRACKED critical orbit
        converges to at that parameter; index 1 is 1.0 where no tracked
        seed resolved within budget (a genuine residual -- Siegel/Herman/
        parabolic -- never a fabricated period, 0.0 elsewhere); index 2 is
        1.0 where the resolved cycle is the point at infinity (kept OUT of
        index 0's own value and its golden-hue color family -- see
        app.color.color_parameter_period), 0.0 elsewhere. At most one of
        indices 1/2 is ever 1.0 for the same pixel. `n`, the family's own
        structural parameter (the power in relaxed-Newton-of-z^n-1), is
        NOT a caller-supplied argument here -- it is recovered as
        round(rational_map.degree(param)), which equals n exactly for
        this family (see cdx.CriticalPointFamily.RelaxedNewtonPower's own
        doc comment) regardless of what `param`/`a` happens to be bound
        to. CURRENTLY SUPPORTS EXACTLY ONE seed family (relaxed Newton of
        z^n-1) -- calling this mode on an unrelated map produces seeds
        that do not correspond to its actual critical points, and the
        result, while not a crash, is not meaningful (see cdx.Renderer.
        render_parameter_period's own doc comment). Escape-radius-free.
    """
    if mode not in RENDER_MODES:
        raise ValueError(f"unknown render mode {mode!r}; must be one of {RENDER_MODES}")

    key = None
    if cache is not None:
        key = make_key(rational_map.serialize(), param, mode, viewport.center, viewport.scale,
                       viewport.resolution, settings.max_iter, settings.escape_radius, settings.tol,
                       potential if mode in ("greens", "parameter_greens") else None)
        cached = cache.get(key)
        if cached is not None:
            return cached

    renderer = cdx.Renderer(map=cdx.Map.custom(rational_map, param), viewport=viewport,
                            settings=settings)
    if mode == "julia":
        # Stage 2: cycles are only computed (a real cost -- find_attractors
        # iterates every critical orbit) when the map genuinely needs them
        # -- a certified polynomial's fast path ignores them entirely, so
        # skipping find_attractors for it isn't just an optimization, it's
        # what keeps that path's performance IDENTICAL to before Stage 2
        # existed (the milestone's own acceptance requirement).
        # The binding ALWAYS returns (values, labels) -- labels is simply
        # all-zero for a certified polynomial (no basin concept on that
        # path at all; see cdx::Renderer::render_julia's own doc comment)
        # -- so unpacking is unconditional; only whether `array` ends up
        # STACKED (rational) or plain (certified polynomial) differs.
        if cdx.polynomial_escape_certified(rational_map):
            values, _labels = renderer.render_julia(cancel)
            array = values
        else:
            cycles = cdx.find_attractors(rational_map, param)
            values, labels = renderer.render_julia(cancel, cycles)
            array = np.stack([values, labels])
    elif mode == "parameter":
        array = renderer.render_parameter(cancel)
    elif mode == "basin":
        # complete_attractors (NOT find_attractors) -- see its own C++ doc
        # comment: the plain critical-seeded search is not a completeness
        # guarantee, so basin classification unions in every algebraically-
        # attracting fixed point RationalMap.fixed_points already knows
        # exactly, the same set the fact sheet's attracting_cycles and
        # Parameter_basin's per-pixel count now agree on too. A real cost
        # for a root-finding-heavy custom map, but only on a cache MISS
        # now -- a repeat request at the same key (map, param, viewport,
        # settings) skips it entirely, same as skipping render_basin itself.
        cycles = cdx.complete_attractors(rational_map, param)
        labels, iterations = renderer.render_basin(cycles, cancel)
        array = np.stack([labels, iterations])
    elif mode == "greens":
        # Same certified-gate reasoning as "julia" above: find_attractors
        # is only paid for when the rational path genuinely needs it.
        pot = _greens_potential(potential)
        if cdx.polynomial_escape_certified(rational_map):
            values, _exact = renderer.render_greens(cancel, potential=pot)
            array = values
        else:
            cycles = cdx.find_attractors(rational_map, param)
            values, exact = renderer.render_greens(cancel, cycles, pot)
            array = np.stack([values, exact])
    elif mode == "parameter_greens":
        # No find_attractors here even for a rational map: the rational
        # path only ever tracks chordal distance to a FIXED infinity
        # attractor (see Renderer::render_parameter_greens' own doc
        # comment for why a per-parameter-pixel root-find isn't needed,
        # or wanted, for this particular question).
        pot = _greens_potential(potential)
        values, exact = renderer.render_parameter_greens(cancel, pot)
        array = values if cdx.polynomial_escape_certified(rational_map) else np.stack([values, exact])
    elif mode == "parameter_basin":
        # No find_attractors/complete_attractors call HERE either:
        # render_parameter_basin does its OWN per-pixel discovery
        # internally (distinct_critical_points at THAT pixel's own
        # parameter, then complete_attractors_from_seeds) -- unlike
        # "basin" above, there is no single fixed `a` to discover
        # attractors for ONCE up front, since every pixel IS a different
        # `a`. Always stacked: (counts, unresolved, certain) -- see
        # cdx.Renderer.render_parameter_basin's own doc comment.
        counts, unresolved, certain = renderer.render_parameter_basin(cancel)
        array = np.stack([counts, unresolved, certain])
    elif mode == "parameter_period":
        # `n` is a FAMILY-structural choice (the power in relaxed-Newton-
        # of-z^n-1), not the map's own single active dynamical parameter
        # -- recovered from the map's own degree rather than threaded
        # through as a separate piece of session state, since for this
        # family map.degree(a) equals n exactly (numerator degree n,
        # denominator degree n-1, after clearing denominators -- see
        # cdx.CriticalPointFamily.RelaxedNewtonPower's own doc comment)
        # for any `a` short of the single degenerate a=n point. Only ONE
        # seed_family exists today; see this function's own docstring for
        # why this mode is currently scoped to that one family's shape.
        n = int(round(rational_map.degree(param)))
        periods, undetermined, is_infinity = renderer.render_parameter_period(
            cdx.CriticalPointFamily.RelaxedNewtonPower, n, cancel)
        array = np.stack([periods, undetermined, is_infinity])
    else:
        raise AssertionError(f"unreachable: mode={mode!r}")

    if cache is not None and (cancel is None or not cancel.is_cancelled):
        cache.put(key, array)
    return array


class Session:
    """Sandbox session: SHARED state only -- the current map, its bound
    parameter, render settings, the family library, and the render cache.
    Deliberately does NOT own a viewport or a render mode: those are now
    per-Pane (see app.pane.Pane's own docstring for why), since two panes
    showing different planes of the same map need two independent ones.
    Every RENDERED PIXEL still ultimately comes from map/param/
    render_settings here plus whichever pane's own viewport/mode asked
    for it -- this class holds what stays the same no matter how many
    panes are looking at it.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.map: cdx.RationalMap = cdx.RationalMap.mandelbrot()
        self.param: complex = DEFAULT_JULIA_PARAM
        self._settings = settings if settings is not None else Settings()
        self.render_settings: cdx.RenderSettings = cdx.RenderSettings(
            self._settings.max_iter, self._settings.escape_radius,
            self._settings.tol, self._settings.threads)
        self.library: cdx.FamilyLibrary = cdx.FamilyLibrary.with_defaults()
        # One cache for this session's lifetime, shared by every render --
        # foreground (render() below) and background (app/sandbox.py's
        # RenderTask, which is handed this same instance). A resolution or
        # setting change makes old entries unreachable, not wrong; they age
        # out under the byte budget rather than needing an explicit flush
        # (see app.render_cache's own docstring).
        self.cache: RenderCache = RenderCache(self._settings.cache_budget_bytes)

    @property
    def settings(self) -> Settings:
        """The Settings this session is currently rendering with -- kept in
        sync by apply_settings() below, the only way this changes after
        construction (render_settings/cache.budget aren't mutated directly
        by anything else in this class; a pane's own viewport.resolution
        is the CALLER's job to update per-pane -- see app/sandbox.py's
        _on_settings_applied, which loops over every pane -- since this
        class no longer owns any one viewport to update).
        """
        return self._settings

    def apply_settings(self, settings: Settings) -> None:
        """Applies a (validated -- see app.settings.validate_field, used by
        the Settings panel before ever calling this) Settings to this
        session: render_settings and the cache's byte budget. Does NOT
        touch resolution -- that is now a per-pane viewport field (see
        app.pane.Pane), so updating it for every visible pane is the
        CALLER's job (app/sandbox.py's _on_settings_applied loops over
        every pane), not something this class can do on its own anymore.
        Does NOT clear the cache either; see RenderCache's own docstring
        for why a resolution/setting change should age old entries out
        under the budget rather than flush them.
        """
        self._settings = settings
        self.render_settings = cdx.RenderSettings(settings.max_iter, settings.escape_radius,
                                                  settings.tol, settings.threads)
        self.cache.set_budget(settings.cache_budget_bytes)

    def render(self, viewport: cdx.Viewport, render_mode: str,
              cancel: cdx.CancelToken | None = None):
        """Renders THIS session's map/parameter/render_settings at the
        given viewport and mode (typically a pane's own -- see
        app.pane.Pane). Explicit rather than reading a self.viewport/
        self.render_mode this class no longer has; see render_map() above
        for the return shape, this is just that function applied to this
        session's own map/param/settings/cache.
        """
        return render_map(self.map, self.param, viewport, self.render_settings,
                          render_mode, cancel=cancel, cache=self.cache,
                          potential=self._settings.greens_potential)

    # ---- equation editing (Stage 4 of the P/Q milestone) ------------------------
    # The equation panel's own "Apply"/"Add Pole"/"Add Zero" actions, each a
    # thin wrapper over the P/Q-backed RationalMap operations Stage 2/3 added
    # (from_expression, add_pole_at, add_zero_at) -- see their own doc
    # comments (rational.hpp) for the actual math. Every one of these
    # replaces or mutates self.map only on success; on failure it raises
    # ValueError (from_expression's own C++ exception, translated by
    # pybind11) and self.map is left exactly as it was, matching the
    # existing term-editing methods' own "reject, don't half-apply"
    # convention below.
    def build_pq_map(self, source: str, active_param: str, fixed_values: dict[str, complex],
                     name: str | None = None) -> None:
        """Parses `source` and replaces self.map wholesale with a fresh
        P/Q-backed RationalMap (cdx.RationalMap.from_expression) -- see its
        own doc comment for what `active_param`/`fixed_values` mean and
        when each raises ValueError. `name` defaults to the CURRENT map's
        own name (editing a formula doesn't rename it), not "untitled".
        """
        target_name = name if name is not None else self.map.name
        self.map = cdx.RationalMap.from_expression(source, active_param, fixed_values, target_name)

    def add_pq_pole(self, location: complex) -> None:
        """P/Q-backed analogue of add_pole_term: multiplies the CURRENT
        map's own denominator by (z-location) -- see RationalMap.
        add_pole_at's own doc comment. If self.map is not YET P/Q-backed
        (e.g. a built-in preset just loaded, never edited through the
        equation field), it is converted first via its own to_formula() --
        exact for a term-based map, since to_formula() is a faithful
        reconstruction and every built-in preset has at most one parameter
        (from_expression's own auto-active rule applies cleanly). Raises
        ValueError if that implicit conversion itself somehow fails (not
        expected for any preset, but not assumed either); otherwise never
        raises (add_pole_at itself only raises for a NON-P/Q-backed map,
        which this method never leaves self.map as by the time it calls
        it).
        """
        if not self.map.is_pq_backed():
            self.build_pq_map(self.map.to_formula(), "", {})
        self.map.add_pole_at(location)

    def add_pq_zero(self, location: complex) -> None:
        """Same as add_pq_pole, multiplying the numerator instead (see
        RationalMap.add_zero_at's own doc comment).
        """
        if not self.map.is_pq_backed():
            self.build_pq_map(self.map.to_formula(), "", {})
        self.map.add_zero_at(location)

    # ---- term editing ----------------------------------------------------------
    # Thin wrappers over RationalMap's own term operations. poly_terms()/
    # pole_terms() are already live-mutable (see bindings.cpp's opaque
    # vector binding), so in-place edits like
    #   session.map.poly_terms()[0].enabled = False
    # already work directly; edit_poly_term/edit_pole_term below exist for
    # the common case of setting several fields at once by index.
    def add_poly_term(self, coeff: complex, exponent: int, param_power: int = 0,
                      label: str = "") -> int:
        """Raises ValueError for exponent < 0 -- that represents a pole, and
        every pole must go through add_pole_term instead, so there is
        exactly one representation for "a pole at a location" (see
        RationalMap::add_poly's own comment). The map is left unchanged on
        rejection; the caller (e.g. the term editor) should catch this and
        show the message inline, not let it propagate as a crash.
        """
        return self.map.add_poly(coeff, exponent, param_power, label)

    def add_pole_term(self, location: complex, strength: complex, order: int = 1,
                      param_power: int = 0, label: str = "") -> int:
        """Raises ValueError if `location` coincides with an existing
        enabled pole -- either another pole term's location, or (for a map
        built before add_poly's own restriction above, e.g. one loaded from
        an older saved family) a still-present negative-exponent poly term,
        which only ever means a pole at the origin (see RationalMap::
        add_pole's own comment for why two representations of what is
        conceptually one pole are rejected rather than silently combined).
        The map is left unchanged on rejection. NOTE: this only guards
        ADDING a new pole -- editing an EXISTING term's location via
        edit_pole_term below to collide with another is not (yet) caught
        here; out of scope for what "reject a second pole" was asked to
        cover.
        """
        return self.map.add_pole(location, strength, order, param_power, label)

    def remove_poly_term(self, index: int) -> None:
        self.map.remove_poly(index)

    def remove_pole_term(self, index: int) -> None:
        self.map.remove_pole(index)

    def edit_poly_term(self, index: int, **fields) -> None:
        """Sets attributes (coeff, exponent, param_power, enabled, label) on
        poly_terms()[index]. Raises IndexError for an out-of-range index
        (unlike remove_poly_term, which RationalMap defines as a no-op --
        editing a term that does not exist has no sensible silent meaning).
        """
        terms = self.map.poly_terms()
        if not 0 <= index < len(terms):
            raise IndexError(f"poly term index {index} out of range (0..{len(terms) - 1})")
        term = terms[index]
        for name, value in fields.items():
            setattr(term, name, value)

    def edit_pole_term(self, index: int, **fields) -> None:
        """Same as edit_poly_term, for pole_terms(): location, strength,
        order, param_power, enabled, location_is_param, label.
        """
        terms = self.map.pole_terms()
        if not 0 <= index < len(terms):
            raise IndexError(f"pole term index {index} out of range (0..{len(terms) - 1})")
        term = terms[index]
        for name, value in fields.items():
            setattr(term, name, value)

    def move_poly_term(self, index: int, direction: int) -> bool:
        """Swaps poly_terms()[index] with its neighbor at index+direction
        (direction must be -1 or +1). Returns False (no-op) if that
        neighbor doesn't exist -- index is already at that end of the list.

        Deliberately NOT `terms[i], terms[j] = terms[j], terms[i]`: the
        opaque-vector binding's __getitem__ returns a LIVE reference into
        the underlying vector, not a snapshot copy (that's what makes
        `poly_terms()[0].enabled = False` work at all), so that idiom's
        usual "safe simultaneous swap" guarantee does not hold here -- the
        second assignment reads through a reference to a slot the first
        assignment already overwrote, silently duplicating one term into
        both slots instead of swapping them. Snapshotting every field into
        plain Python values (via getattr, which DOES copy out a value)
        before writing any of them back avoids that trap.
        """
        return _swap_terms(self.map.poly_terms(), index, index + direction, _POLY_TERM_FIELDS)

    def move_pole_term(self, index: int, direction: int) -> bool:
        """Same as move_poly_term, for pole_terms()."""
        return _swap_terms(self.map.pole_terms(), index, index + direction, _POLE_TERM_FIELDS)

    # ---- library -----------------------------------------------------------------
    def save_to_library(self, name: str | None = None) -> None:
        """Adds (or replaces, by name) the current map in the session's
        library. Renaming the map itself when a name is given. Raises
        ValueError for a PRESET_FAMILY_NAMES target -- the six built-ins
        are read-only, and this is the one path that could otherwise
        silently overwrite one (e.g. loading "mandelbrot", tweaking it,
        then saving without renaming).
        """
        target = name if name is not None else self.map.name
        if target in PRESET_FAMILY_NAMES:
            raise ValueError(f"{target!r} is a built-in preset; save under a different name")
        if name is not None:
            self.map.name = name
        self.library.add(self.map)

    def load_from_library(self, name: str) -> None:
        """Makes `name` the current map. Raises KeyError if not found --
        FamilyLibrary.find returns a COPY, so the loaded map is independent
        of whatever is still stored in the library.
        """
        found = self.library.find(name)
        if found is None:
            raise KeyError(f"no map named {name!r} in the library")
        self.map = found

    def rename_in_library(self, old_name: str, new_name: str) -> None:
        """Renames a library entry IN THE LIBRARY -- does not touch
        self.map, even if self.map happens to be a copy of that same
        family; renaming a saved entry and editing what is currently
        loaded are different actions. Raises ValueError for a preset name
        (either side: renaming FROM one, since presets don't move, or TO
        one, since that would shadow it) or a new_name that collides with
        an existing entry; KeyError if old_name isn't in the library.
        """
        if old_name in PRESET_FAMILY_NAMES:
            raise ValueError(f"{old_name!r} is a built-in preset and cannot be renamed")
        found = self.library.find(old_name)
        if found is None:
            raise KeyError(f"no map named {old_name!r} in the library")
        if new_name in PRESET_FAMILY_NAMES:
            raise ValueError(f"{new_name!r} is a built-in preset name and cannot be reused")
        if new_name != old_name and self.library.find(new_name) is not None:
            raise ValueError(f"a family named {new_name!r} already exists")
        found.name = new_name
        self.library.add(found)
        self.library.remove(old_name)

    def delete_from_library(self, name: str) -> None:
        """Raises ValueError for a preset name, KeyError if name isn't in
        the library at all.
        """
        if name in PRESET_FAMILY_NAMES:
            raise ValueError(f"{name!r} is a built-in preset and cannot be deleted")
        if not self.library.remove(name):
            raise KeyError(f"no map named {name!r} in the library")

    def set_library_notes(self, name: str, notes: str) -> None:
        """Edits an entry's notes field IN THE LIBRARY, independent of
        self.map. FamilyLibrary.find returns a COPY (see load_from_library's
        own docstring), so this reads the entry, mutates the copy, and
        writes it back via add() (which replaces by name) rather than
        mutating something that was never actually stored. Raises
        ValueError for a preset name, KeyError if name isn't in the library.
        """
        if name in PRESET_FAMILY_NAMES:
            raise ValueError(f"{name!r} is a built-in preset; its notes cannot be edited")
        found = self.library.find(name)
        if found is None:
            raise KeyError(f"no map named {name!r} in the library")
        found.notes = notes
        self.library.add(found)

    def save_library_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.library.serialize())

    def load_library_file(self, path: str) -> None:
        with open(path) as f:
            text = f.read()
        self.library = cdx.FamilyLibrary.deserialize(text)

    def save_user_library(self, path: str) -> None:
        """Persists only the NON-preset entries -- the six built-ins are
        always available via FamilyLibrary.with_defaults() (see __init__)
        and are never written out, so this file only ever grows with what
        the user actually saved, and reloading it (load_user_library)
        never needs to reconcile a stale copy of a preset against the
        real one.
        """
        user_only = cdx.FamilyLibrary()
        for entry_name in self.library.names():
            if entry_name in PRESET_FAMILY_NAMES:
                continue
            entry = self.library.find(entry_name)
            if entry is not None:
                user_only.add(entry)
        with open(path, "w") as f:
            f.write(user_only.serialize())

    def load_user_library(self, path: str) -> None:
        """MERGES a previously-saved user library file into self.library
        (which already has the six presets from with_defaults() -- see
        __init__), rather than replacing it wholesale the way
        load_library_file does -- so presets are never at the mercy of
        what's in this file. Never raises: a missing OR malformed file
        just means "no (usable) saved user families yet," the same
        treatment app.settings.load_settings gives a missing or malformed
        settings.json.
        """
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            return
        try:
            loaded = cdx.FamilyLibrary.deserialize(text)
        except ValueError:
            return
        for entry_name in loaded.names():
            if entry_name in PRESET_FAMILY_NAMES:
                continue   # a preset name in a USER file is stale/foreign data; never trust it
            entry = loaded.find(entry_name)
            if entry is not None:
                self.library.add(entry)

    # ---- experiment snapshots ---------------------------------------------------
    # A full reproducible EXPERIMENT -- map + parameter + view + mode + render
    # settings + orbit -- NOT a saved MAP. Deliberately separate from the
    # library above (save_to_library/FamilyLibrary), which stores named maps
    # only and has read-only presets; a snapshot restores the map alongside
    # everything else, but never touches the library itself.
    def snapshot_to_dict(self, panes: list[tuple[cdx.Viewport, str]], focused_index: int,
                         coupled: bool, orbit: tuple[int, complex, int] | None = None,
                         preview_png_base64: str | None = None) -> dict:
        """Assembles a JSON-serializable dict capturing everything needed to
        reconstruct exactly what is currently on screen -- EVERY pane, not
        just one (Stage 4). `panes` is one (viewport, render_mode) pair per
        pane, in the SAME order the caller also uses to interpret
        `focused_index` and orbit's own pane index -- this class owns
        neither a viewport, a render_mode, nor a notion of "panes" itself
        (see app.pane.Pane), so all of it is the caller's own state handed
        in explicitly. `orbit`, if given, is (pane_index, z0, n) -- WHICH
        pane the orbit belongs to (orbit tracking is per-pane -- see
        app.sandbox.ImageView.orbit_tracker -- so this is the one place a
        plain (z0, n) pair alone would be ambiguous), plus OrbitState's own
        two independent fields; z and history are DERIVED from stepping z0
        n times (see restore_from_snapshot), so storing more would just be
        redundant, recomputable state. The param MARKER isn't stored at
        all: it's just session.param drawn on whichever panes are
        currently parameter-plane (see ImageView._param_marker_pixel), and
        param is already captured below -- nothing else to keep in sync.

        `preview_png_base64` (Stage C) is an OPAQUE string as far as this
        class is concerned -- Session has no Qt/rendering dependency (see
        this module's own imports), so the caller (app.sandbox) renders a
        clean thumbnail and base64-encodes it before calling this; this
        method just stores whatever it's given, or None.
        """
        rs = self.render_settings
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "app_version": VERSION,
            "map": self.map.serialize(),
            "param": [self.param.real, self.param.imag],
            "render_settings": {
                "max_iter": rs.max_iter,
                "escape_radius": rs.escape_radius,
                "tol": rs.tol,
                "threads": rs.threads,
            },
            "layout": {
                "coupled": coupled,
                "focused_index": focused_index,
                "panes": [
                    {
                        "render_mode": mode,
                        "viewport": {
                            "center": [vp.center.real, vp.center.imag],
                            "scale": vp.scale,
                            "resolution": vp.resolution,
                        },
                    }
                    for vp, mode in panes
                ],
            },
            "orbit": None if orbit is None else {
                "pane_index": orbit[0],
                "z0": [orbit[1].real, orbit[1].imag],
                "n": orbit[2],
            },
            "preview": preview_png_base64,
        }

    def restore_from_snapshot(self, data: dict) -> tuple[list[tuple[cdx.Viewport, str]], int, bool,
                                                          tuple[int, complex, int] | None,
                                                          str | None]:
        """The inverse of snapshot_to_dict. Fully validates and parses
        EVERY field -- every pane's own viewport/render_mode included --
        before mutating any live state -- a malformed or wrong-version
        snapshot raises ValueError and leaves this Session byte-for-byte
        unchanged, the same raise-don't-swallow treatment load_library_file
        already gives an explicit-path load (as opposed to
        load_user_library's silent-degrade treatment for the merge-at-
        startup path, which this is not). A version-1 (pre-Stage-4,
        single-pane) snapshot is REJECTED here, not migrated -- see
        SNAPSHOT_SCHEMA_VERSION's own comment for why. A version-2
        (pre-Stage-C, no "preview" key) snapshot, by contrast, is accepted
        -- see MIN_SNAPSHOT_SCHEMA_VERSION's own comment.

        Only map/param/render_settings are actually SET on this Session --
        it has no panes, viewport, render_mode, or OrbitTracker of its own
        to assign/seed. Returns (panes, focused_index, coupled, orbit,
        preview) for the caller to rebuild its own panes (viewport/
        render_mode each), layout (coupled flag, which pane is focused),
        tracker (orbit, as (pane_index, z0, n) or None): reconstructing an
        orbit needs to go through the tracker's own seed()/step() so
        z/history regenerate through the exact same code path a live orbit
        does, not a Session-level shortcut around it -- and preview (the
        base64 PNG string a version-3 file may carry, or None for a
        version-2 file / a version-3 one saved without one).
        """
        if not isinstance(data, dict):
            raise ValueError("experiment snapshot must be a JSON object")
        version = data.get("schema_version")
        if not isinstance(version, int) or not (
                MIN_SNAPSHOT_SCHEMA_VERSION <= version <= SNAPSHOT_SCHEMA_VERSION):
            raise ValueError(
                f"unsupported experiment snapshot schema_version {version!r} "
                f"(expected {MIN_SNAPSHOT_SCHEMA_VERSION}-{SNAPSHOT_SCHEMA_VERSION})")
        try:
            rational_map = cdx.RationalMap.deserialize(data["map"])
            param = complex(*data["param"])
            rs_data = data["render_settings"]
            render_settings = cdx.RenderSettings(int(rs_data["max_iter"]),
                                                 float(rs_data["escape_radius"]),
                                                 float(rs_data["tol"]), int(rs_data["threads"]))

            layout = data["layout"]
            coupled = bool(layout["coupled"])
            focused_index = int(layout["focused_index"])
            panes_data = layout["panes"]
            if not isinstance(panes_data, list) or len(panes_data) == 0:
                raise ValueError("layout.panes must be a non-empty list")
            panes: list[tuple[cdx.Viewport, str]] = []
            for pane_data in panes_data:
                mode = pane_data["render_mode"]
                if mode not in RENDER_MODES:
                    raise ValueError(f"unknown render_mode {mode!r}")
                vp_data = pane_data["viewport"]
                viewport = cdx.Viewport(complex(*vp_data["center"]), float(vp_data["scale"]),
                                        int(vp_data["resolution"]))
                panes.append((viewport, mode))
            if not (0 <= focused_index < len(panes)):
                raise ValueError(
                    f"focused_index {focused_index} out of range for {len(panes)} panes")

            orbit_data = data.get("orbit")
            if orbit_data is None:
                orbit = None
            else:
                orbit_pane_index = int(orbit_data["pane_index"])
                if not (0 <= orbit_pane_index < len(panes)):
                    raise ValueError(
                        f"orbit pane_index {orbit_pane_index} out of range for {len(panes)} panes")
                orbit = (orbit_pane_index, complex(*orbit_data["z0"]), int(orbit_data["n"]))
            preview = data.get("preview")
            if preview is not None and not isinstance(preview, str):
                raise ValueError("preview must be a string (base64 PNG) or absent/null")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed experiment snapshot: {exc}") from exc

        # Only now, once everything above has parsed successfully, mutate
        # live state -- an error partway through would otherwise leave the
        # session half-updated (new map, old settings), which is worse than
        # rejecting the whole file outright.
        self.map = rational_map
        self.param = param
        self.render_settings = render_settings
        return panes, focused_index, coupled, orbit, preview

    def save_snapshot(self, path: str, panes: list[tuple[cdx.Viewport, str]], focused_index: int,
                      coupled: bool, orbit: tuple[int, complex, int] | None = None,
                      preview_png_base64: str | None = None) -> None:
        with open(path, "w") as f:
            json.dump(self.snapshot_to_dict(panes, focused_index, coupled, orbit,
                                            preview_png_base64), f, indent=2)
            f.write("\n")

    def load_snapshot(self, path: str) -> tuple[list[tuple[cdx.Viewport, str]], int, bool,
                                                 tuple[int, complex, int] | None, str | None]:
        """Raises ValueError for a missing/unreadable file (OSError is NOT
        caught here -- unlike load_user_library's silent-degrade startup
        path, an explicit Open Experiment... action that fails should tell
        the user why, not quietly do nothing) or malformed JSON
        (json.JSONDecodeError is itself a ValueError subclass) or a
        malformed/wrong-version snapshot (restore_from_snapshot's own
        checks) alike.
        """
        with open(path) as f:
            data = json.load(f)
        return self.restore_from_snapshot(data)

    # ---- data extraction -----------------------------------------------------------
    def dynamical_facts(self, opts: cdx.FindAttractorsOptions | None = None) -> cdx.DynamicalFacts:
        """The dynamical facts of the CURRENT map at the CURRENT parameter:
        degree, critical points (with multiplicity), attracting cycles
        (with period and multiplier), pole locations and orders, and fixed
        points. See cdx.DynamicalFacts's fields for the exact shape.
        """
        if opts is None:
            return cdx.dynamical_facts(self.map, self.param)
        return cdx.dynamical_facts(self.map, self.param, opts)

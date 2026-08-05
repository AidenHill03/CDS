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

import cdx
from app.render_cache import RenderCache, make_key
from app.settings import Settings

RENDER_MODES = ("julia", "parameter", "basin", "greens")

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


def render_map(rational_map: cdx.RationalMap, param: complex, viewport: cdx.Viewport,
               settings: cdx.RenderSettings, mode: str,
               cancel: cdx.CancelToken | None = None,
               cache: RenderCache | None = None):
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

    `cache`, if given, is consulted BEFORE rendering anything (a hit skips
    computation entirely, including find_attractors for basin mode) and
    populated AFTER, but only with a result that completed -- a cancelled
    render's partial array is never stored, since `cancel` itself (not some
    separate flag the caller has to remember to pass) is what render_map
    checks to decide that. RenderCache.make_key deliberately excludes
    `settings.threads`: it changes how fast this runs, never what it
    produces, so two requests differing only in thread count share a hit.

    Returns a NumPy array (row 0 at the bottom -- see cdx's own orientation
    convention; plot with origin='lower').
    """
    if mode not in RENDER_MODES:
        raise ValueError(f"unknown render mode {mode!r}; must be one of {RENDER_MODES}")

    key = None
    if cache is not None:
        key = make_key(rational_map.serialize(), param, mode, viewport.center, viewport.scale,
                       viewport.resolution, settings.max_iter, settings.escape_radius, settings.tol)
        cached = cache.get(key)
        if cached is not None:
            return cached

    renderer = cdx.Renderer(map=cdx.Map.custom(rational_map, param), viewport=viewport,
                            settings=settings)
    if mode == "julia":
        array = renderer.render_julia(cancel)
    elif mode == "parameter":
        array = renderer.render_parameter(cancel)
    elif mode == "basin":
        # find_attractors is a real cost for a root-finding-heavy custom
        # map, but only on a cache MISS now -- a repeat request at the same
        # key (map, param, viewport, settings) skips it entirely, same as
        # skipping render_basin itself.
        cycles = cdx.find_attractors(rational_map, param)
        array = renderer.render_basin(cycles, cancel)
    elif mode == "greens":
        array, _normalized = renderer.render_greens(cancel=cancel)
    else:
        raise AssertionError(f"unreachable: mode={mode!r}")

    if cache is not None and (cancel is None or not cancel.is_cancelled):
        cache.put(key, array)
    return array


class Session:
    """Sandbox session: current map + parameter + viewport + render mode,
    term editing, a family library, and dynamical-facts extraction.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.map: cdx.RationalMap = cdx.RationalMap.mandelbrot()
        self.param: complex = DEFAULT_JULIA_PARAM
        self._settings = settings if settings is not None else Settings()
        # Starts on the PARAMETER PLANE (the Mandelbrot set): unlike the
        # filled disc z^2+0 gives in julia mode, it teaches the tool by
        # itself -- clicking around it is the natural first interaction.
        self.viewport: cdx.Viewport = cdx.Viewport(DEFAULT_PARAMETER_VIEW_CENTER,
                                                    DEFAULT_PARAMETER_VIEW_SCALE,
                                                    self._settings.resolution)
        self.render_settings: cdx.RenderSettings = cdx.RenderSettings(
            self._settings.max_iter, self._settings.escape_radius,
            self._settings.tol, self._settings.threads)
        self.render_mode: str = "parameter"
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
        construction (viewport.resolution/render_settings/cache.budget
        aren't mutated directly by anything else in this class).
        """
        return self._settings

    def apply_settings(self, settings: Settings) -> None:
        """Applies a (validated -- see app.settings.validate_field, used by
        the Settings panel before ever calling this) Settings to this
        session: render_settings, the viewport's resolution (keeping its
        current center/scale -- Settings does not own WHERE the user is
        looking, only how it's rendered), and the cache's byte budget.
        Does NOT clear the cache; see RenderCache's own docstring for why a
        resolution/setting change should age old entries out under the
        budget rather than flush them.
        """
        self._settings = settings
        vp = self.viewport
        self.viewport = cdx.Viewport(vp.center, vp.scale, settings.resolution)
        self.render_settings = cdx.RenderSettings(settings.max_iter, settings.escape_radius,
                                                  settings.tol, settings.threads)
        self.cache.set_budget(settings.cache_budget_bytes)

    # ---- render mode ---------------------------------------------------------
    def set_render_mode(self, mode: str) -> None:
        if mode not in RENDER_MODES:
            raise ValueError(f"unknown render mode {mode!r}; must be one of {RENDER_MODES}")
        self.render_mode = mode

    def render(self):
        """Renders the current map/parameter/viewport in the current mode.
        See render_map() above for the return shape; this is just that
        function applied to the session's own current state.
        """
        return render_map(self.map, self.param, self.viewport, self.render_settings,
                          self.render_mode, cache=self.cache)

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
        library. Renaming the map itself when a name is given.
        """
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

    def save_library_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.library.serialize())

    def load_library_file(self, path: str) -> None:
        with open(path) as f:
            text = f.read()
        self.library = cdx.FamilyLibrary.deserialize(text)

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

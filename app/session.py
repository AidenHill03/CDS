"""app/session.py -- sandbox session state.

Owns the mutable state a user is actually operating on: the current map,
its bound parameter, the view window, which render mode is selected, and a
library of saved maps. The engine (cdx/) stays stateless apart from
Renderer's own configuration -- see ARCHITECTURE.md's layering rules. This
module holds no mathematics of its own; every operation here is a thin
wrapper delegating straight to a cdx call.

Requires the cdx extension module to be importable, e.g.:

    PYTHONPATH=../cdx/build python session.py

(run from this directory), or add cdx/build to PYTHONPATH some other way.
"""

from __future__ import annotations

import cdx

RENDER_MODES = ("julia", "parameter", "basin", "greens")

# Default full-render resolution (display pixels per side, before overscan --
# see app.sandbox's FULL_OVERSCAN_FACTOR/PREVIEW_OVERSCAN_FACTOR for how the
# actual rendered buffer ends up larger than this). Chosen by measuring: the
# largest resolution whose full render -- already at the 1.3x overscan
# factor every full render now uses, in "parameter" mode (the startup
# default -- see render_mode below) -- stayed under ~0.4s (median of 9 runs)
# for z^2+a on the development machine.
#
# The number that first came out of this measurement (~1100) used the WRONG
# map: cdx.Map(Family.Quadratic, a), the hardcoded fast path. Session always
# renders through cdx.Map.custom(self.map, ...) (self.map is a RationalMap,
# even for the "mandelbrot" preset -- see load_from_library/the map field
# below), which goes through RationalMap's generic term evaluator instead --
# measured at roughly 5-10x the cost per pixel. Re-measuring against THAT
# path (what actually renders) put the real budget-respecting default
# around 120, not 1100. See the P5a-final commit message for the numbers
# and for a note on where that per-pixel cost actually goes -- there is a
# concrete, currently-unexploited optimization available there (an
# unnecessary full root-find of the critical point on EVERY pixel of a
# parameter-plane render, even for shapes like z^2+a where it's the
# parameter-independent constant 0), but fixing RationalMap's evaluator is
# out of scope for this milestone, which only picks a default that respects
# what the engine can do TODAY.
DEFAULT_RESOLUTION = 120

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


def render_map(rational_map: cdx.RationalMap, param: complex, viewport: cdx.Viewport,
               settings: cdx.RenderSettings, mode: str,
               cancel: cdx.CancelToken | None = None):
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

    Returns a NumPy array (row 0 at the bottom -- see cdx's own orientation
    convention; plot with origin='lower').
    """
    if mode not in RENDER_MODES:
        raise ValueError(f"unknown render mode {mode!r}; must be one of {RENDER_MODES}")
    renderer = cdx.Renderer(map=cdx.Map.custom(rational_map, param), viewport=viewport,
                            settings=settings)
    if mode == "julia":
        return renderer.render_julia(cancel)
    if mode == "parameter":
        return renderer.render_parameter(cancel)
    if mode == "basin":
        # Recomputed on every call rather than cached: find_attractors is a
        # real cost for a root-finding-heavy custom map, but there is no
        # cache-invalidation machinery here to get wrong, and nothing so far
        # has needed one. Revisit if profiling says so.
        cycles = cdx.find_attractors(rational_map, param)
        return renderer.render_basin(cycles, cancel)
    if mode == "greens":
        array, _normalized = renderer.render_greens(cancel=cancel)
        return array
    raise AssertionError(f"unreachable: mode={mode!r}")


class Session:
    """Sandbox session: current map + parameter + viewport + render mode,
    term editing, a family library, and dynamical-facts extraction.
    """

    def __init__(self) -> None:
        self.map: cdx.RationalMap = cdx.RationalMap.mandelbrot()
        self.param: complex = DEFAULT_JULIA_PARAM
        # Starts on the PARAMETER PLANE (the Mandelbrot set): unlike the
        # filled disc z^2+0 gives in julia mode, it teaches the tool by
        # itself -- clicking around it is the natural first interaction.
        self.viewport: cdx.Viewport = cdx.Viewport(DEFAULT_PARAMETER_VIEW_CENTER,
                                                    DEFAULT_PARAMETER_VIEW_SCALE,
                                                    DEFAULT_RESOLUTION)
        self.render_settings: cdx.RenderSettings = cdx.RenderSettings()
        self.render_mode: str = "parameter"
        self.library: cdx.FamilyLibrary = cdx.FamilyLibrary.with_defaults()

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
                          self.render_mode)

    # ---- term editing ----------------------------------------------------------
    # Thin wrappers over RationalMap's own term operations. poly_terms()/
    # pole_terms() are already live-mutable (see bindings.cpp's opaque
    # vector binding), so in-place edits like
    #   session.map.poly_terms()[0].enabled = False
    # already work directly; edit_poly_term/edit_pole_term below exist for
    # the common case of setting several fields at once by index.
    def add_poly_term(self, coeff: complex, exponent: int, param_power: int = 0,
                      label: str = "") -> int:
        return self.map.add_poly(coeff, exponent, param_power, label)

    def add_pole_term(self, location: complex, strength: complex, order: int = 1,
                      param_power: int = 0, label: str = "") -> int:
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

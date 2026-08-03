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


class Session:
    """Sandbox session: current map + parameter + viewport + render mode,
    term editing, a family library, and dynamical-facts extraction.
    """

    def __init__(self) -> None:
        self.map: cdx.RationalMap = cdx.RationalMap.mandelbrot()
        self.param: complex = 0j
        self.viewport: cdx.Viewport = cdx.Viewport()
        self.render_settings: cdx.RenderSettings = cdx.RenderSettings()
        self.render_mode: str = "julia"
        self.library: cdx.FamilyLibrary = cdx.FamilyLibrary.with_defaults()

    # ---- render mode ---------------------------------------------------------
    def set_render_mode(self, mode: str) -> None:
        if mode not in RENDER_MODES:
            raise ValueError(f"unknown render mode {mode!r}; must be one of {RENDER_MODES}")
        self.render_mode = mode

    def render(self):
        """Renders the current map/parameter/viewport in the current mode.
        Returns a NumPy array (row 0 at the bottom -- see cdx's own
        orientation convention; plot with origin='lower').
        """
        renderer = cdx.Renderer(
            map=cdx.Map.custom(self.map, self.param),
            viewport=self.viewport,
            settings=self.render_settings,
        )
        if self.render_mode == "julia":
            return renderer.render_julia()
        if self.render_mode == "parameter":
            return renderer.render_parameter()
        if self.render_mode == "basin":
            # Recomputed on every call rather than cached: find_attractors
            # is a real cost for a root-finding-heavy custom map, but there
            # is no cache-invalidation machinery here to get wrong, and
            # nothing so far has needed one. Revisit if profiling says so.
            cycles = cdx.find_attractors(self.map, self.param)
            return renderer.render_basin(cycles)
        if self.render_mode == "greens":
            array, _normalized = renderer.render_greens()
            return array
        raise AssertionError(f"unreachable: render_mode={self.render_mode!r}")

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

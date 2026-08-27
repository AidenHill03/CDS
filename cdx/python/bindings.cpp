// =============================================================================
// bindings.cpp -- pybind11 bindings for the cdx rendering core.
//
// Exposes Map, Viewport, RenderSettings, Cycle and Renderer to Python, with
// render methods returning NumPy arrays. The array shares the Image's buffer
// through a capsule that owns the moved-from Image, so there is no copy of
// the pixel data on the way out -- important because at 2000x2000 an image is
// 32 MB and copying it would dominate the render time we worked to reduce.
//
// Also exposes the sandbox pieces -- RationalMap (with its PolyTerm/PoleTerm
// term lists live-mutable from Python, not just readable), FamilyLibrary and
// Expr -- so a custom map can be built, edited, saved and rendered without
// leaving Python. Map::custom() wraps a RationalMap as a renderable Map,
// same as the six built-in families.
//
// Build:  pip install pybind11 && cmake --build build
// Use:    import cdx
// =============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "cdx/renderer.hpp"
#include "cdx/expr.hpp"
#include "cdx/analysis.hpp"

#include <atomic>
#include <memory>
#include <stdexcept>

// Opaque containers: without this, pybind11/stl.h would convert
// poly_terms()/pole_terms() into a fresh Python list of COPIES on every
// access, silently breaking in-place edits like
// `rmap.poly_terms()[0].enabled = False`. Must come before PYBIND11_MODULE
// and before any implicit std::vector<PolyTerm>/<PoleTerm> conversion.
PYBIND11_MAKE_OPAQUE(std::vector<cdx::PolyTerm>);
PYBIND11_MAKE_OPAQUE(std::vector<cdx::PoleTerm>);

namespace py = pybind11;
using namespace cdx;

namespace {

// Wrap an Image as a NumPy array WITHOUT copying: the array's base is a
// capsule owning a heap-allocated Image, so the buffer stays alive exactly as
// long as Python references it.
py::array_t<double> image_to_numpy(Image&& img) {
    auto held = std::make_unique<Image>(std::move(img));
    const int h = held->height, w = held->width;
    double* ptr = held->data.data();

    py::capsule owner(held.release(), [](void* p) {
        delete reinterpret_cast<Image*>(p);
    });

    // shape (rows, cols); row 0 is the BOTTOM of the view, matching
    // Viewport::coord, so plot with origin='lower'.
    return py::array_t<double>(
        {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)},
        {static_cast<py::ssize_t>(sizeof(double) * w),
         static_cast<py::ssize_t>(sizeof(double))},
        ptr, owner);
}

// Python-side handle for a render's cancellation flag. Renderer's render_*
// methods take a raw const std::atomic<bool>*, which Python cannot hold
// directly; this wraps one so a Python caller can create it, hand it to a
// render call, and set it from anywhere (another thread, typically) while
// that call is running. shared_ptr-held (see the py::class_ registration
// below) rather than the pybind11 default unique_ptr: a token is meant to
// be passed around and outlive the exact scope that created it -- held by
// the render call via a raw pointer into this object for the call's
// duration, and potentially also still referenced from Python (e.g. to
// call .cancel() from the GUI thread while a worker thread's render call is
// using it) -- so its lifetime needs real reference counting, not a single
// owner.
class CancelToken {
public:
    void cancel() { flag_.store(true, std::memory_order_relaxed); }
    void reset()  { flag_.store(false, std::memory_order_relaxed); }
    bool is_cancelled() const { return flag_.load(std::memory_order_relaxed); }
    const std::atomic<bool>* ptr() const { return &flag_; }

private:
    std::atomic<bool> flag_{false};
};

// The reverse of image_to_numpy, for analysis functions (wada_diagnostic,
// extract_boundary_points) that take an Image: every render_* method hands
// Python a NumPy array, never an Image object (Image itself is not bound),
// so this is how a basin array a user rendered gets back into C++. Copies
// (unlike the zero-copy render path) since there is no way to alias a
// NumPy-owned buffer as an Image's std::vector<double> storage.
Image numpy_to_image(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 2) throw std::invalid_argument("expected a 2D array (rows, cols)");
    auto buf = arr.unchecked<2>();
    const int h = static_cast<int>(buf.shape(0));
    const int w = static_cast<int>(buf.shape(1));
    Image img(w, h);
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col)
            img.at(col, row) = buf(row, col);
    return img;
}

Family family_from_py(const std::string& s) {
    Family f;
    if (!family_from_string(s, f)) {
        throw std::invalid_argument(
            "unknown family '" + s +
            "'. Known: quadratic/mandelbrot, cubic/multibrot3, "
            "quintic/multibrot5, mcmullen2, mcmullen3, newton3");
    }
    return f;
}

}  // namespace

PYBIND11_MODULE(cdx, m) {
    m.doc() = "Complex dynamics rendering core (C++ backend).";

    // ---- Family ------------------------------------------------------------
    py::enum_<Family>(m, "Family")
        .value("Quadratic", Family::Quadratic)
        .value("Cubic",     Family::Cubic)
        .value("Quintic",   Family::Quintic)
        .value("McMullen2", Family::McMullen2)
        .value("McMullen3", Family::McMullen3)
        .value("Newton3",   Family::Newton3)
        .value("Custom",    Family::Custom);

    m.def("family_from_name", &family_from_py, py::arg("name"),
          "Resolve a family name ('mandelbrot', 'mcmullen3', ...) to a Family. "
          "Custom maps are constructed via Map.custom(), not by name.");

    // ---- GreensPotential -----------------------------------------------------
    py::enum_<GreensPotential>(m, "GreensPotential")
        .value("Pragmatic", GreensPotential::Pragmatic)
        .value("Conformal", GreensPotential::Conformal);

    // ---- CriticalPointFamily (Renderer.render_parameter_period) --------------
    py::enum_<CriticalPointFamily>(m, "CriticalPointFamily")
        .value("RelaxedNewtonPower", CriticalPointFamily::RelaxedNewtonPower,
              "Relaxed Newton of z^n-1 -- see the C++ enum's own doc comment for "
              "the closed-form critical points this generates per pixel.");

    // ---- Map ---------------------------------------------------------------
    py::class_<Map>(m, "Map")
        .def(py::init<>())
        .def(py::init<Family, Cplx>(), py::arg("family"), py::arg("param"))
        .def(py::init([](const std::string& name, Cplx param) {
                 return Map(family_from_py(name), param);
             }),
             py::arg("family"), py::arg("param"),
             "Construct from a family name and parameter.")
        .def_static("custom", &Map::custom,
                    py::arg("rational_map"), py::arg("param") = Cplx{0.0, 0.0},
                    "Wrap a RationalMap as a renderable Map (Family.Custom).")
        .def_property_readonly("family", &Map::family)
        .def_property("param", &Map::param, &Map::set_param)
        .def_property_readonly("degree", &Map::degree)
        .def_property_readonly("custom_map",
            [](const Map& mp) -> py::object {
                const RationalMap* p = mp.custom_map();
                return p ? py::cast(*p) : py::none();
            },
            "A COPY of the wrapped RationalMap, or None for a built-in family. "
            "Edit the RationalMap before wrapping it, not after.")
        .def_static("critical_point", &Map::critical_point,
                    py::arg("family"), py::arg("param"),
                    "Critical point governing parameter-plane membership. "
                    "Built-in families only; a Custom map's critical points "
                    "come from its wrapped RationalMap.critical_points(a).")
        .def("__repr__", [](const Map& mp) {
            if (mp.family() == Family::Custom) {
                return "<cdx.Map custom '" + mp.custom_map()->name() + "' param=(" +
                       std::to_string(mp.param().real()) + "," +
                       std::to_string(mp.param().imag()) + ")>";
            }
            return "<cdx.Map " + to_string(mp.family()) + " param=(" +
                   std::to_string(mp.param().real()) + "," +
                   std::to_string(mp.param().imag()) + ")>";
        });

    // ---- RationalMap sandbox ------------------------------------------------
    py::bind_vector<std::vector<PolyTerm>>(m, "PolyTermList");
    py::bind_vector<std::vector<PoleTerm>>(m, "PoleTermList");

    py::class_<PolyTerm>(m, "PolyTerm")
        .def(py::init<>())
        .def_readwrite("coeff",       &PolyTerm::coeff)
        .def_readwrite("exponent",    &PolyTerm::exponent)
        .def_readwrite("param_power", &PolyTerm::param_power)
        .def_readwrite("enabled",     &PolyTerm::enabled)
        .def_readwrite("label",       &PolyTerm::label)
        .def("effective_coeff", &PolyTerm::effective_coeff, py::arg("a"))
        .def("__repr__", [](const PolyTerm& t) {
            return "<cdx.PolyTerm exponent=" + std::to_string(t.exponent) +
                   (t.enabled ? "" : " (disabled)") + ">";
        });

    py::class_<PoleTerm>(m, "PoleTerm")
        .def(py::init<>())
        .def_readwrite("location",           &PoleTerm::location)
        .def_readwrite("strength",           &PoleTerm::strength)
        .def_readwrite("order",              &PoleTerm::order)
        .def_readwrite("param_power",        &PoleTerm::param_power)
        .def_readwrite("enabled",            &PoleTerm::enabled)
        .def_readwrite("location_is_param",  &PoleTerm::location_is_param)
        .def_readwrite("label",              &PoleTerm::label)
        .def("effective_strength", &PoleTerm::effective_strength, py::arg("a"))
        .def("effective_location", &PoleTerm::effective_location, py::arg("a"))
        .def("__repr__", [](const PoleTerm& t) {
            return "<cdx.PoleTerm order=" + std::to_string(t.order) +
                   (t.enabled ? "" : " (disabled)") + ">";
        });

    py::class_<FixedPoint>(m, "FixedPoint")
        .def(py::init<>())
        .def_readwrite("point",      &FixedPoint::point)
        .def_readwrite("multiplier", &FixedPoint::multiplier)
        .def("__repr__", [](const FixedPoint& fp) {
            return "<cdx.FixedPoint point=(" + std::to_string(fp.point.real()) + "," +
                   std::to_string(fp.point.imag()) + ") multiplier=(" +
                   std::to_string(fp.multiplier.real()) + "," +
                   std::to_string(fp.multiplier.imag()) + ")>";
        });

    py::class_<RationalMap>(m, "RationalMap")
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("name"))
        .def_property("name",  &RationalMap::name,  &RationalMap::set_name)
        .def_property("notes", &RationalMap::notes, &RationalMap::set_notes)
        .def("add_poly", &RationalMap::add_poly,
             py::arg("coeff"), py::arg("exponent"), py::arg("param_power") = 0,
             py::arg("label") = std::string())
        .def("add_pole", &RationalMap::add_pole,
             py::arg("location"), py::arg("strength"), py::arg("order") = 1,
             py::arg("param_power") = 0, py::arg("label") = std::string())
        .def("remove_poly", &RationalMap::remove_poly, py::arg("i"))
        .def("remove_pole", &RationalMap::remove_pole, py::arg("i"))
        .def("clear", &RationalMap::clear)
        // reference_internal: keeps the RationalMap alive as long as Python
        // holds the returned term list, and (via the opaque bindings above)
        // aliases the actual storage rather than a snapshot copy.
        .def("poly_terms",
             static_cast<std::vector<PolyTerm>& (RationalMap::*)()>(&RationalMap::poly_terms),
             py::return_value_policy::reference_internal)
        .def("pole_terms",
             static_cast<std::vector<PoleTerm>& (RationalMap::*)()>(&RationalMap::pole_terms),
             py::return_value_policy::reference_internal)
        .def("eval",  &RationalMap::eval,  py::arg("z"), py::arg("a"))
        .def("deriv", &RationalMap::deriv, py::arg("z"), py::arg("a"))
        .def("degree", &RationalMap::degree, py::arg("a"))
        .def("pole_locations", &RationalMap::pole_locations, py::arg("a"))
        .def("pole_orders", &RationalMap::pole_orders, py::arg("a"),
             "TRUE local order at each of pole_locations(a)'s entries, same "
             "index for index.")
        .def("critical_points", &RationalMap::critical_points, py::arg("a"),
             "All critical points on the Riemann sphere, WITH multiplicity: "
             "ordinary derivative zeros, each pole (multiplicity order-1), "
             "and infinity (multiplicity |p-q|-1 when |p-q| >= 2). Total "
             "count is 2*degree(a)-2 (Riemann-Hurwitz). Infinity is "
             "complex(float('inf'), 0).")
        .def("distinct_critical_points", &RationalMap::distinct_critical_points,
             py::arg("a"), py::arg("rel_tol") = 1e-4,
             "critical_points(a), deduplicated to one representative per "
             "critical point -- what you want when seeding one orbit per "
             "critical point rather than iterating multiplicity times.")
        .def("fixed_points", &RationalMap::fixed_points, py::arg("a"),
             "ALL fixed points R(z)==z (not just attracting ones -- see "
             "cdx.complete_attractors for that), each with its multiplier. "
             "Includes infinity when R(infinity)==infinity.")
        .def("to_formula", &RationalMap::to_formula)
        .def("serialize", &RationalMap::serialize)
        .def_static("deserialize", [](const std::string& text) {
                 RationalMap out;
                 std::string err;
                 if (!RationalMap::deserialize(text, out, err))
                     throw std::invalid_argument(err);
                 return out;
             }, py::arg("text"))
        .def_static("mandelbrot",   &RationalMap::mandelbrot)
        .def_static("multibrot",    &RationalMap::multibrot, py::arg("n"))
        .def_static("mcmullen",     &RationalMap::mcmullen, py::arg("n"))
        .def_static("newton_cubic", &RationalMap::newton_cubic)
        // ---- P/Q-backed (the equation-authoring path, Stages 2-4) ----------
        .def_static("from_expression", &RationalMap::from_expression,
             py::arg("source"), py::arg("active_param"), py::arg("fixed_values"),
             py::arg("name") = std::string("untitled"),
             "Parses `source` (a rational expression in z with arbitrary named "
             "parameters) and reduces it to this engine's single-active-"
             "parameter P/Q form. `active_param` may be '' iff `source` has at "
             "most one parameter (exactly one -> auto-active); with two or "
             "more, it must name which one is active. Every OTHER parameter "
             "needs a value in `fixed_values` (a dict[str, complex]), "
             "substituted as a constant. Raises ValueError for a parse "
             "failure, an unresolvable/ambiguous active parameter, or a "
             "missing fixed value.")
        .def("is_pq_backed", &RationalMap::is_pq_backed,
             "TRUE if this map was built via from_expression (or the lower-"
             "level from_canonical), not add_poly/add_pole -- poly_terms()/"
             "pole_terms() are both empty for a map built this way.")
        .def_property_readonly("pq_source", &RationalMap::pq_source,
             "The literal authored source text (e.g. 'a*z^3 - b*z + c'), if "
             "is_pq_backed() -- '' otherwise. Prefer this over to_formula() "
             "when non-empty: it's the user's own text, not a "
             "reconstruction, and (unlike the reduced P/Q actually used for "
             "computation) still shows every parameter the user originally "
             "typed.")
        .def_property_readonly("pq_active_param", &RationalMap::pq_active_param,
             "Which parsed parameter is bound to eval/deriv/etc's own `a`, "
             "if is_pq_backed() -- '' otherwise (including the genuine "
             "zero-parameter case).")
        .def_property_readonly("pq_fixed_params", &RationalMap::pq_fixed_params,
             "dict[str, complex] of every OTHER parameter from_expression "
             "substituted as a constant.")
        .def("add_pole_at", &RationalMap::add_pole_at, py::arg("location"),
             "P/Q-backed analogue of add_pole: multiplies the denominator by "
             "(z-location) -- forward root->factor, not a search for a "
             "location with some PRESCRIBED dynamical property (that's an "
             "inverse problem, out of scope here). Raises RuntimeError if "
             "not is_pq_backed().")
        .def("add_zero_at", &RationalMap::add_zero_at, py::arg("location"),
             "P/Q-backed analogue of add_poly's own zero-placing role: "
             "multiplies the numerator by (z-location). Raises RuntimeError "
             "if not is_pq_backed().")
        .def("is_polynomial_structurally", &RationalMap::is_polynomial_structurally,
             "TRUE iff this map is STRUCTURALLY a polynomial of degree >= 2 "
             "(no pole for ANY parameter value) -- representation-agnostic, "
             "works identically for a term-based or P/Q-backed map.")
        .def("__repr__", [](const RationalMap& r) {
            return "<cdx.RationalMap '" + r.name() + "': " + r.to_formula() + ">";
        });

    py::class_<FamilyLibrary>(m, "FamilyLibrary")
        .def(py::init<>())
        .def("add", &FamilyLibrary::add, py::arg("map"))
        .def("remove", &FamilyLibrary::remove, py::arg("name"))
        .def("find", [](const FamilyLibrary& lib, const std::string& name) -> py::object {
                 const RationalMap* p = lib.find(name);
                 return p ? py::cast(*p) : py::none();
             }, py::arg("name"))
        .def("names", &FamilyLibrary::names)
        .def("__len__", &FamilyLibrary::size)
        .def("serialize", &FamilyLibrary::serialize)
        .def_static("deserialize", [](const std::string& text) {
                 FamilyLibrary out;
                 std::string err;
                 if (!FamilyLibrary::deserialize(text, out, err))
                     throw std::invalid_argument(err);
                 return out;
             }, py::arg("text"))
        .def_static("with_defaults", &FamilyLibrary::with_defaults);

    // ---- Expr ----------------------------------------------------------------
    py::class_<Expr>(m, "Expr")
        .def(py::init<>())
        .def("compile", [](Expr& e, const std::string& src) {
                 std::string err;
                 if (!e.compile(src, err)) throw std::invalid_argument(err);
             }, py::arg("source"),
             "Compiles a formula in z and a (e.g. 'z^5 + a/z^2 - 0.3'). "
             "Raises ValueError with the parser's message on failure.")
        .def_property_readonly("valid", &Expr::valid)
        .def_property_readonly("source", &Expr::source)
        .def_property_readonly("stack_depth", &Expr::stack_depth)
        .def("__call__", [](const Expr& e, Cplx z, Cplx a) { return e(z, a); },
             py::arg("z"), py::arg("a"))
        .def("__repr__", [](const Expr& e) {
            return "<cdx.Expr '" + e.source() + "'>";
        });

    // ---- rational-expression parser (Stage 1 of the P/Q milestone) -----------
    m.def("parse_rational_parameters", [](const std::string& source) {
              CanonicalRational cr;
              std::string error;
              if (!parse_rational(source, cr, error)) throw std::invalid_argument(error);
              return cr.parameters;
          }, py::arg("source"),
          "Validates `source` as a RATIONAL expression (+ - * / integer-^ "
          "parens unary-minus z i numbers and named parameters -- no "
          "transcendental functions) and returns its parameter names, "
          "sorted and deduplicated (never including z/i). Raises ValueError "
          "with a specific message (unknown symbol, transcendental function "
          "-- distinguished from an unrecognized one, malformed syntax, a "
          "non-integer/non-literal exponent, ...) if `source` is not valid "
          "-- the SAME validation RationalMap.from_expression itself uses "
          "when actually building a map, exposed standalone for live-typing "
          "feedback before committing to that.");

    // ---- Viewport ----------------------------------------------------------
    py::class_<Viewport>(m, "Viewport")
        .def(py::init<>())
        .def(py::init([](Cplx center, double scale, int resolution) {
                 Viewport v; v.center = center; v.scale = scale;
                 v.resolution = resolution; return v;
             }),
             py::arg("center") = Cplx{0.0, 0.0},
             py::arg("scale") = 1.5,
             py::arg("resolution") = 800)
        .def_readwrite("center",     &Viewport::center)
        .def_readwrite("scale",      &Viewport::scale)
        .def_readwrite("resolution", &Viewport::resolution)
        .def_property_readonly("pixel_size", &Viewport::pixel_size)
        .def("extent", [](const Viewport& v) {
                 // (left, right, bottom, top) -- ready for matplotlib imshow
                 return py::make_tuple(v.center.real() - v.scale,
                                       v.center.real() + v.scale,
                                       v.center.imag() - v.scale,
                                       v.center.imag() + v.scale);
             },
             "Extent tuple for matplotlib imshow(..., origin='lower').")
        .def("__repr__", [](const Viewport& v) {
            return "<cdx.Viewport center=(" + std::to_string(v.center.real()) +
                   "," + std::to_string(v.center.imag()) + ") scale=" +
                   std::to_string(v.scale) + " res=" +
                   std::to_string(v.resolution) + ">";
        });

    // ---- RenderSettings ----------------------------------------------------
    py::class_<RenderSettings>(m, "RenderSettings")
        .def(py::init<>())
        .def(py::init([](int max_iter, double escape_radius, double tol, int threads) {
                 RenderSettings s; s.max_iter = max_iter;
                 s.escape_radius = escape_radius; s.tol = tol;
                 s.threads = threads; return s;
             }),
             py::arg("max_iter") = 200,
             py::arg("escape_radius") = 2.0,
             py::arg("tol") = 1e-6,
             py::arg("threads") = 0)
        .def_readwrite("max_iter",      &RenderSettings::max_iter)
        .def_readwrite("escape_radius", &RenderSettings::escape_radius)
        .def_readwrite("tol",           &RenderSettings::tol)
        .def_readwrite("threads",       &RenderSettings::threads);

    // ---- Cycle -------------------------------------------------------------
    py::class_<Cycle>(m, "Cycle")
        .def(py::init<>())
        .def(py::init([](std::vector<Cplx> pts, int id) {
                 Cycle c; c.points = std::move(pts); c.id = id; return c;
             }),
             py::arg("points"), py::arg("id") = 1,
             "An attracting cycle; a fixed point is a cycle of length 1. "
             "Use complex(float('inf'), 0) for the point at infinity.")
        .def_readwrite("points", &Cycle::points)
        .def_readwrite("id",     &Cycle::id);

    // ---- chordal metric ----------------------------------------------------
    m.def("chordal_distance",
          [](Cplx z, Cplx w) {
              return chordal_distance(z.real(), z.imag(), w.real(), w.imag());
          },
          py::arg("z"), py::arg("w"),
          "Chordal (spherical) distance; infinity is an ordinary point.");

    // ---- CancelToken ---------------------------------------------------------
    py::class_<CancelToken, std::shared_ptr<CancelToken>>(m, "CancelToken")
        .def(py::init<>())
        .def("cancel", &CancelToken::cancel,
             "Requests that any render currently using this token stop at "
             "its next per-column check. Safe to call from any thread.")
        .def("reset", &CancelToken::reset,
             "Clears a previous cancel() so the token can be reused.")
        .def_property_readonly("is_cancelled", &CancelToken::is_cancelled)
        .def("__repr__", [](const CancelToken& t) {
            return std::string("<cdx.CancelToken cancelled=") +
                   (t.is_cancelled() ? "True>" : "False>");
        });

    // ---- Renderer ----------------------------------------------------------
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<>())
        .def(py::init<Map, Viewport, RenderSettings>(),
             py::arg("map"), py::arg("viewport"), py::arg("settings"))
        .def(py::init([](const std::string& family, Cplx param,
                         Cplx center, double scale, int resolution,
                         int max_iter, double escape_radius, int threads) {
                 Viewport v; v.center = center; v.scale = scale;
                 v.resolution = resolution;
                 RenderSettings s; s.max_iter = max_iter;
                 s.escape_radius = escape_radius; s.threads = threads;
                 return Renderer(Map(family_from_py(family), param), v, s);
             }),
             py::arg("family"), py::arg("param") = Cplx{0.0, 0.0},
             py::arg("center") = Cplx{0.0, 0.0},
             py::arg("scale") = 1.5,
             py::arg("resolution") = 800,
             py::arg("max_iter") = 200,
             py::arg("escape_radius") = 2.0,
             py::arg("threads") = 0,
             "Convenience constructor taking a family name and flat settings.")

        .def_property("map",      &Renderer::map,      &Renderer::set_map)
        .def_property("viewport", &Renderer::viewport, &Renderer::set_viewport)
        .def_property("settings", &Renderer::settings, &Renderer::set_settings)

        .def("zoom", &Renderer::zoom, py::arg("target"), py::arg("factor"),
             "Recentre on target and divide the half-width by factor.")
        .def_property_readonly("precision_floor", &Renderer::precision_floor,
             "Smallest half-width the double grid still resolves.")

        .def("render_julia",
             [](const Renderer& r, std::shared_ptr<CancelToken> cancel,
                const std::vector<Cycle>& cycles) {
                 const std::atomic<bool>* cp = cancel ? cancel->ptr() : nullptr;
                 Image labels;
                 py::array_t<double> values_arr, labels_arr;
                 {
                     py::gil_scoped_release release;   // let other threads run
                     Image img = r.render_julia(cp, cycles, &labels);
                     py::gil_scoped_acquire acquire;
                     values_arr = image_to_numpy(std::move(img));
                     labels_arr = image_to_numpy(std::move(labels));
                 }
                 return py::make_tuple(values_arr, labels_arr);
             },
             py::arg("cancel") = nullptr, py::arg("cycles") = std::vector<Cycle>{},
             "Julia set of the bound map -- TWO PATHS (see Renderer::render_julia's "
             "own doc comment): a CERTIFIED polynomial (Renderer.polynomial_"
             "escape_certified... see the module-level cdx.polynomial_escape_"
             "certified) ignores `cycles` entirely and returns today's smooth "
             "escape-time values (0 = never escaped) with an all-zero labels "
             "array (no basin concept there); a RATIONAL map (has poles) needs "
             "`cycles` (cdx.find_attractors' own output, including infinity when "
             "it's attracting) and returns (smooth chordal approach-rate values, "
             "0 = unresolved; labels = which attractor's Cycle.id each pixel "
             "reached, 0 = unresolved) -- the SAME (primary, shading-channel) "
             "split render_basin already returns, for the SAME reason (per-basin "
             "hue + rate-based shading). Pass a CancelToken to make this "
             "interruptible from another thread; on cancellation the (partial) "
             "result should be discarded, not displayed.")

        .def("render_parameter",
             [](const Renderer& r, std::shared_ptr<CancelToken> cancel) {
                 const std::atomic<bool>* cp = cancel ? cancel->ptr() : nullptr;
                 py::array_t<double> arr;
                 {
                     py::gil_scoped_release release;
                     Image img = r.render_parameter(cp);
                     py::gil_scoped_acquire acquire;
                     arr = image_to_numpy(std::move(img));
                 }
                 return arr;
             },
             py::arg("cancel") = nullptr,
             "Parameter plane (Mandelbrot / multibrot / McMullenbrot) -- escape-time, "
             "for every family alike (see Renderer::render_parameter's own doc "
             "comment for why this mode stays escape_radius-governed rather than "
             "escape-radius-free/sphere-aware the way Julia/Green's are: it's a "
             "VISUALIZATION, and escape_radius is a real tuning knob here, not an "
             "invariant of the map).")

        .def("render_parameter_basin",
             [](const Renderer& r, std::shared_ptr<CancelToken> cancel) {
                 const std::atomic<bool>* cp = cancel ? cancel->ptr() : nullptr;
                 Image unresolved, certain;
                 py::array_t<double> counts_arr, unresolved_arr, certain_arr;
                 {
                     py::gil_scoped_release release;
                     Image img = r.render_parameter_basin(cp, &unresolved, &certain);
                     py::gil_scoped_acquire acquire;
                     counts_arr = image_to_numpy(std::move(img));
                     unresolved_arr = image_to_numpy(std::move(unresolved));
                     certain_arr = image_to_numpy(std::move(certain));
                 }
                 return py::make_tuple(counts_arr, unresolved_arr, certain_arr);
             },
             py::arg("cancel") = nullptr,
             "Parameter_basin: each pixel is a parameter value; returns (counts, "
             "unresolved, certain). counts is the NUMBER OF DISTINCT ATTRACTING "
             "CYCLES the map has at that parameter (infinity counts as one when "
             "it's the limit of a critical orbit, or an algebraically-injected "
             "fixed point -- see cdx.complete_attractors); unresolved is how many "
             "of that pixel's critical orbits are explained by NO attractor in "
             "the complete set (Siegel/Herman/parabolic land here, tracked "
             "separately -- never silently folded into counts, and already "
             "reconciled against injected fixed points: an orbit that failed to "
             "close numerically but actually landed on one is not counted here); "
             "certain is how many of that pixel's counted cycles are CERTAIN -- "
             "algebraically injected fixed points, not numerically-discovered "
             "closures still subject to a finite search budget/tolerance (see "
             "app.color.color_parameter_basin's own use of this for its display "
             "policy: a certain attractor is shown as confirmed even when "
             "unresolved > count overall). Escape-radius-free. Requires a "
             "Custom-wrapped map (see Renderer::render_parameter_basin's own doc "
             "comment) -- degrades to an honest all-zero (counts, unresolved, "
             "certain) triple otherwise, rather than guessing.")

        .def("render_parameter_period",
             [](const Renderer& r, CriticalPointFamily seed_family, int n,
                std::shared_ptr<CancelToken> cancel) {
                 const std::atomic<bool>* cp = cancel ? cancel->ptr() : nullptr;
                 Image undetermined, is_infinity;
                 py::array_t<double> periods_arr, undetermined_arr, is_infinity_arr;
                 {
                     py::gil_scoped_release release;
                     Image img = r.render_parameter_period(seed_family, n, cp, &undetermined,
                                                           &is_infinity);
                     py::gil_scoped_acquire acquire;
                     periods_arr = image_to_numpy(std::move(img));
                     undetermined_arr = image_to_numpy(std::move(undetermined));
                     is_infinity_arr = image_to_numpy(std::move(is_infinity));
                 }
                 return py::make_tuple(periods_arr, undetermined_arr, is_infinity_arr);
             },
             py::arg("seed_family"), py::arg("n"), py::arg("cancel") = nullptr,
             "Parameter_period: each pixel is a parameter value; returns (periods, "
             "undetermined, is_infinity). periods is the PERIOD (measured in the "
             "raw z-plane -- see per_seed_outcomes, never a symmetry quotient) of "
             "the attracting cycle a TRACKED critical orbit converges to, where "
             "`seed_family` (a CriticalPointFamily) supplies that orbit's closed-"
             "form location per pixel -- deliberately NOT RationalMap.distinct_"
             "critical_points(a) the way Parameter_basin uses, since that includes "
             "critical points structurally unrelated to the tracked orbit (see "
             "Renderer.render_parameter_period's own C++ doc comment for a measured "
             "example). `n` is the family's own structural parameter (e.g. the "
             "power in relaxed-Newton-of-z^n-1; for that family it equals map."
             "degree(a) exactly, if the caller only has the map itself). "
             "undetermined is 1.0 where no tracked seed resolved within budget "
             "(a genuine residual -- Siegel/Herman/parabolic -- never a fabricated "
             "period); is_infinity is 1.0 where the resolved cycle is the point at "
             "infinity, kept OUT of `periods`' own value and its golden-hue color "
             "family. At most one of the two is ever 1.0 for the same pixel. "
             "Escape-radius-free. Requires a Custom-wrapped map -- degrades to an "
             "honest all-undetermined image otherwise. CURRENTLY SUPPORTS EXACTLY "
             "ONE seed_family (RelaxedNewtonPower) -- see CriticalPointFamily's "
             "own doc comment before calling this on any other map shape.")

        .def("render_basin",
             [](const Renderer& r, const std::vector<Cycle>& cycles,
                std::shared_ptr<CancelToken> cancel) {
                 const std::atomic<bool>* cp = cancel ? cancel->ptr() : nullptr;
                 Image iterations;
                 py::array_t<double> labels_arr, iterations_arr;
                 {
                     py::gil_scoped_release release;
                     Image img = r.render_basin(cycles, &iterations, cp);
                     py::gil_scoped_acquire acquire;
                     labels_arr = image_to_numpy(std::move(img));
                     iterations_arr = image_to_numpy(std::move(iterations));
                 }
                 return py::make_tuple(labels_arr, iterations_arr);
             },
             py::arg("cycles"), py::arg("cancel") = nullptr,
             "Basin classification (chordal metric). Returns (labels, "
             "iterations): labels is 0 for unresolved pixels, else the "
             "cycle id; iterations is the per-pixel iteration count it "
             "took to resolve (not a meaningful 'convergence speed' where "
             "labels==0 -- see Renderer::render_basin's own doc comment). "
             "For basin SHADING: hue from labels, brightness from "
             "iterations (see app/color.py's color_basin).")

        .def("render_greens",
             [](const Renderer& r, std::shared_ptr<CancelToken> cancel,
                const std::vector<Cycle>& cycles, GreensPotential potential) {
                 const std::atomic<bool>* cp = cancel ? cancel->ptr() : nullptr;
                 Image exact;
                 py::array_t<double> values_arr, exact_arr;
                 {
                     py::gil_scoped_release release;
                     Image img = r.render_greens(cp, cycles, potential, &exact);
                     py::gil_scoped_acquire acquire;
                     values_arr = image_to_numpy(std::move(img));
                     exact_arr = image_to_numpy(std::move(exact));
                 }
                 return py::make_tuple(values_arr, exact_arr);
             },
             py::arg("cancel") = nullptr, py::arg("cycles") = std::vector<Cycle>{},
             py::arg("potential") = GreensPotential::Pragmatic,
             "Green's function (dynamical potential) -- TWO PATHS (see Renderer::"
             "render_greens' own doc comment): a CERTIFIED polynomial ignores "
             "`cycles`/`potential` entirely and returns today's G_f(z) = lim "
             "d^-n log+|f^n(z)|, normalized at each pixel's own escape "
             "iteration (0 = never escaped), with an all-zero exact array; a "
             "RATIONAL map needs `cycles` (cdx.find_attractors' own output) "
             "and returns (potential values, exact) where `potential` selects "
             "GreensPotential.Pragmatic (smooth chordal approach-rate, the "
             "SAME quantity render_julia's own rational path computes) or "
             "GreensPotential.Conformal (log|phi(z)| for the Boettcher/"
             "Koenigs coordinate, estimated numerically; falls back to "
             "Pragmatic -- flagged via exact==0 -- when the orbit's own local "
             "behaviour is too close to parabolic to classify). exact is 1 "
             "where Conformal was genuinely computed, 0 where it fell back "
             "(meaningless for Pragmatic, which has no such fallback).")

        .def("render_parameter_greens",
             [](const Renderer& r, std::shared_ptr<CancelToken> cancel,
                GreensPotential potential) {
                 const std::atomic<bool>* cp = cancel ? cancel->ptr() : nullptr;
                 Image exact;
                 py::array_t<double> values_arr, exact_arr;
                 {
                     py::gil_scoped_release release;
                     Image img = r.render_parameter_greens(cp, potential, &exact);
                     py::gil_scoped_acquire acquire;
                     values_arr = image_to_numpy(std::move(img));
                     exact_arr = image_to_numpy(std::move(exact));
                 }
                 return py::make_tuple(values_arr, exact_arr);
             },
             py::arg("cancel") = nullptr, py::arg("potential") = GreensPotential::Pragmatic,
             "The family escape-rate function on the PARAMETER plane (G_M(c) "
             "for the quadratic family) -- a DIFFERENT function on a "
             "DIFFERENT space from render_greens: the pixel is a parameter, "
             "the orbit starts at THAT parameter's critical point (like "
             "render_parameter), and the accumulated quantity is that "
             "critical orbit's escape rate.");

    // ---- analysis layer ------------------------------------------------------
    py::class_<FindAttractorsOptions>(m, "FindAttractorsOptions")
        .def(py::init<>())
        .def_readwrite("burn_in",           &FindAttractorsOptions::burn_in)
        .def_readwrite("max_period",        &FindAttractorsOptions::max_period)
        .def_readwrite("tol",               &FindAttractorsOptions::tol)
        .def_readwrite("inf_cutoff",        &FindAttractorsOptions::inf_cutoff)
        .def_readwrite("verify_multiplier", &FindAttractorsOptions::verify_multiplier)
        .def_readwrite("confirm_weakly_attracting", &FindAttractorsOptions::confirm_weakly_attracting)
        .def_readwrite("loose_tol",              &FindAttractorsOptions::loose_tol)
        .def_readwrite("extended_max_period",    &FindAttractorsOptions::extended_max_period)
        .def_readwrite("newton_iterations",      &FindAttractorsOptions::newton_iterations)
        .def_readwrite("attracting_margin",      &FindAttractorsOptions::attracting_margin);

    m.def("find_attractors", &find_attractors, py::arg("map"), py::arg("a"),
          py::arg("opts") = FindAttractorsOptions{},
          "Discovers attracting cycles via critical orbits (Fatou's theorem). "
          "Returns a list of Cycle, ready to pass to Renderer.render_basin. "
          "PURELY critical-seeded -- a finite-budget numerical search, not a "
          "completeness guarantee; prefer complete_attractors for any real "
          "consumer that wants the full attractor set (see its own doc "
          "comment). This is kept as its own, unreconciled function because "
          "some callers -- and this project's own tests -- specifically want "
          "the plain critical-seeded algorithm's own behavior.");

    m.def("complete_attractors", &complete_attractors, py::arg("map"), py::arg("a"),
          py::arg("opts") = FindAttractorsOptions{},
          "find_attractors' own critical-seeded cycles, UNIONED with every "
          "algebraically-attracting fixed point map.fixed_points(a) reports "
          "(|multiplier| < 1, including infinity) that no critical-seeded "
          "cycle already represents -- see the C++ analysis.hpp doc comment "
          "for why find_attractors alone is not a completeness guarantee. "
          "This is what the fact sheet, basin classification, and "
          "Parameter_basin all actually use for 'the attractor set'.");

    py::class_<SeedOutcome>(m, "SeedOutcome")
        .def_readonly("period",      &SeedOutcome::period)
        .def_readonly("is_infinity", &SeedOutcome::is_infinity)
        .def_readonly("resolved",    &SeedOutcome::resolved);

    m.def("per_seed_outcomes", &per_seed_outcomes, py::arg("seeds"), py::arg("map"), py::arg("a"),
          py::arg("opts") = FindAttractorsOptions{},
          "What did EACH seed's own orbit converge to, one SeedOutcome per "
          "seed, in order -- NOT the deduplicated attractor SET find_"
          "attractors/complete_attractors return (see the C++ doc comment "
          "for a measured example of why those are the wrong tool for this "
          "question: the algebraic union can attribute a fixed point to a "
          "seed that never actually reaches it, and a map can have several "
          "SIMULTANEOUSLY attracting cycles from different critical points). "
          "resolved=False means a genuine residual (Siegel/Herman/parabolic) "
          "-- period/is_infinity are meaningless (left at 0/False) in that "
          "case, never a fabricated period. period is measured in the raw "
          "z-plane, never a symmetry quotient. This is what Renderer."
          "render_parameter_period uses per pixel.");

    m.def("polynomial_escape_certified", &polynomial_escape_certified, py::arg("map"),
          "True iff `map` has no poles anywhere (structurally -- independent of the "
          "parameter) and degree >= 2, in which case infinity is ALWAYS "
          "superattracting and a fixed |z|>R escape radius is a provably "
          "forward-invariant trap -- the condition that makes the classical "
          "polynomial escape-time fast path valid, not just fast. False for any "
          "map with a pole (even one whose infinity happens to be attracting too, "
          "e.g. mcmullen -- see this function's own C++ doc comment for why).");

    py::class_<WadaStats>(m, "WadaStats")
        .def(py::init<>())
        .def_readwrite("n_basins",            &WadaStats::n_basins)
        .def_readwrite("unresolved_fraction", &WadaStats::unresolved_fraction)
        .def_readwrite("boundary_fraction",   &WadaStats::boundary_fraction)
        .def_readwrite("wada_fraction",       &WadaStats::wada_fraction)
        .def_readwrite("radius_px",           &WadaStats::radius_px);

    m.def("wada_diagnostic",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> labels,
             double radius_fraction) {
              return wada_diagnostic(numpy_to_image(labels), radius_fraction);
          },
          py::arg("labels"), py::arg("radius_fraction") = 0.004,
          "Wada-boundary signatures on a basin label image (the array "
          "Renderer.render_basin returns). radius_fraction scales with "
          "resolution deliberately -- see WadaStats.wada_fraction's doc "
          "comment for why the trend across resolutions is the signal, "
          "not the absolute value at one setting.");

    py::class_<HausdorffResult>(m, "HausdorffResult")
        .def(py::init<>())
        .def_readwrite("chordal",                   &HausdorffResult::chordal)
        .def_readwrite("euclidean",                 &HausdorffResult::euclidean)
        .def_readwrite("chordal_julia_to_target",    &HausdorffResult::chordal_julia_to_target)
        .def_readwrite("chordal_target_to_julia",    &HausdorffResult::chordal_target_to_julia)
        .def_readwrite("euclidean_julia_to_target",  &HausdorffResult::euclidean_julia_to_target)
        .def_readwrite("euclidean_target_to_julia",  &HausdorffResult::euclidean_target_to_julia);

    m.def("hausdorff_distance", &hausdorff_distance,
          py::arg("julia_points"), py::arg("target_points"), py::arg("max_points") = 4000,
          "Symmetric Hausdorff distance (chordal and Euclidean) between two "
          "point sets, with all four directed distances also on the result -- "
          "large julia_to_target means spurious structure, large "
          "target_to_julia means missed boundary.");

    m.def("extract_boundary_points",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> labels,
             const Viewport& view) {
              return extract_boundary_points(numpy_to_image(labels), view);
          },
          py::arg("labels"), py::arg("view"),
          "Basin-label image (the array Renderer.render_basin returns) "
          "boundary pixels as complex points -- how julia_points is "
          "obtained in practice for hausdorff_distance.");

    py::class_<DynamicalFacts::AttractingCycle>(m, "AttractingCycle")
        .def(py::init<>())
        .def_readwrite("points",     &DynamicalFacts::AttractingCycle::points)
        .def_readwrite("period",     &DynamicalFacts::AttractingCycle::period)
        .def_readwrite("multiplier", &DynamicalFacts::AttractingCycle::multiplier);

    py::class_<DynamicalFacts>(m, "DynamicalFacts")
        .def(py::init<>())
        .def_readwrite("degree",            &DynamicalFacts::degree)
        .def_readwrite("critical_points",   &DynamicalFacts::critical_points)
        .def_readwrite("attracting_cycles", &DynamicalFacts::attracting_cycles)
        .def_readwrite("pole_locations",    &DynamicalFacts::pole_locations)
        .def_readwrite("pole_orders",       &DynamicalFacts::pole_orders)
        .def_readwrite("fixed_points",      &DynamicalFacts::fixed_points);

    m.def("dynamical_facts", &dynamical_facts, py::arg("map"), py::arg("a"),
          py::arg("opts") = FindAttractorsOptions{},
          "Bundles degree, critical_points, pole locations/orders, "
          "fixed_points and find_attractors' attracting cycles (each with "
          "period and multiplier) into one report -- the data-extraction "
          "call app.session wraps for the sandbox UI.");
}

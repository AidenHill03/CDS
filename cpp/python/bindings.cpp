// =============================================================================
// bindings.cpp -- pybind11 bindings for the cdx rendering core.
//
// Exposes Map, Viewport, RenderSettings, Cycle and Renderer to Python, with
// render methods returning NumPy arrays. The array shares the Image's buffer
// through a capsule that owns the moved-from Image, so there is no copy of
// the pixel data on the way out -- important because at 2000x2000 an image is
// 32 MB and copying it would dominate the render time we worked to reduce.
//
// Build:  pip install pybind11 && cmake --build build
// Use:    import cdx
// =============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "cdx/renderer.hpp"

#include <memory>
#include <stdexcept>

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
        .value("Newton3",   Family::Newton3);

    m.def("family_from_name", &family_from_py, py::arg("name"),
          "Resolve a family name ('mandelbrot', 'mcmullen3', ...) to a Family.");

    // ---- Map ---------------------------------------------------------------
    py::class_<Map>(m, "Map")
        .def(py::init<>())
        .def(py::init<Family, Cplx>(), py::arg("family"), py::arg("param"))
        .def(py::init([](const std::string& name, Cplx param) {
                 return Map(family_from_py(name), param);
             }),
             py::arg("family"), py::arg("param"),
             "Construct from a family name and parameter.")
        .def_property_readonly("family", &Map::family)
        .def_property("param", &Map::param, &Map::set_param)
        .def_property_readonly("degree", &Map::degree)
        .def_static("critical_point", &Map::critical_point,
                    py::arg("family"), py::arg("param"),
                    "Critical point governing parameter-plane membership.")
        .def("__repr__", [](const Map& mp) {
            return "<cdx.Map " + to_string(mp.family()) + " param=(" +
                   std::to_string(mp.param().real()) + "," +
                   std::to_string(mp.param().imag()) + ")>";
        });

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
             [](const Renderer& r) {
                 py::gil_scoped_release release;   // let other threads run
                 Image img = r.render_julia();
                 py::gil_scoped_acquire acquire;
                 return image_to_numpy(std::move(img));
             },
             "Escape-time Julia set; 0 means the orbit never escaped.")

        .def("render_parameter",
             [](const Renderer& r) {
                 py::gil_scoped_release release;
                 Image img = r.render_parameter();
                 py::gil_scoped_acquire acquire;
                 return image_to_numpy(std::move(img));
             },
             "Parameter plane (Mandelbrot / multibrot / McMullenbrot).")

        .def("render_basin",
             [](const Renderer& r, const std::vector<Cycle>& cycles) {
                 py::gil_scoped_release release;
                 Image img = r.render_basin(cycles);
                 py::gil_scoped_acquire acquire;
                 return image_to_numpy(std::move(img));
             },
             py::arg("cycles"),
             "Basin classification (chordal metric); 0 means unresolved.")

        .def("render_greens",
             [](const Renderer& r) {
                 bool normalized = false;
                 py::array_t<double> arr;
                 {
                     py::gil_scoped_release release;
                     Image img = r.render_greens(&normalized);
                     py::gil_scoped_acquire acquire;
                     arr = image_to_numpy(std::move(img));
                 }
                 return py::make_tuple(arr, normalized);
             },
             "Green's function. Returns (array, normalized); normalized is "
             "False when degree^max_iter overflowed and the values are "
             "comparable only within this image.");
}

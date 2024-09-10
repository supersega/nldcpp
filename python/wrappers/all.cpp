// Inherit flags from autodiff4py
#include <autodiff/python/bindings/pybind11.hxx>

#include <python/wrappers/arc_length.hpp>
#include <python/wrappers/continuation_parameters.hpp>
#include <python/wrappers/mappers.hpp>
#include <python/wrappers/math.hpp>
#include <python/wrappers/systems.hpp>
#include <python/wrappers/types.hpp>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <nld/core.hpp>

#include <boost/hana.hpp>

namespace py = pybind11;

using namespace boost::hana::literals;

namespace wrappers {
void wrap_make_duffing(py::module &m);
} // namespace wrappers

PYBIND11_MODULE(nldpy, m) {
    m.doc() = "nld bindings to python"; // optional module docstring

    // Wrap structures for continuation configuration
    wrappers::wrap_direction(m);
    wrappers::wrap_continuation_parameters(m);

    // Wrap mappers
    wrappers::wrap_mappers(m);

    // Wrap math
    wrappers::wrap_math(m);

    // Wrap systems
    wrappers::wrap_systems(m);

    // Wrap arc length continuation
    wrappers::wrap_arc_length(m);

    // Wrap duffing
    wrappers::wrap_make_duffing(m);
}

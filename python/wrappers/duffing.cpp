#include <autodiff/python/bindings/pybind11.hxx>
#include <nld/core.hpp>
#include <nld/systems.hpp>
#include <numbers>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <python/wrappers/types.hpp>

namespace py = pybind11;

namespace wrappers {

nld::vector_xdd duffing(const nld::vector_xdd &u, nld::dual t,
                        nld::dual Omega) {
    nld::vector_xdd du(u.size());

    nld::dual t8 = cos(t);

    du[0] = u[1];
    du[1] = -0.1e-1 * u[1] - 0.1000000000e1 * u[0] -
            0.1499999998e2 * u[0] * u[0] * u[0] - 1.0 * 0.8600261454e-2 * t8;

    du *= 2.0 * std::numbers::pi;
    du /= Omega;

    return du;
}

wrappers::non_autonomous::RnPlusOneToRnMapFnDual make_duffing() {
    return duffing;
}

void wrap_make_duffing(py::module &m) { m.def("make_duffing", &make_duffing); }

} // namespace wrappers

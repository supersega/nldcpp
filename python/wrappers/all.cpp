#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <nld/core.hpp>

#include <iostream>

namespace py = pybind11;

namespace wrappers {

void wrap_systems(py::module &m);
void wrap_make_duffing(py::module &m);
void wrap_test_caller(py::module &m);

void caller(std::function<void(nld::dual)> f) { f(1.0); }

auto make_callable() -> std::function<void(nld::dual)> {
    return [](nld::dual x) {
        x = 1.0;
        std::cout << x << std::endl;
    };
}

void wrap_test_caller(py::module &m) {
    m.def("caller", &wrappers::caller);
    m.def("make_callable", &wrappers::make_callable);
}
} // namespace wrappers
//

PYBIND11_MODULE(nldpy, m) {
    m.doc() = "nld bindings to python"; // optional module docstring

    wrappers::wrap_systems(m);
    wrappers::wrap_make_duffing(m);

    wrappers::wrap_test_caller(m);
}

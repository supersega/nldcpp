#pragma once

#include <autodiff/python/bindings/pybind11.hxx>
#include <nld/autocont.hpp>

namespace wrappers {
void wrap_direction(py::module &m);
void wrap_continuation_parameters(py::module &m);
} // namespace wrappers

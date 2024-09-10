#pragma once
#include <autodiff/python/bindings/pybind11.hxx>

#include <nld/systems.hpp>

namespace wrappers {
void wrap_systems(py::module &m);
} // namespace wrappers

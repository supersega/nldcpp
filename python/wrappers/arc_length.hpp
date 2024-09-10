#pragma once

#include <autodiff/python/bindings/pybind11.hxx>

#include <python/wrappers/systems.hpp>
#include <python/wrappers/types.hpp>

#include <pybind11/pybind11.h>

#include <nld/autocont.hpp>

namespace py = pybind11;

namespace wrappers {
void wrap_arc_length(py::module &m);

} // namespace wrappers

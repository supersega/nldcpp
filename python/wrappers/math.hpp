#pragma once
#include <autodiff/python/bindings/pybind11.hxx>

#include <nld/core.hpp>
#include <nld/math.hpp>

#include <python/wrappers/types.hpp>

namespace wrappers {
void wrap_math(py::module &m);
} // namespace wrappers

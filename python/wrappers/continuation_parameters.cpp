#include <nld/autocont/continuation_parameters.hpp>
#include <python/wrappers/continuation_parameters.hpp>

#include <nld/autocont.hpp>
#include <nld/math.hpp>

#include <pybind11/operators.h>

#include <sstream>

using namespace pybind11::literals;

namespace wrappers {

auto repr(const nld::newton_parameters &self) -> std::string;
auto str(const nld::newton_parameters &self) -> std::string;

auto to_string(const nld::direction &self) -> std::string {
    switch (self) {
    case nld::direction::forward:
        return "Forward";
    case nld::direction::reverse:
        return "Reverse";
    }
    return "Unknown";
}

auto repr(const nld::direction &self) -> std::string {
    std::stringstream ss;
    ss << "Direction(" << to_string(self) << ")";
    return ss.str();
}

auto str(const nld::direction &self) -> std::string {
    std::stringstream ss;
    ss << to_string(self);
    return ss.str();
}

auto repr(const nld::continuation_parameters &self) -> std::string {
    std::stringstream ss;
    ss << "ContinuationParameters(newton_parameters="
       << repr(self.newton_parameters)
       << ", total_param_length=" << self.total_param_length
       << ", param_min_step=" << self.param_min_step
       << ", param_max_step=" << self.param_max_step
       << ", direction=" << repr(self.direction) << ")";
    return ss.str();
}

auto str(const nld::continuation_parameters &self) -> std::string {
    std::stringstream ss;
    ss << str(self.newton_parameters) << ", " << self.total_param_length << ", "
       << self.param_min_step << ", " << self.param_max_step << ", "
       << str(self.direction);
    return ss.str();
}

void wrap_direction(py::module &m) {
    py::enum_<nld::direction>(m, "Direction")
        .value("Forward", nld::direction::forward)
        .value("Backward", nld::direction::reverse);
}

void wrap_continuation_parameters(py::module &m) {
    py::class_<nld::continuation_parameters>(m, "ContinuationParameters")
        .def(py::init<nld::newton_parameters, double, double, double,
                      nld::direction>())
        .def_readwrite("newton_parameters",
                       &nld::continuation_parameters::newton_parameters)
        .def_readwrite("total_param_length",
                       &nld::continuation_parameters::total_param_length)
        .def_readwrite("param_min_step",
                       &nld::continuation_parameters::param_min_step)
        .def_readwrite("param_max_step",
                       &nld::continuation_parameters::param_max_step)
        .def_readwrite("direction", &nld::continuation_parameters::direction)
        .def(
            "__repr__",
            [](const nld::continuation_parameters &self) { return repr(self); })
        .def("__str__",
             [](const nld::continuation_parameters &self) { return str(self); })
        .doc() = "Continuation parameters used in nld to control the "
                 "continuation solver";

    m.def(
        "continuation_parameters",
        [](nld::newton_parameters newton_parameters, double total_param_length,
           double param_min_step, double param_max_step,
           nld::direction direction) {
            return nld::continuation_parameters(
                std::move(newton_parameters), total_param_length,
                param_min_step, param_max_step, direction);
        },
        "newton_parameters"_a, "total_param_length"_a, "param_min_step"_a,
        "param_max_step"_a, "direction"_a);
}
} // namespace wrappers

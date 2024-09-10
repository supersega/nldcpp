#include <python/wrappers/math.hpp>

using namespace boost::hana::literals;
using namespace std::string_literals;
using namespace pybind11::literals;

namespace wrappers {

auto repr(const nld::newton_parameters &self) -> std::string {
    std::stringstream ss;
    ss << "NewtonParameters(max_iterations=" << self.max_iterations
       << ", tolerance=" << self.tolerance << ")";
    return ss.str();
}

auto str(const nld::newton_parameters &self) -> std::string {
    std::stringstream ss;
    ss << self.max_iterations << ", " << self.tolerance;
    return ss.str();
}

void wrap_newton_parameters(py::module &m) {
    py::class_<nld::newton_parameters>(m, "NewtonParameters")
        .def(py::init<std::size_t, double>())
        .def_readwrite("max_iterations",
                       &nld::newton_parameters::max_iterations)
        .def_readwrite("tolerance", &nld::newton_parameters::tolerance)
        .def("__repr__",
             [](const nld::newton_parameters &self) { return repr(self); })
        .def("__str__",
             [](const nld::newton_parameters &self) { return str(self); })
        .doc() = "Newton parameters used in nld to control the Newton solver";

    m.def(
        "newton_parameters",
        [](std::size_t max_iterations, double tolerance) {
            return nld::newton_parameters(max_iterations, tolerance);
        },
        "max_iterations"_a, "tolerance"_a);
}

template <typename Solver>
void wrap_solver(py::module &m, std::string_view name) {
    py::class_<Solver>(m, name.data()).def(py::init<>());
}

void wrap_solvers(py::module &m) {
    auto ss = wrappers::solvers;

    boost::hana::for_each(ss, [&m](auto pair) {
        using solver_t = typename decltype(+pair[0_c])::type;
        std::string_view name = pair[1_c];

        wrap_solver<solver_t>(m, name.data());
    });
}

void wrap_math(py::module &m) {
    wrap_newton_parameters(m);
    wrap_solvers(m);
}
} // namespace wrappers

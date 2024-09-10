#include <autodiff/python/bindings/pybind11.hxx>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <autodiff/autodiff/forward/dual/eigen.hpp>
#include <nld/core.hpp>
#include <nld/systems.hpp>

#include <python/wrappers/systems.hpp>
#include <python/wrappers/types.hpp>

#include <boost/hana.hpp>
#include <cstdlib>
#include <cxxabi.h>
#include <memory>

std::string demangle(const char *name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};

    return (status == 0) ? res.get() : name;
}
namespace py = pybind11;

using namespace boost::hana::literals;
using namespace std::string_literals;
using namespace pybind11::literals;

namespace wrappers {

template <typename T>
struct periodic_parameters_selector {
    using type = T;
};

template <>
struct periodic_parameters_selector<nld::runge_kutta_4> {
    using type = nld::periodic_parameters_constant;
};

template <>
struct periodic_parameters_selector<nld::runge_kutta_45> {
    using type = nld::periodic_parameters_adaptive;
};

auto repr(const nld::periodic_parameters_constant &self) -> std::string {
    std::stringstream ss;
    ss << "PeriodicParametersConstant(periods=" << self.periods
       << ", intervals=" << self.intervals << ")";
    return ss.str();
}

auto str(const nld::periodic_parameters_constant &self) -> std::string {
    std::stringstream ss;
    ss << self.periods << ", " << self.intervals;
    return ss.str();
}

auto repr(const nld::periodic_parameters_adaptive &self) -> std::string {
    std::stringstream ss;
    ss << "PeriodicParametersAdaptive(periods=" << self.periods
       << ", min_step=" << self.min_step << ", max_step=" << self.max_step
       << ", error=" << self.error << ")";
    return ss.str();
}

auto str(const nld::periodic_parameters_adaptive &self) -> std::string {
    std::stringstream ss;
    ss << self.periods << ", " << self.min_step << ", " << self.max_step << ", "
       << self.error;
    return ss.str();
}

void wrap_periodic_parameters_constant(py::module &m) {
    py::class_<nld::periodic_parameters_constant>(m,
                                                  "PeriodicParametersConstant")
        .def(py::init<std::size_t, std::size_t>())
        .def_readwrite("periods", &nld::periodic_parameters_constant::periods)
        .def_readwrite("intervals",
                       &nld::periodic_parameters_constant::intervals)
        .def("__repr__",
             [](const nld::periodic_parameters_constant &self) {
                 return repr(self);
             })
        .def("__str__",
             [](const nld::periodic_parameters_constant &self) {
                 return str(self);
             })
        .doc() = "Parameters for periodic shooting method with constant step "
                 "solver. Used to "
                 "control the number of periods and integration intervals";

    m.def(
        "periodic_parameters",
        [](std::size_t periods, std::size_t intervals) {
            return nld::periodic_parameters_constant(periods, intervals);
        },
        "periods"_a, "intervals"_a);
}

void wrap_periodic_parameters_adaptive(py::module &m) {
    py::class_<nld::periodic_parameters_adaptive>(m,
                                                  "PeriodicParametersAdaptive")
        .def(py::init<std::size_t, std::size_t, double, double>())
        .def_readwrite("periods", &nld::periodic_parameters_adaptive::periods)
        .def_readwrite("min_step", &nld::periodic_parameters_adaptive::min_step)
        .def_readwrite("max_step", &nld::periodic_parameters_adaptive::max_step)
        .def_readwrite("error", &nld::periodic_parameters_adaptive::error)
        .def("__repr__",
             [](const nld::periodic_parameters_adaptive &self) {
                 return repr(self);
             })
        .def("__str__",
             [](const nld::periodic_parameters_adaptive &self) {
                 return str(self);
             })
        .doc() = "Parameters for periodic shooting method with adaptive step "
                 "solver. Used to "
                 "control the number of periods, minimal and maximal step size "
                 "and the error";

    m.def(
        "periodic_parameters",
        [](std::size_t periods, double min_step, double max_step,
           double error) {
            return nld::periodic_parameters_adaptive(periods, min_step,
                                                     max_step, error);
        },
        "periods"_a, "min_step"_a, "max_step"_a, "error"_a);
}

template <typename NonAutnomousFunction>
void wrap_non_autonomous(py::module &m, std::string_view name) {
    py::class_<nld::non_autonomous<NonAutnomousFunction>>(m, name.data())
        .def(py::init<NonAutnomousFunction>())
        .doc() = "Non-autonomous dynamic system used in nld to represent "
                 "non-autonomous ODEs as a strong type";

    m.def("non_autonomous", [](NonAutnomousFunction fn) {
        std::cout << "non_autonomous" << std::endl;
        std::cout << demangle(typeid(fn).name()) << std::endl;
        return nld::non_autonomous<NonAutnomousFunction>(std::move(fn));
    });
}

template <typename AutonomousFunction>
void wrap_autonomous(py::module &m, std::string_view name) {
    py::class_<nld::autonomous<AutonomousFunction>>(m, name.data())
        .def(py::init<AutonomousFunction>())
        .doc() = "Autonomous dynamic system used in nld to represent "
                 "autonomous ODEs as a strong type";

    m.def("autonomous", [](AutonomousFunction fn) {
        return nld::autonomous<AutonomousFunction>(std::move(fn));
    });
}

template <typename OdeSolver, typename PeriodicFunction>
void wrap_periodic(py::module &m, std::string_view name) {
    using periodic_type = nld::internal::periodic<OdeSolver, PeriodicFunction>;
    using parameters_type =
        typename periodic_parameters_selector<OdeSolver>::type;

    py::class_<periodic_type>(m, name.data())
        .def(py::init<PeriodicFunction, parameters_type>())
        .doc() = "Periodic dynamic system used in nld to represent "
                 "periodic BVP as a strong type";

    m.def(
        "periodic",
        [](PeriodicFunction fn, parameters_type params,
           [[maybe_unused]] OdeSolver solver) {
            return periodic_type(std::move(fn), std::move(params));
        },
        "dynamic_system"_a, "parameters"_a, "solver"_a);
}

void wrap_non_autonomous(py::module &m) {
    auto fs = wrappers::non_autonomous::functions;

    boost::hana::for_each(fs, [&m](auto pair) {
        using function_t = typename decltype(+pair[0_c])::type;
        auto name = "NonAutonomous"s + pair[1_c];

        wrap_non_autonomous<function_t>(m, name.data());
    });
}

void wrap_autonomous(py::module &m) {
    auto fs = wrappers::autonomous::functions;

    boost::hana::for_each(fs, [&m](auto pair) {
        using function_t = typename decltype(+pair[0_c])::type;
        auto name = "Autonomous"s + pair[1_c];

        wrap_autonomous<function_t>(m, name.data());
    });
}

void wrap_periodic(py::module &m) {
    auto nas = boost::hana::make_tuple(wrappers::non_autonomous::functions,
                                       wrappers::solvers);

    boost::hana::for_each(boost::hana::cartesian_product(nas), [&m](auto pair) {
        using function_t = typename decltype(+pair[0_c][0_c])::type;
        using solver_t = typename decltype(+pair[1_c][0_c])::type;
        auto name = "PeriodicNonAutonomous"s + pair[0_c][1_c] + pair[1_c][1_c];

        wrap_periodic<solver_t, nld::non_autonomous<function_t>>(m,
                                                                 name.data());
    });

    auto as = boost::hana::make_tuple(wrappers::autonomous::functions,
                                      wrappers::solvers);

    boost::hana::for_each(boost::hana::cartesian_product(as), [&m](auto pair) {
        using function_t = typename decltype(+pair[0_c][0_c])::type;
        using solver_t = typename decltype(+pair[1_c][0_c])::type;
        auto name = "PeriodicAutonomous"s + pair[0_c][1_c] + pair[1_c][1_c];

        wrap_periodic<solver_t, nld::autonomous<function_t>>(m, name.data());
    });
}

void wrap_systems(py::module &m) {
    wrap_periodic_parameters_constant(m);
    wrap_periodic_parameters_adaptive(m);
    wrap_non_autonomous(m);
    wrap_autonomous(m);
    wrap_periodic(m);
}

} // namespace wrappers

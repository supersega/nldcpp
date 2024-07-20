#include <pybind11/pybind11.h>

#include <functional>
#include <nld/systems.hpp>
#include <variant>

namespace py = pybind11;

namespace wrappers {

namespace non_autonomous {
using RnToRnMapFnDual = std::function<nld::vector_xd(
    const nld::vector_xd & /*u0*/, nld::dual /*t*/)>;

using RnPlusOneToRnMapFnDual = std::function<nld::vector_xd(
    const nld::vector_xd & /*u0*/, nld::dual /*t*/, nld::dual /*lambda*/)>;

using RnPlusNToRnDual =
    std::function<nld::vector_xd(const nld::vector_xd & /*u0*/, nld::dual /*t*/,
                                 const nld::vector_xd & /*lambda*/)>;
} // namespace non_autonomous

using NonAutonomousType = std::variant<
    nld::non_autonomous<::wrappers::non_autonomous::RnToRnMapFnDual>,
    nld::non_autonomous<::wrappers::non_autonomous::RnPlusOneToRnMapFnDual>,
    nld::non_autonomous<::wrappers::non_autonomous::RnPlusNToRnDual>>;

struct NonAutonomous {
    NonAutonomousType system;

    NonAutonomous(::wrappers::non_autonomous::RnToRnMapFnDual fn)
        : system(
              nld::non_autonomous<::wrappers::non_autonomous::RnToRnMapFnDual>(
                  std::move(fn))) {}

    NonAutonomous(::wrappers::non_autonomous::RnPlusOneToRnMapFnDual fn)
        : system(nld::non_autonomous<
                 ::wrappers::non_autonomous::RnPlusOneToRnMapFnDual>(
              std::move(fn))) {}

    NonAutonomous(::wrappers::non_autonomous::RnPlusNToRnDual fn)
        : system(
              nld::non_autonomous<::wrappers::non_autonomous::RnPlusNToRnDual>(
                  std::move(fn))) {}
};

namespace autonomous {
using RnToRnMapFnDual =
    std::function<nld::vector_xd(const nld::vector_xd & /*u0*/)>;
using RnPlusOneToRnMapFnDual = std::function<nld::vector_xd(
    const nld::vector_xd & /*u0*/, const nld::vector_xd & /*lambda*/)>;
} // namespace autonomous

using AutonomousType = std::variant<
    nld::autonomous<::wrappers::autonomous::RnToRnMapFnDual>,
    nld::autonomous<::wrappers::autonomous::RnPlusOneToRnMapFnDual>>;
// TODO: It is not supported by nld::autonomous yet.
// nld::autonomous<RnPlusNToRnDual>>;

struct Autonomous {
    AutonomousType system;

    Autonomous(::wrappers::autonomous::RnToRnMapFnDual fn)
        : system(nld::autonomous<::wrappers::autonomous::RnToRnMapFnDual>(
              std::move(fn))) {}

    Autonomous(::wrappers::autonomous::RnPlusOneToRnMapFnDual fn)
        : system(
              nld::autonomous<::wrappers::autonomous::RnPlusOneToRnMapFnDual>(
                  std::move(fn))) {}
};

struct Periodic {};

void wrap_non_autonomous(py::module &m) {
    py::class_<NonAutonomous>(m, "NonAutonomous")
        .def(py::init<::wrappers::non_autonomous::RnToRnMapFnDual>())
        .def(py::init<::wrappers::non_autonomous::RnPlusOneToRnMapFnDual>())
        .def(py::init<::wrappers::non_autonomous::RnPlusNToRnDual>())
        .doc() = "Non-autonomous dynamic system used in nld to represent "
                 "non-autonomous ODEs as a strong type";
}

void wrap_autonomous(py::module &m) {
    py::class_<Autonomous>(m, "Autonomous")
        .def(py::init<::wrappers::autonomous::RnToRnMapFnDual>())
        .def(py::init<::wrappers::autonomous::RnPlusOneToRnMapFnDual>())
        .doc() = "Autonomous dynamic system used in nld to represent "
                 "autonomous ODEs as a strong type";
}

void wrap_systems(py::module &m) {
    wrap_non_autonomous(m);
    wrap_autonomous(m);
}

} // namespace wrappers

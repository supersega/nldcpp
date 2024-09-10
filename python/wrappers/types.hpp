#pragma once

#include <autodiff/python/bindings/pybind11.hxx>

#include <nld/autocont.hpp>
#include <nld/core.hpp>
#include <nld/systems.hpp>

#include <boost/hana.hpp>

#include <functional>

namespace nld {
inline auto test(double x, double y) {
    return [x, y](const auto &fn, const auto &v) { return std::tuple(x, y); };
}
} // namespace nld

namespace wrappers {

namespace non_autonomous {
using RnToRnMapFnDual = std::function<nld::vector_xdd(
    const nld::vector_xdd & /*u0*/, nld::dual /*t*/)>;

using RnPlusOneToRnMapFnDual = std::function<nld::vector_xdd(
    const nld::vector_xdd & /*u0*/, nld::dual /*t*/, nld::dual /*lambda*/)>;

using RnPlusNToRnMapDual = std::function<nld::vector_xdd(
    const nld::vector_xdd & /*u0*/, nld::dual /*t*/,
    const nld::vector_xdd & /*lambda*/)>;

// Non-autonomous functions
inline auto rn_to_rn_map_dual = boost::hana::make_tuple(
    boost::hana::type_c<non_autonomous::RnToRnMapFnDual>, "RnToRnMapDual");
inline auto rn_plus_one_to_rn_map_dual = boost::hana::make_tuple(
    boost::hana::type_c<non_autonomous::RnPlusOneToRnMapFnDual>,
    "RnPlusOneToRnMapDual");
inline auto rn_plus_n_to_rn_map_dual = boost::hana::make_tuple(
    boost::hana::type_c<non_autonomous::RnPlusNToRnMapDual>,
    "RnPlusNToRnMapDual");

inline auto functions = boost::hana::make_tuple(
    rn_to_rn_map_dual, rn_plus_one_to_rn_map_dual, rn_plus_n_to_rn_map_dual);

} // namespace non_autonomous

namespace autonomous {
using RnToRnMapFnDual =
    std::function<nld::vector_xdd(const nld::vector_xdd & /*u0*/)>;
using RnPlusOneToRnMapFnDual = std::function<nld::vector_xdd(
    const nld::vector_xdd & /*u0*/, const nld::vector_xdd & /*lambda*/)>;

// Autonomous functions
inline auto rn_to_rn_map_dual = boost::hana::make_tuple(
    boost::hana::type_c<autonomous::RnToRnMapFnDual>, "RnToRnMapDual");
inline auto rn_plus_one_to_rn_map_dual = boost::hana::make_tuple(
    boost::hana::type_c<autonomous::RnPlusOneToRnMapFnDual>,
    "RnPlusOneToRnMapDual");

inline auto functions =
    boost::hana::make_tuple(rn_to_rn_map_dual, rn_plus_one_to_rn_map_dual);

} // namespace autonomous

// Solvers
inline auto runge_kutta_4 = boost::hana::make_tuple(
    boost::hana::type_c<nld::runge_kutta_4>, "RungeKutta4");
inline auto runge_kutta_45 = boost::hana::make_tuple(
    boost::hana::type_c<nld::runge_kutta_45>, "RungeKutta45");
inline auto solvers = boost::hana::make_tuple(runge_kutta_4, runge_kutta_45);

#define CONCAT(a, b) a##b
#define TEXTIFY(a) #a

#define DECLARE_MAPPER_STRUCT(mapper_name, map_result, pyname, ...)            \
    template <typename... Args>                                                \
    struct CONCAT(mapper_name, _mapper) {                                      \
        using args = std::tuple<Args...>;                                      \
        using fn = decltype(std::apply(std::function{nld::mapper_name},        \
                                       std::declval<args>()));                 \
        using result = map_result;                                             \
                                                                               \
        CONCAT(mapper_name, _mapper)                                           \
        (Args... args)                                                         \
            : function{nld::mapper_name(args...)}, arguments(args...) {}       \
                                                                               \
        auto operator()(const auto &fn, const auto &v) const {                 \
            return std::invoke(function, fn, v);                               \
        }                                                                      \
                                                                               \
        args arguments;                                                        \
        fn function;                                                           \
    };                                                                         \
    using mapper_name = CONCAT(mapper_name, _mapper)<__VA_ARGS__>;             \
                                                                               \
    inline auto CONCAT(mapper_name, _tag) = boost::hana::make_tuple(           \
        boost::hana::type_c<mapper_name>, pyname, TEXTIFY(mapper_name));

using point2d_t = std::tuple<double, double>;

DECLARE_MAPPER_STRUCT(solution, nld::vector_xd, "Solution")
DECLARE_MAPPER_STRUCT(point2d, point2d_t, "Point2D", nld::index, nld::index)
DECLARE_MAPPER_STRUCT(unknown, double, "Unknown", nld::index)
DECLARE_MAPPER_STRUCT(mean_amplitude, double, "MeanAmplitude", nld::index)
DECLARE_MAPPER_STRUCT(half_swing, double, "HalfSwing", nld::index)
DECLARE_MAPPER_STRUCT(monodromy, nld::matrix_xd, "Monodromy")
DECLARE_MAPPER_STRUCT(test, point2d_t, "Test", double, double)

#undef DECLARE_MAPPER_STRUCT
#undef CONCAT
#undef TEXTIFY

inline auto mapper_tags = boost::hana::make_tuple(
    solution_tag, point2d_tag, unknown_tag, mean_amplitude_tag, half_swing_tag,
    monodromy_tag, test_tag);

inline auto periodic_mapper_two_tags =
    boost::hana::make_tuple(solution_tag, mean_amplitude_tag);

inline auto periodic_mapper_three_tags =
    boost::hana::make_tuple(solution_tag, mean_amplitude_tag, monodromy_tag);

template <typename... MaperTypes>
struct concat_mappers {
    using result = std::tuple<typename MaperTypes::result...>;
    using mappers_t = std::tuple<MaperTypes...>;

    concat_mappers(MaperTypes... mappers) : mappers(mappers...) {}

    auto operator()(const auto &fn, const auto &v) const {
        return std::apply(
            [&fn, &v](auto... mappers) {
                return std::tuple(mappers(fn, v)...);
            },
            mappers);
    }

    mappers_t mappers;
};

} // namespace wrappers

#pragma once

#include "nld/core/utils.hpp"
#include <nld/core.hpp>
#include <type_traits>

namespace nld {
namespace concepts::non_autonomous {
template <typename F>
concept RnToRnMap = requires(F f) {
    {
        f(std::declval<nld::utils::any_type>(),
          std::declval<nld::utils::any_type>())
    } -> Vector;
};

template <typename F>
concept RnPlusOneToRnMap = requires(F f) {
    {
        f(std::declval<nld::utils::any_type>(),
          std::declval<nld::utils::any_type>(),
          std::declval<nld::utils::any_type>())
    } -> Vector;
};

template <RnToRnMap Fn>
auto call_fn() -> decltype(std::declval<Fn>()(nld::utils::any_type{},
                                              nld::utils::any_type{}));

template <RnPlusOneToRnMap Fn>
auto call_fn() -> decltype(std::declval<Fn>()(nld::utils::any_type{},
                                              nld::utils::any_type{},
                                              nld::utils::any_type{}));
} // namespace concepts::non_autonomous

/// @brief Non autonomous dynamic system
/// Function should have signature f(u, t, lambda) where:
/// u - vector of unknown variables
/// t - time
/// lambda - continuation parameter
template <typename Fn>
struct non_autonomous final {
    /// TODO: investigate if continuation wrt non period requires
    /// Integral-Phase cindition, and make better name.
    static constexpr bool is_continuation_wrt_period = true;

    using vector_t = decltype(concepts::non_autonomous::call_fn<Fn>());

    /// @brief Make non autonomous system.
    explicit non_autonomous(Fn f) : function(f) {}

    /// @brief Operator to wrap function for earthier usage.
    /// @details For now assume that Period or frequence is passed
    /// explicitly, e.g. y' = f(t, t, T).
    /// @param y State variables for ODE.
    /// @param t Time.
    /// @param parameters Parameters for continuation.
    /// @returns Value of ODE right side at given point, time, parameters.
    template <typename T, typename S, typename P>
    auto operator()(const T &y, S t, P parameters) const
        requires concepts::non_autonomous::RnPlusOneToRnMap<Fn>
    {
        return function(y, t, parameters);
    }

    /// @brief Operator to wrap function for earthier usage.
    /// @details Wrap syster y' = f(y, t),
    /// assuming continuation wrt period.
    /// @param y State variables for ODE.
    /// @param t Time.
    /// @param T Period.
    /// @returns Value of ODE right side at given point, time, parameters.
    template <typename Ty, typename S, typename P>
    auto operator()(const Ty &y, S t, P T) const
        requires concepts::non_autonomous::RnToRnMap<Fn>
    {
        auto result = function(y, t);
        result *= T;
        return result;
    }

private:
    Fn function; ///< Underlying function represented dynamic system.
};

template <typename T>
struct is_non_autonomous : std::false_type {};

template <typename Fn>
struct is_non_autonomous<non_autonomous<Fn>> : std::true_type {};

template <typename T>
inline constexpr auto is_non_autonomous_v = is_non_autonomous<T>::value;

namespace concepts::autonomous {
template <typename F>
concept RnToRnMap = requires(F f) {
    { f(std::declval<nld::utils::any_type>()) } -> Vector;
};

template <typename F>
concept RnPlusOneToRnMap = requires(F f) {
    {
        f(std::declval<nld::utils::any_type>(),
          std::declval<nld::utils::any_type>())
    } -> Vector;
};

template <RnToRnMap Fn>
auto call_fn() -> decltype(std::declval<Fn>()(nld::utils::any_type{}));

template <RnPlusOneToRnMap Fn>
auto call_fn() -> decltype(std::declval<Fn>()(nld::utils::any_type{},
                                              nld::utils::any_type{}));
} // namespace concepts::autonomous

/// @brief Autonomous dynamic system
/// Function should have signature f(u, t, lambda) where:
/// u - vector of unknown variables
/// t - time
/// lambda - continuation parameter
template <typename Fn>
struct autonomous final {
    /// @brief indentify if period is a continuation parameter
    static constexpr bool is_continuation_wrt_period =
        concepts::autonomous::RnToRnMap<Fn>;

    using vector_t = decltype(concepts::autonomous::call_fn<Fn>());

    /// @brief Make autonomous system.
    explicit autonomous(Fn f) : function(f) {}

    /// @brief Operator to wrap function for earthier usage.
    /// @param y State variables for ODE.
    /// @param t Time.
    /// @param T Period.
    /// @param parameters Parameters for continuation.
    /// @returns Value of ODE right side at given point, time, parameters.
    template <typename Y, typename S, typename Pd, typename P>
    auto operator()(const Y &y, S t, Pd T, P parameters) const
        requires concepts::autonomous::RnPlusOneToRnMap<Fn>
    {
        auto result = function(y, parameters);
        // Time variable transformation t -> t / period,
        // To be able to evaluate period in each iteration
        result *= T;
        return result;
    }

    /// @brief Operator to wrap function for earthier usage.
    /// @param y State variables for ODE.
    /// @param t Time.
    /// @param T Period.
    /// @param parameters Parameters for continuation.
    /// @returns Value of ODE right side at given point, time, parameters.
    template <typename Y, typename S, typename Pd>
    auto operator()(const Y &y, S t, Pd T) const
        requires concepts::autonomous::RnToRnMap<Fn>
    {
        auto result = function(y);
        // Time variable transformation t -> t / period,
        // To be able to evaluate period in each iteration
        result *= T;
        return result;
    }

private:
    Fn function; ///< Underlying function represented dynamic system.
};

template <typename T>
struct is_autonomous : std::false_type {};

template <typename Fn>
struct is_autonomous<autonomous<Fn>> : std::true_type {};

template <typename T>
inline constexpr auto is_autonomous_v = is_autonomous<T>::value;
} // namespace nld

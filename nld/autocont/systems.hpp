#pragma once

#include <type_traits>

namespace nld {
/// @brief Non autonomous dynamic system
/// Function should have signature f(u, t, lambda) where:
/// u - vector of unknown variables
/// t - time
/// lambda - continuation parameter
template <typename Fn> struct non_autonomous final {
  using vector_t = decltype(std::declval<Fn>()(
      nld::utils::any_type{}, nld::utils::any_type{}, nld::utils::any_type{}));

  /// @brief Make non autonomous system.
  explicit non_autonomous(Fn f) : function(f) {}

  /// @brief Operator to wrap function for earthier usage.
  /// @param y State variables for ODE.
  /// @param t Time.
  /// @param parameters Parameters for continuation.
  /// @returns Value of ODE right side at given point, time, parameters.
  template <typename T, typename S, typename P>
  auto operator()(const T &y, S t, P parameters) const {
    return function(y, t, parameters);
  }

private:
  Fn function; ///< Underlying function represented dynamic system.
};

template <typename T> struct is_non_autonomous : std::false_type {};

template <typename Fn>
struct is_non_autonomous<non_autonomous<Fn>> : std::true_type {};

template <typename T>
inline constexpr auto is_non_autonomous_v = is_non_autonomous<T>::value;

/// @brief Autonomous dynamic system
/// Function should have signature f(u, t, lambda) where:
/// u - vector of unknown variables
/// t - time
/// lambda - continuation parameter
template <typename Fn> struct autonomous final {
  using vector_t = decltype(std::declval<Fn>()(nld::utils::any_type{},
                                               nld::utils::any_type{}));

  /// @brief Make autonomous system.
  explicit autonomous(Fn f) : function(f) {}

  /// @brief Operator to wrap function for earthier usage.
  /// @param y State variables for ODE.
  /// @param t Time.
  /// @param T Period.
  /// @param parameters Parameters for continuation.
  /// @returns Value of ODE right side at given point, time, parameters.
  template <typename Y, typename S, typename Pd, typename P>
  auto operator()(const Y &y, S t, Pd T, P parameters) const {
    auto result = function(y, parameters);
    // Time variable transformation t -> t / period,
    // To be able to evaluate period in each iteration
    result *= T;
    return result;
  }

private:
  Fn function; ///< Underlying function represented dynamic system.
};

template <typename T> struct is_autonomous : std::false_type {};

template <typename Fn> struct is_autonomous<autonomous<Fn>> : std::true_type {};

template <typename T>
inline constexpr auto is_autonomous_v = is_autonomous<T>::value;
} // namespace nld

#pragma once

#include <nld/core.hpp>
#include <nld/math/concepts.hpp>

#include <tuple>

namespace nld {
/// @brief Calculate integral using integration traits.
/// @tparam I Type what implements integration traits.
/// @tparam F Function to integrate.
/// @tparam D Domain.
/// @tparam T Additional arguments.
/// @param function Function to integrate.
/// @param domain Integration domain.
/// @param options Integration options.
/// @param args Additional arguments.
/// @return Value of the integral.
template <typename I, typename F, Domain1d D, typename O,
          typename T = std::tuple<>>
inline auto integrate(F function, D domain, O options,
                      T &&args = no_arguments()) {
    return I::integrate(function, domain, options, std::forward<T>(args));
}
} // namespace nld

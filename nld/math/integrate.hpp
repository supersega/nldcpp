#pragma once

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
template <typename I, typename F, typename D, typename T = std::tuple<>>
inline auto integrate(F function, D domain, integration_options options = integration_options{}, T&& args = no_arguments()) {
    return I::integrate(function, domain, options, args);
}
}
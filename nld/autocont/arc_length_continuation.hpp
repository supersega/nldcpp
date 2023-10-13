#pragma once

#include <nld/autocont/arc_length_raw.hpp>
#include <nld/autocont/arc_length_representation.hpp>
#include <nld/autocont/continuation_parameters.hpp>
#include <nld/autocont/dimension.hpp>
#include <nld/autocont/first_guess.hpp>

#include <cppcoro/generator.hpp>

namespace nld {
/// @brief Async function which solves continuation problem for nonlinear function using arc-length continuation.
/// @details Use this function if you already know initial guess and point.
/// This function requires initial point (see unknowns),
/// and tangent (see tangential), for continuation. Use map to transform
/// solution e.g. project on 2d plane, or half swing for periodic problem.
/// Example of usage:
/// @code
/// // Assume that we have equilibrium problem
/// auto problem(nld::vector_xdd& x, dual lambda) -> nld::vector_xdd;
/// 
/// int main() {
///     nld::vector_xdd x = ...; ///< Some code to eval initial guess.
///     nld::vector_xdd t = ...; ///< Some code to eval tangential.
///
///     for (auto solution : nld::arc_length(nld::equilibrium(problem), x, t, nld::solution()) {
///         ///< Do something with solution, e.g. dump to file, or collect into some types and make plot
///     }
/// }
/// @endcode
/// @param function Parametric function which we want to continue.
/// @param parameters Continuation parameters.
/// @param unknowns Initial guess vector which has structure (u, lambda)
/// @param tangential Initial guess vector which has structure (u, lambda)
/// @param map Function to map solution f: (V, R) -> MappedType.
/// @return Generator which yields solution on iteration.
template<typename F, typename V, typename T, typename M>
auto arc_length(F&& function, nld::continuation_parameters parameters, V unknowns, T tangential, M map) {
    return nld::internal::arc_length_raw(std::forward<F>(function), parameters, unknowns, tangential, map);
}

/// @brief Solves continuation problem for nonlinear function using arc length continuation.
/// @details This version of function will pass tangential as unit vector with N-1 zeros and 
/// last element equal to one.
/// @param function Parametric function which we want to continue.
/// @param size The size of nonlinear problem.
/// @param parameters Continuation parameters.
/// @param map Function to map solution f: (V, R) -> MappedType.
/// @return Generator which yields solution on iteration.
template<typename F, typename V, typename M>
auto arc_length(F&& function, nld::continuation_parameters parameters, V unknowns, M map) {
    V tangential = V::Zero(unknowns.size());
    tangential(unknowns.size() - 1) = 1.0;
    return nld::arc_length(std::forward<F>(function), parameters, unknowns, tangential, map);
}

/// @brief Solves continuation problem for nonlinear function using arc length continuation.
/// @details This version of function will evaluate initial guess by itself. Only dimension
/// of problem is required for algorithm, since size is unknown from nld types.
/// @param function Parametric function which we want to continue.
/// @param size The size of nonlinear problem.
/// @param parameters Continuation parameters.
/// @param map Function to map solution f: (V, R) -> MappedType.
/// @return Generator which yields solution on iteration.
template<typename F, typename M>
auto arc_length(F&& function, dimension size, nld::continuation_parameters parameters, M map) {
    auto [unknowns, tangential] = nld::evaluate_firs_guess(function, size);
    return nld::arc_length(std::forward<F>(function), parameters, unknowns, tangential, map);
}
} /// end namespace nld

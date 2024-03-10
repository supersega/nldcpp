#pragma once

#include <nld/core.hpp>

#include <nld/autocont/arc_length_representation.hpp>
#include <nld/autocont/continuation_parameters.hpp>
#include <nld/autocont/step_updater.hpp>

/// @note internal namespace should not be used in client code
namespace nld::internal {
/// @brief Raw version of arc length continuation algorithm.
/// @details This function requires initial point see unknowns,
/// and tangent see tangential, for continuation. Use map to transform
/// solution e.g. project on 2d plane, or half swing for periodic problem.
/// @param function Parametric function which we want to continue.
/// @param parameters Continuation parameters.
/// @param unknowns Initial guess vector which has structure (u, lambda)
/// @param tangential Initial guess vector which has structure (u, lambda)
/// @param map Function to map solution f: (V, R) -> MappedType.
/// @return Generator which yields solution on iteration.
/// @note This takes F as a value, because coroutines can't handle
/// references properly, for more details see:
/// https://toby-allsopp.github.io/2017/04/22/coroutines-reference-params.html
template <typename F, Vector V, Vector T, typename M>
auto arc_length_raw(F function, nld::continuation_parameters parameters,
                    V unknowns, T tangential, M map) noexcept
    -> cppcoro::generator<decltype(map(std::declval<F>(), std::declval<V>()))> {
    auto [newton_parameters, tail, min_step, max_step, dir] = parameters;

    tangential *= static_cast<double>(dir);
    nld::step_updater updater(
        nld::step_updater_parameters{min_step, max_step, 1.67, 4});
    nld::arc_length_representation problem(std::forward<F>(function), unknowns,
                                           tangential, updater.step());

    V old_unknowns;
    T old_tangential;

    while ((tail -= updater.step()) > 0) {
        // Correct solution
        if (auto info = nld::newton(problem, unknowns, newton_parameters);
            info) {
            tangential = problem.tangential(unknowns);
            problem.set_arc_length_properties(unknowns, tangential,
                                              updater.step());

            co_yield map(problem.underlying_function(), unknowns);

            // Store old point to fallback on failed iteration.
            old_unknowns = unknowns;
            old_tangential = tangential;

            // Predicate solution for next step
            unknowns += updater.step() * tangential;

            // We was able to calculate this point, so let us try to make
            // continuation step bigger.
            updater.increase_step_if_possible();
        } else {
            // Minimal step reached, we have to stop continuation.
            if (!updater.decrease_step())
                break;

            // Fallback to previous point with smaller continuation step.
            problem.set_arc_length_properties(old_unknowns, old_tangential,
                                              updater.step());
            unknowns = old_unknowns;
        }
    }
}
} // namespace nld::internal

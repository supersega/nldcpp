#pragma once

#include <nld/autocont/continuation_parameters.hpp>
#include <nld/autocont/utils.hpp>

#include <cppcoro/generator.hpp>

namespace nld {
/// @brief Async function which solves continuation problem for nonlinear function using Moore-Penrose continuation.
/// TODO: Add user defined Jacobian (??), try numerical differentiation.
/// @param f Parametric function which we want to continue.
/// @param p Continuation parameters.
/// @param v Initial guess.
/// @param r Initial parameter.
/// @param map Function to map solution f: (V, R) -> MappedType.
/// @return generator which yields solution on iteration.
template<typename F, typename P, typename V, typename R, typename M>
auto moore_penrose(F f, P&& p, V&& v, R r, M map) -> cppcoro::generator<decltype(map(std::declval<V>(), std::declval<R>()))> {
    using nld::math::moore_penrose_newton;
    using Vector = std::remove_reference_t<V>;

    auto [newton_params, tail, step, max_step, direction] = p;
    auto dir = static_cast<int>(direction);
    auto min_step = step;

    auto vars = v;
    auto param = r;

    Vector tan = Vector::Zero(vars.size());
    R param_tan = 1.0;

    while ((tail -= step) > 0)
    {
        // predicate solution on this step
        vars += dir * step * tan;
        param += dir * step * param_tan;

        // correct solution
        if (auto info = moore_penrose_newton(f, wrt(vars, param), wrt(tan, param_tan), at(vars, param), newton_params); info) {
            step = detail::updated_step(info.number_of_done_iterations, step, min_step, max_step);
            co_yield map(vars, param);
        }
        else {
            break;
        }
    }
}

/// @brief Same as above, but returns whole solution
template<typename F, typename P, typename V, typename R>
auto moore_penrose(F f, P&& p, V&& v, R r) {
    return moore_penrose(f, std::forward<P>(p), std::forward<V>(v), r, [](V v, auto r){ return std::tuple(v, r); });
}
} /// end namespace nld

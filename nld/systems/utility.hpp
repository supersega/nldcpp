#pragma once

#include <nld/collocations/mesh.hpp>
#include <nld/systems/periodic_collocations.hpp>

namespace nld {

/// @brief make collocation unknowns from initial conditions.
/// @param fn perio.
/// @param parameters mesh parameters.
/// @param initial_conditions initial conditions and parameter.
/// @param dimension dimension of the problem.
/// @return vector of unknowns on uniform mesh.
template <typename Fn, typename Basis>
auto make_collocation_unknowns(const nld::periodic_collocations<Fn, Basis> &fn,
                               const nld::vector_xdd &initial_conditions) {
    auto dim = initial_conditions.size() - 1;
    auto N = fn.mesh_parameters().intervals;
    auto m = fn.mesh_parameters().collocation_points;
    auto n = dim;
    auto size = N * m * dim + dim;

    nld::vector_xdd u0 = initial_conditions.head(n);

    nld::vector_xdd u(N * n * m + (N - 1) * n + n + 1);
    u.head(n) = initial_conditions.head(n);
    u(u.size() - 1) = initial_conditions(n);

    const auto &ode = fn.underlying_function();

    double t = 0.0;
    double dt = 1.0 / (N * (m + 1));

    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < m + 1; ++k) {
            auto idx = j * (m + 1) * n + k * n;
            u.segment(j * (m + 1) * n + k * n, n) = u0;

            u0 = nld::math::runge_kutta_4::end_solution(
                ode, nld::math::constant_step_parameters{t, t + dt, 1}, u0,
                std::tuple(nld::vector_xdd{u.tail(1)}));
            t += dt;
        }

        if (j < N - 1) {
            u.segment(j * (m + 1) * n + (m)*n, n) = u0;
        }
    }

    return u;
}

} // namespace nld

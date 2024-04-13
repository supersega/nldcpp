#pragma once

#include <array>
#include <nld/core.hpp>
#include <nld/systems/ode.hpp>
#include <nld/systems/periodic_parameters.hpp>
#include <nld/systems/problem.hpp>

namespace nld {
namespace internal {

inline constexpr index parameters = 2;

template <typename F, typename Wrt, typename At>
auto hessians(const F &f, Wrt &&wrt, At &&at) -> tensor_3d {
    auto &dw = std::get<0>(wrt);
    const auto full_size = dw.size();

    auto v = std::apply(f, at);

    const auto dim = v.size();

    tensor_3d Hs(dim, dim, dim + parameters);

    for (index i = 0; i < dim; i++) {
        dw[i].grad = 1.0;

        for (index j = i; j < dim; j++) {
            dw[j].val.grad = 1.0;
            v = std::apply(f, at);

            for (index k = 0; k < dim; k++) {
                Hs(k, j, i) = v[k].grad.grad;
                Hs(k, i, j) = Hs(k, j, i);
            }

            dw[j].val.grad = 0.0;
        }

        dw[i].grad = 0.0;
    }

    // 2nd Derivatives wrt parameters
    for (index i = 0; i < parameters; i++) {
        dw[full_size - parameters + i].grad = 1.0;

        for (index j = 0; j < dim; j++) {
            dw[j].val.grad = 1.0;
            v = std::apply(f, at);

            for (index k = 0; k < dim; k++)
                Hs(k, j, i + dim) = v[k].grad.grad;

            dw[j].val.grad = 0.0;
        }

        dw[full_size - 2 + i].grad = 0.0;
    }

    return Hs;
}

template <OdeSolver S, typename Ds>
struct saddle_node;

/// @brief Make integration arguments for periodic problem.
/// @param variables Continuation problem variables.
/// @param problem Periodic problem.
/// @return Integration arguments as a tuple<Vector, tuple<Scalar>> for non
/// autonomous and tuple<Vector, tuple<Scalar, Scalar>> for autonomous.
template <typename S, typename Fn, typename V>
auto integration_arguments(const V &variables,
                           const saddle_node<S, Fn> &problem);

/// @brief Saddle node bifurcation nonlinear problem.
/// @details Standard extended boundary value problem used for
/// saddle-node bifurcation detection. This class requires at
/// least 2nd order derivable autiodiff::numbers. This class
/// can be used for continuation of bifurcations on parametric
/// plane. TODO: Add math description for extended simplifications.
template <OdeSolver S, typename Ds>
struct saddle_node final : nld::problem<Ds> {
    using problem_t = nld::problem<Ds>;
    using solver_t = S;
    using integration_parameters_t =
        typename solver_t::integration_parameters_t;

    /// @brief Construct saddle node.
    /// @param ds dynamic system which we want analyze saddle node bifurcation.
    /// @param dim system dimension.
    /// @param prams periodic problem parameters.
    /// @param n coordinate number where saddle node bifurcation happened.
    explicit saddle_node(Ds &&ds, nld::periodic_parameters_constant params)
        : problem_t(std::forward<Ds>(ds)),
          parameters(0, 1.0 * params.periods, params.intervals) {
        static_assert(is_non_autonomous_v<Ds>,
                      "Wrong function type for saddle_node see docs");
    }

    /// @brief system integration parameters.
    auto integration_parameters() const -> integration_parameters_t {
        return parameters;
    }

    /// @brief Get ODEs and state variables.
    /// @param variables problem unknowns (initial conditions + other
    /// parameters).
    /// @return ODE and initial conditions from variables.
    template <typename Vector>
    auto ode(const Vector &variables) const {
        using namespace std::placeholders;

        auto parameters_on_plane = variables.tail(internal::parameters);
        auto ds =
            std::bind(this->function.function, _1, _2, parameters_on_plane);
        const auto states = (variables.size() - internal::parameters) / 2;
        Vector state = variables.head(states);

        return std::tuple(ds, state);
    }

    /// @brief Add function like access.
    /// @details Saddle node bifurcation nonlinear system.
    /// @param variables First N are initial conditions for ODE, next N are
    /// eigen values, last are two parameters.
    /// @return value of system at given point.
    template <typename Vector>
    typename std::enable_if_t<is_non_autonomous_v<Ds>, Vector>
    operator()(Vector &variables) const {
        using namespace std::placeholders;

        const auto dim = variables.size();

        auto solution_at_end = end_solution();

        Vector value(dim - 1);
        const auto states = (dim - internal::parameters) / 2;
        auto state = variables.head(states);

        value.head(states) = solution_at_end(variables) - state;

        auto dyTdy0 = autodiff::forward::jacobian(solution_at_end, wrt(state),
                                                  at(variables));
        auto phi = variables.segment(states, states);

        value.segment(states, states) = dyTdy0 * phi - phi;
        value(dim - 2) = phi.squaredNorm() - 1.0;

        return value;
    }

    /// @brief Jacobian of saddle-node bifurcation detection system.
    /// @param wrt eval Jacobian w.r.t. variables.
    /// @param at eval Jacobian at point.
    /// @param v value of function.
    /// @return Jacobian of saddle-node bifurcation detection system.
    template <typename At, typename Result>
    auto jacobian(At &at, Result &v) const -> matrix_xd {
        using Vector = decltype(std::apply(*this, nld::at(at)));

        auto &variables = at;
        v = std::apply(*this, nld::at(at));
        const auto dim = variables.size();
        const auto states = (dim - internal::parameters) / 2;

        auto solution_at_end = end_solution();

        nld::matrix_xd J = nld::matrix_xd::Zero(dim - 1, dim);

        auto dyTdy0 = autodiff::forward::jacobian(solution_at_end, nld::wrt(at),
                                                  nld::at(at));

        auto Hs =
            internal::hessians(solution_at_end, nld::wrt(at), nld::at(at));
        vector_xd phi =
            variables.segment(states, states).template cast<double>();
        std::array dimensions = {Eigen::IndexPair(1, 0)};
        tensor_2d tH = Hs.contract(nld::utils::tensor_view(phi), dimensions);
        Eigen::Map<matrix_xd> dJphidy0(tH.data(), states,
                                       states + internal::parameters);

        J.topLeftCorner(states, dim) = dyTdy0;
        J.topLeftCorner(states, states) -=
            nld::matrix_xd::Identity(states, states);
        J.block(states, states, states, states) =
            J.topLeftCorner(states, states);
        J.block(states, 0, states, states) =
            dJphidy0.topLeftCorner(states, states);
        J.block(states, states + states, states, internal::parameters) =
            dJphidy0.bottomRightCorner(states, internal::parameters);
        J.row(dim - internal::parameters).segment(states, states) = 2.0 * phi;

        return J;
    }

private:
    auto end_solution() const {
        return [this]<typename Vector>(const Vector &variables) {
            auto [state, args] = integration_arguments(variables, *this);
            return solver_t::end_solution(this->function, parameters, state,
                                          args);
        };
    }

    integration_parameters_t parameters; ///< Integration parameters.
};

template <typename S, typename Fn, typename V>
auto integration_arguments(const V &variables,
                           [[maybe_unused]] const saddle_node<S, Fn> &sn) {
    const auto states = (variables.size() - internal::parameters) / 2;
    V state = variables.head(states);
    V parameters_on_plane = variables.tail(internal::parameters);
    return std::make_tuple(state, std::make_tuple(parameters_on_plane));
}
} // namespace internal

/// @brief Periodic boundary value problem wrapper maker function.
/// @param dynamic_system
/// \f$\dot{q} = \theta(q, t, \lambda)\f$.
/// @param parameters parameters of ODE integration
template <OdeSolver S, typename Ds>
auto saddle_node(Ds dynamic_system, periodic_parameters_constant parameters) {
    return internal::saddle_node<S, Ds>(std::move(dynamic_system),
                                        std::move(parameters));
}

template <OdeSolver S, typename Ds>
struct is_simple_shooting_discretization<internal::saddle_node<S, Ds>>
    : std::true_type {};

} // namespace nld

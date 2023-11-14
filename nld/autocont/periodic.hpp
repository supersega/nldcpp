#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

#include <nld/autocont/integration_arguments.hpp>
#include <nld/autocont/jacobian_mixin.hpp>
#include <nld/autocont/periodic_base.hpp>
#include <nld/autocont/periodic_parameters.hpp>
#include <nld/autocont/problem.hpp>
#include <nld/autocont/systems.hpp>

namespace nld {
namespace internal {

template <typename Ds>
concept PoincareEquationNeeded = Ds::is_poincare_equation_needed;

template <OdeSolver S, typename Ds>
struct periodic;

/// @brief Make integration arguments for periodic problem.
/// @param variables Continuation problem variables.
/// @param problem Periodic problem.
/// @return Integration arguments as a tuple<Vector, tuple<Scalar>> for non
/// autonomous and tuple<Vector, tuple<Scalar, Scalar>> for autonomous.
template <typename S, typename Fn, typename V>
auto integration_arguments(const V &variables, const periodic<S, Fn> &problem);

/// @brief Class which represent boundary value problem for ODE periodic
/// solution.
/// @details This class is in internal namespace since we can't deduce template
/// arguments. User defined deduction guide does not work too.
/// Two different types of periodic problems are represented by this class.
/// First is periodic problem for non autonomous ODE like:
/// \f$\dot{q} = \theta(q, t, \lambda)\f$.
/// Second one is for autonomous ode:
/// \f$\dot{q} = \theta(q, \lambda)\f$.
/// For second type of problem Pincare equation is added:
/// (q(0) - q_0(0)) / dot{q_0(0)} = 0
/// to determinate the period.
/// @see concepts.hpp for OdeSolver concept details.
template <OdeSolver S, typename Ds>
struct periodic final : nld::jacobian_mixin<periodic<S, Ds>>,
                        nld::periodic_base<Ds> {
private:
    using problem_t = nld::problem<Ds>;
    using dynamic_system_t = Ds;
    using vector_t = typename dynamic_system_t::vector_t;
    using integration_parameters_t = typename S::integration_parameters_t;
    using periodic_base_t = nld::periodic_base<Ds>;

public:
    using solver_t = S;

    /// @brief Constructor which create boundary value for periodic solution.
    /// @param ds Function which rep dynamic system (ODE).
    /// \f$\dot{q} = \theta(q, t, \lambda)\f$.
    /// @param parameters Parameters of boundary value problem.
    template <typename P>
    explicit periodic(Ds ds, P parameters)
        : periodic_base_t(std::forward<Ds>(ds)),
          parameters(parameters.to_integration_parameters()) {
        static_assert(
            is_non_autonomous_v<Ds> || is_autonomous_v<Ds>,
            "Wrong type for two_point_boundary_value_problem see docs");
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
        if constexpr (!PoincareEquationNeeded<Ds>) {
            auto parameter = variables(variables.size() - 1);
            auto ds = std::bind(this->underlying_function(), _1, _2, parameter);
            Vector state = variables.head(variables.size() - 1);

            return std::tuple(ds, state);
        } else {
            auto parameter = variables(variables.size() - 1);
            auto period = variables(variables.size() - 2);
            auto ds = std::bind(this->underlying_function(), _1, _2, period,
                                parameter);
            Vector state = variables.head(variables.size() - 2);

            return std::tuple(ds, state);
        }
    }

    /// @brief Add function like access.
    /// @details Two point boundary value problem for non-autonomous ODE y' =
    /// f(y(t), t), where f(x, t + T) = f(x, t), T is period.
    /// @param variables Initial conditions for ODE.
    template <typename Vector>
    Vector operator()(const Vector &variables) const
        requires(!PoincareEquationNeeded<Ds>)
    {
        auto [state, args] = integration_arguments(variables, *this);
        return solver_t::end_solution(this->underlying_function(), parameters,
                                      state, args) -
               state;
    }

    /// @brief Add function like access.
    /// @details Two point boundary value problem for autonomous ODE y' =
    /// f(y(t)), where Poincare equation added for period computation.
    /// @param variables Initial conditions for ODE.
    template <typename Vector>
    Vector operator()(const Vector &variables) const
        requires PoincareEquationNeeded<Ds>
    {
        auto dim = variables.size();
        Vector result(dim - 1);

        auto [state, args] = integration_arguments(variables, *this);
        // Boundary value problem right parts that corresponds to q(1) - q(0) =
        // 0
        result.head(dim - 2) =
            solver_t::end_solution(this->underlying_function(), parameters,
                                   state, args) -
            state;
        // Previous q_0(0)
        Vector prev = this->previous_solution().head(
            this->previous_solution().size() - 2);
        // Phase conditions, now it is Poincare condition for trajectories in
        // phase space (q(0) - q_0(0)) d(q_0(0))/dt = 0
        // TODO: Use integral conditions to make computations better
        result(dim - 2) =
            (state - prev)
                .dot(nld::math::detail::evaluate(this->underlying_function(),
                                                 prev, 0.0, args));

        return result;
    }

    /// @brief Number of non state variables
    /// @returns Number of non state variables, non autonomous (lambda),
    /// autonomous (T, lambda)
    static constexpr index non_state_variables() {
        return PoincareEquationNeeded<Ds> ? 2 : 1;
    }

private:
    integration_parameters_t parameters; ///< ODE integration parameters.
};

template <typename S, typename Fn, typename V>
auto integration_arguments(
    const V &variables,
    [[maybe_unused]] const nld::internal::periodic<S, Fn> &pc) {
    if constexpr (!PoincareEquationNeeded<Fn>) {
        auto dim = variables.size();
        V state = variables.head(variables.size() - 1);
        auto parameter = variables(dim - 1);
        return std::make_tuple(state, std::make_tuple(parameter));
    } else {
        auto dim = variables.size();
        V state = variables.head(variables.size() - 2);
        auto period = variables(dim - 2);
        auto parameter = variables(dim - 1);
        return std::make_tuple(state, std::make_tuple(period, parameter));
    }
}
} // namespace internal

/// @brief Periodic boundary value problem wrapper maker function.
/// @param dynamic_system
/// \f$\dot{q} = \theta(q, t, \lambda)\f$.
/// @param parameters parameters of ODE integration
template <OdeSolver S, typename Ds, typename P>
auto periodic(Ds dynamic_system, P parameters) {
    return internal::periodic<S, Ds>(std::move(dynamic_system),
                                     std::move(parameters));
}
} // namespace nld

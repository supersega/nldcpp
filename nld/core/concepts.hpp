#pragma once

#include <concepts>

#include <nld/core/aliases.hpp>
#include <nld/core/utils.hpp>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace nld {
/// @brief The archetypes for concepts definition.
namespace archetypes {
/// @brief The scalar type used to define scalars.
struct scalar {
    template <typename U>
    scalar(U &&);

    template <typename U>
    scalar &operator=(U &&other);

    operator double() const;
    operator nld::dual() const;
    operator nld::dual2() const;
};

/// @brief The vector type used to define vectors.
struct vector {
    auto operator[](index) -> scalar;
    auto operator()(index) -> scalar;

    template <typename T>
    auto cast() const -> nld::vector_x<T>;

    template <typename T>
    operator nld::vector_x<T>() const;
};

/// @brief The ode type used to define ode.
struct ode {
    template <typename Args>
    auto operator()(const vector &, scalar, Args &&...) -> vector;
};
} // namespace archetypes

/// @brief Concept for ode.
template <typename S>
concept Scalar = requires(S s) { S(std::declval<double>()); };

static_assert(Scalar<archetypes::scalar>,
              "Please update scalar archetype to satisfy Scalar concept");

/// @brief Concept for ode.
/// TODO: make this more strict.
template <typename V>
concept Vector = requires(V v) {
    // Vector should have an operator[] with index
    requires Scalar<decltype(v[std::declval<index>()])>;

    // Vector should have an operator() with index
    requires Scalar<decltype(v(std::declval<index>()))>;

    // Vector should be convertible to vector_xd
    v.template cast<double>();
};

static_assert(Vector<archetypes::vector>,
              "Please update vector archetype to satisfy Vector concept");

/// @brief Concept for ode.
template <typename M>
concept DenseMatrix = std::is_base_of_v<Eigen::MatrixBase<M>, M>;

template <typename M>
concept SparseMatrix = std::is_base_of_v<Eigen::SparseMatrixBase<M>, M>;

template <typename M>
concept Matrix = DenseMatrix<M> || SparseMatrix<M>;

/// @brief OdeSolver concept.
/// @details OdeSolver should provide solution static function and end_solution
/// function. See arguments bellow to understand signature. solution - assumed
/// to return whole solution on interval stored in Matrix. end_solution -
/// assumed to return end solution on interval stored in Vector.
template <typename S>
concept OdeSolver = requires(S s) {
    typename S::integration_parameters_t;
    {
        S::solution(std::declval<archetypes::ode>(),
                    std::declval<typename S::integration_parameters_t>(),
                    std::declval<archetypes::vector>())
    } -> Matrix;
    {
        S::end_solution(std::declval<archetypes::ode>(),
                        std::declval<typename S::integration_parameters_t>(),
                        std::declval<archetypes::vector>())
    } -> Vector;
};

/// @brief Function with solution from previous step.
/// @details This concept will help to decide if previos
/// solution is required for continuation algorithm on
/// Newton steps. If this method is implemented in problem
/// Newton method step will set previous solution when called.
template <typename F>
concept FunctionWithPreviousStepSolution =
    requires(F f) { f.set_previous_solution(std::declval<utils::any_type>()); };

/// @brief Function which provides Jacobian method.
/// @details Concept of function supports Jacobian evaluation.
/// Jacobian computed in Neton method, if function provides
/// method bellow, it will be used for computations
/// otherwise jacobian from autodiff will be used.
template <typename F, typename Wrt, typename At, typename R>
concept Jacobian = requires(F f, Wrt wrt, At at, R &r) {
    { f.jacobian(wrt, at, r) } -> Matrix;
};

/// @brief Function which provides Norm method
template <typename F, typename At>
concept Norm = requires(F f, At at) {
    { f.norm(at) } -> Scalar;
};
}; // namespace nld

// namespace nld

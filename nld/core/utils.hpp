#pragma once

#include <nld/core/aliases.hpp>

namespace nld::utils {
/// @brief CRTP helper class.
/// @details CRTP used insted of dynamic polymorphism
/// because of author...
template <typename T>
struct crtp {
    /// @brief Get reference to derived.
    /// @return reference to derived.
    T &derived() { return static_cast<T &>(*this); }

    /// @brief Get constant reference to derived.
    /// @return constant reference to derived.
    const T &derived() const { return const_cast<crtp<T> &>(*this).derived(); }

protected:
    crtp() = default;
};

/// @brief Simple class to represent any type.
/// @details Useful when it does not metter what type of
/// input arguments for the function. Will help to deduce
/// return type.
struct any_type {
    /// @brief Cast this one to any type.
    template <typename T>
    operator T() const;
};

/// @brief Create 'view' on matrix type as on 'tensor'
/// @details This function is useful when contraction should
/// applied with 'tensor' and 'matrix' type. This one makes
/// 'view' on non owned memory.
/// @param matrix The matrix what will look like tensor.
/// @return tensor representation of memory owned by matrix.
template <typename Real, int Rows, int Cols>
auto tensor_view(const matrix_generic<Real, Rows, Cols> &matrix) {
    if constexpr (Cols == 1 || Rows == 1)
        return Eigen::TensorMap<Eigen::Tensor<const Real, 1>>(matrix.data(),
                                                              matrix.size());
    else
        return Eigen::TensorMap<Eigen::Tensor<const Real, 2>>(
            matrix.data(), matrix.rows(), matrix.cols());
}
} // namespace nld::utils

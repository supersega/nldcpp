#pragma once

#include <nld/calculus/adnum.hpp>

namespace nld {
/// @brief The space structure to represent space of variational problem.
/// @tparam Dim 
template<std::size_t Dim>
struct space final {
    static constexpr std::size_t dimension = Dim;

    template<typename... Basis>
    auto make_test_functions(Basis... basis) const {
        
    }

    /// @brief 
    /// @return std::array<autodiff::forward::HigherOrderDual<4>, Dim> 
    auto coords() const -> std::array<adnum, Dim>& {
        return space_variables;
    }
private:
    mutable std::array<adnum, Dim> space_variables = {}; ///< 
};
}
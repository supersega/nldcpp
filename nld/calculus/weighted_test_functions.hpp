#pragma once 

#include <nld/calculus/adnum.hpp>
#include <nld/calculus/test_functions.hpp>

namespace nld {
/// @brief Test functions multiplied by weight BC function.
template<typename T, typename B>
struct weighted_test_functions final {
    static constexpr nld::index dimension = T::dimension;

    /// @brief Create one dimension test functions.
    /// @param definition The basis definition (number of basis functions and interval) in dimension.
    /// @param boundary_condition Boundary condition for 'weak' formulation.
    template<typename Space, typename Basis, typename Bc> 
    explicit weighted_test_functions(test_functions<Space, Basis> tf, Bc bc) : 
        underlying_test_functions { tf }, boundary_condition { bc } 
    {

    }

    /// @brief The value compute function of i-th test function 
    /// @param i index of test function
    /// @return Function f(x) -> std::array<double, Dim> to compute i-th function at point x.
    auto value(nld::index i) const {
        return [basis = underlying_test_functions, bc = boundary_condition, i](auto x) -> adnum { 
            return basis.value(i)(x) * bc.value()(x); 
        };
    }

    /// @brief The value compute function of i-th test function 
    /// @param i index of test function
    /// @return Function f(x) -> std::array<double, Dim> to compute i-th function at point x.
    auto value(std::tuple<nld::index> i) const {
        return value(std::get<0>(i));
    }

    /// @brief Get the domain of i-th test function.
    /// @param i index of test function.
    /// @return I-th test function domain.
    auto subdomain(std::tuple<nld::index> i) const {
        return underlying_test_functions.subdomain(i);
    }

    /// @brief Count of test functions for approximation.
    /// @returns Number of test functions.
    auto count() const -> std::tuple<nld::index> {
        return underlying_test_functions.count();
    }
    
    /// @brief Get the space object.
    /// @return Space where test functions are defined.s
    auto get_space() const -> const auto& {
        return underlying_test_functions.get_space();
    }
private:
    T underlying_test_functions;
    B boundary_condition;
};

template<typename Space, typename Basis, typename Bc> 
weighted_test_functions(test_functions<Space, Basis> tf, Bc bc) -> weighted_test_functions<test_functions<Space, Basis>, Bc>;
}

#pragma once

#include <nld/core.hpp>

namespace nld {
/// @brief The eigenfunctions of differential operator.
template <typename T>
struct eigenfunctions final {
    static constexpr nld::index dimension = 1;

    /// @brief Construct a new eigenfunctions object.
    /// @param test_functions The test functions.
    /// @param eigenvectors The eigenvectors.
    explicit eigenfunctions(T test_functions, nld::matrix_xd &&eigenvectors,
                            nld::index size)
        : test_functions(test_functions), eigenvectors(std::move(eigenvectors)),
          size(size) {}

    /// @brief The value compute function of i-th test function
    /// @param i index of test function
    /// @return Function f(x) -> std::array<double, Dim> to compute i-th
    /// function at point x.
    auto value(nld::index i) const {
        return [this, i](auto x) -> adnum {
            auto eigenvector = this->eigenvectors.col(i);

            adnum acc = 0.0;
            for (nld::index j = 0; j < eigenvector.size(); j++)
                acc += eigenvector(j) * this->test_functions.value(j)(x);

            return acc;
        };
    }

    /// @brief The value compute function of i-th eigenfunction.
    /// @param i index of eigenfunction.
    /// @return Function f(x) -> adnum to compute i-th eigenfunction at point.
    auto value(std::tuple<nld::index> i) const { return value(std::get<0>(i)); }

    /// @brief Count of eigenfunctions for approximation.
    /// @returns Number of eigenfunctions.
    auto count() const -> std::tuple<nld::index> { return std::tuple(size); }

    /// @brief Get the space object.
    /// @return Space where test functions are defined.
    auto get_space() const -> const auto & {
        return test_functions.get_space();
    }

private:
    T test_functions;            ///< The test functions.
    nld::matrix_xd eigenvectors; ///< The eigenvectors.
    nld::index size;             ///< The count of eigenfunctions to analyze.
};

} // namespace nld

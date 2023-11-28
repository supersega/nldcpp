#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

namespace nld::collocations {

/// @brief Lagrange polynomial basis defined on a segmant
struct lagrange_basis final {
    /// @brief Create a lagrange basis
    /// @param interval Segment where the basis is defined
    /// @param degree Degree of the Laagrange polynomial
    explicit lagrange_basis(nld::segment interval, std::size_t degree)
        : interval(interval), degree{degree} {
        auto [begin, end] = interval;
        interpolation_nodes = nld::vector_xd::LinSpaced(degree + 1, begin, end);
    }

    /// @brief Evaluate the ith basis function at a point
    /// @tparam T Type of the point to be able to use dual numbers
    /// @param i Index of the basis function
    /// @param t Point where the basis function is evaluated
    /// @return Value of the ith basis function at t
    template <typename T>
    auto operator()(std::size_t i, T t) const -> T {
        auto h = interval.end - interval.begin;
        auto m = static_cast<std::size_t>(degree);
        auto tij = interpolation_nodes[i];

        T result = 1.0;
        for (std::size_t k = 0; k <= m; ++k) {
            if (k != i) {
                auto tkm = interpolation_nodes[k];
                result *= (t - tkm) / (tij - tkm);
            }
        }

        return result;
    }

    auto interpolate(const nld::vector_xd &values) const {
        auto [begin, end] = interval;
        auto h = end - begin;
        auto m = static_cast<std::size_t>(degree);

        auto f = [=](auto t) -> double {
            decltype(t) result = 0.0;
            for (std::size_t i = 0; i <= m; ++i) {
                result += values[i] * (*this)(i, t);
            }
            return static_cast<double>(result);
        };

        return f;
    }

private:
    nld::segment interval;              ///< Segment where the basis is defined
    std::size_t degree;                 ///< Degree of the Lagrange polynomial
    nld::vector_xd interpolation_nodes; ///< Interpolation nodes
};

} // namespace nld::collocations

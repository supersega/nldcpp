#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

namespace nld {
/// @brief One dimensional basis definition.
/// @details In each direction for solving calculus problem
/// basis should be defined. Basis is described by the number of
/// function in basis, and interval.
struct basis_definition final {
    nld::index count;                   ///< Number of elements in basis.
    std::pair<double, double> interval; ///< Interval where basis is defined.
};

/// @brief Basis function base class.
/// @details Inherit from this class to create the basis in function space.
/// @see https://en.wikipedia.org/wiki/Basis_function.
struct basis {
    /// @brief Construct a new basis object.
    /// @param def The basis definition.
    explicit basis(basis_definition def) : definition(def) { }

    /// @brief Number of basis functions.
    /// @return Number of basis functions.
    auto count() const -> nld::index {
        return definition.count;
    }

    /// @brief Interval where basis is defined.
    /// @return Interval where basis is defined.
    auto interval() const -> std::pair<double, double> {
        return definition.interval;
    }

    /// @brief Interval where basis is defined.
    /// @return Interval where basis is defined.
    auto domain() const -> nld::segment {
        return { definition.interval.first, definition.interval.second };
    }
protected:
    basis_definition definition; ///< Definition of this basis.
};

}

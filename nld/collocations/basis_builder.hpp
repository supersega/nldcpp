#pragma once

#include "nld/math/segment.hpp"
namespace nld::collocations {

/// @brief Make a basis builder for a given basis
/// @tparam B Basis
/// @return Basis builder function
template <typename B>
auto make_basis_builder() {
    return [](nld::segment interval, std::size_t degree) {
        return B{interval, degree};
    };
}
} // namespace nld::collocations

#pragma once 

#include <nld/core.hpp>

namespace nld {

/// @brief Dimension type.
/// @details We prefer to use types that can describe 
/// problem instead of 'raw' c++ types.
struct dimension final {
    /// @brief Dimension type.
    explicit dimension(index s) : size(s) { }

    /// @brief Cast dimension into index type
    operator index() const { return size; }
private:
    const index size; ///< size.
};

}

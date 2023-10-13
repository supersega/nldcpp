#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

namespace nld {
namespace internal {
auto join(const nld::tensor<2>& l, const nld::tensor<2>& r) {
    auto r_cols = r.dimension(1);
    auto r_rows = r.dimension(0);
    auto l_cols = l.dimension(1);
    auto l_rows = l.dimension(0);
    
    nld::tensor<4> result(l_rows, l_cols, r_rows, r_cols);

    for (nld::index r_col = 0; r_col < r_cols; r_col++)
        for (nld::index r_row = 0; r_row < r_rows; r_row++)
            for (nld::index l_col = 0; l_col < l_cols; l_col++)
                for (nld::index l_row = 0; l_row < l_rows; l_row++)
                    result(l_row, l_col, r_row, r_col) = l(l_row, l_col) * r(r_row, r_col);

    return result;
}
}
/// @brief Utility function to join two tensors
/// @tparam DimL 
/// @tparam DimR 
/// @param l 
/// @param r 
/// @return auto 
template<int DimL, int DimR>
auto join(const nld::tensor<DimL>& l, const nld::tensor<DimR>& r) {
    return nld::internal::join(l, r);
}
}
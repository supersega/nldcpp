#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nld/core.hpp>

namespace nld::math::linear_algebra {
// Solve linear system using QR decomposition for dense matrix.
inline auto solve(const nld::matrix_xd &A, const nld::vector_xd &b)
    -> nld::vector_xd {
    return A.fullPivHouseholderQr().solve(b);
}

// Solve linear system using QR decomposition for sparse matrix.
inline auto solve(nld::sparse_matrix_xd &A, const nld::vector_xd &b)
    -> nld::vector_xd {
    Eigen::SparseLU<nld::sparse_matrix_xd, Eigen::COLAMDOrdering<int>> solver;

    solver.analyzePattern(A);
    solver.factorize(A);

    return solver.solve(b);
}
} // namespace nld::math::linear_algebra

#pragma once
#include <nld/core/dissable_warnings.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

/* should be deleted - we have problem in autodiff*/
using namespace Eigen;

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace nld {

/// Internal function wrt.
template <typename... Args>
auto wrt(Args &...args) {
    return autodiff::wrt(args...);
}

using autodiff::at;

using index = Eigen::Index;

using dual = autodiff::dual;
using dual2 = autodiff::HigherOrderDual<2, double>;

template <typename Real, int Rows, int Cols>
using matrix_generic = Eigen::Matrix<Real, Rows, Cols>;

template <typename Real>
using vector_x = Eigen::Matrix<Real, -1, 1>;
using vector_xd = vector_x<double>;

using vector_xdd = vector_x<dual>;
using vector_xdd2 = vector_x<dual2>;

template <typename Real, int Size = -1>
using matrix_x = Eigen::Matrix<Real, Size, Size, 0, Size, Size>;
using matrix_xd = matrix_x<double>;

using matrix_xdd = matrix_x<dual>;

using matrix_xdcd = Eigen::Matrix<std::complex<dual>, -1, -1>;

using sparse_matrix_xd = Eigen::SparseMatrix<double>;

template <int Dim>
using tensor = Eigen::Tensor<double, Dim>;

using tensor_1d = Eigen::Tensor<double, 1>;
using tensor_2d = Eigen::Tensor<double, 2>;
using tensor_3d = Eigen::Tensor<double, 3>;

template <typename U>
struct is_nonlinear_function : std::false_type {};

template <typename U>
constexpr auto is_nonlinear_function_v = is_nonlinear_function<U>::value;
} // namespace nld

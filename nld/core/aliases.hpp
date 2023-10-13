#pragma once
#include <nld/core/dissable_warnings.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

/* should be deleted - we have problem in autodiff*/
using namespace Eigen;

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

namespace nld {

/// Internal function wrt.
template<typename... Args>
auto wrt(Args&... args)
{
    return autodiff::wrtpack(args...);
}

using autodiff::forward::at;

using index = Eigen::Index;

using dual = autodiff::forward::dual;
using dual2 = autodiff::forward::HigherOrderDual<2>;

template<typename Real, int Rows, int Cols>
using matrix_generic = Eigen::Matrix<Real, Rows, Cols>;

template<typename Real>
using vector_x = Eigen::Matrix<Real, -1, 1>;
using vector_xd = vector_x<double>;

using vector_xdd = vector_x<dual>;
using vector_xdd2 = vector_x<dual2>;

template<typename Real, int Size = -1>
using matrix_x = Eigen::Matrix<Real, Size, Size, 0, Size, Size>;
using matrix_xd = matrix_x<double>;

template<int Size = -1>
using matrix_xdd = Eigen::Matrix<dual, Size, Size, 0, Size, Size>;

template<int Dim>
using tensor = Eigen::Tensor<double, Dim>;

using tensor_1d = Eigen::Tensor<double, 1>;
using tensor_2d = Eigen::Tensor<double, 2>;
using tensor_3d = Eigen::Tensor<double, 3>;

template<typename U>
struct is_nonlinear_function : std::false_type { };

template<typename U>
constexpr auto is_nonlinear_function_v = is_nonlinear_function<U>::value;
}

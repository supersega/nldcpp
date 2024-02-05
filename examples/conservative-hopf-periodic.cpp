#include <iostream>
#include <nld/autocont.hpp>
#include <vector>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

/// @brief Conservative system with periodic solutions
/// @details This system has periodic solutions, in examples
/// of AUTO it described as "Model with vectical Hopf".
/// Since it has Hopf bifurcation, we can use it show
/// how to compute periodic solutions for this system.
auto conservative(const nld::vector_xdd &u, nld::dual lambda) {
    nld::vector_xdd f(2);

    f[0] = (lambda)*u[0] - u[1];
    f[1] = u[0] - u[0] * u[0];

    return f;
}

int main() {
    // Make continuation parameters from Newton parameters,
    // total arc length, minimal step size, maximal step size,
    // and direction of continuation.
    nld::continuation_parameters params(nld::newton_parameters(25, 0.000005),
                                        2.1, 0.003, 0.003,
                                        nld::direction::forward);

    // Construct constant parameters for single shooting problem,
    // from num and number ODE integration steps.
    auto ip = nld::periodic_parameters_constant{1, 200};

    // Crate boundary value problem from autonomous system,
    // with Runge-Kutta 4 integration method for autonomous
    // system, and constant parameters for single shooting problem.
    auto bvp = periodic<nld::runge_kutta_4>(nld::autonomous(conservative), ip);

    // Hopf bifurcation occurs at (0, 0, 0)
    // so we can compute Jacobian matrix at this point
    // and find eigenvalues of this matrix.
    nld::vector_xdd u0(2);
    u0 << 0.0, 0.0;
    nld::dual lambda0 = 0.0;
    auto jacobian_hopf =
        autodiff::forward::jacobian(conservative, wrt(u0), at(u0, lambda0));

    // Eigenvalues and vectors of Jacobian matrix at Hopf bifurcation point
    Eigen::EigenSolver<Eigen::MatrixXd> es(jacobian_hopf);
    std::cout << "The eigenvalues of A are:" << std::endl
              << es.eigenvalues() << std::endl;
    std::cout << "The matrix of eigenvectors, V, is:" << std::endl
              << es.eigenvectors() << std::endl
              << std::endl;

    // From eigenvevtors we can make initial guess for continuation
    // of periodic solution.
    Eigen::VectorXcd v = es.eigenvectors().col(0);
    Eigen::VectorXd nc = v.real();
    Eigen::VectorXd ns = v.imag();

    nld::dual T = 2.0 * M_PI;
    nld::dual lambda = 0.0;
    auto s = 0.002;

    // Create initial guess for continuation of periodic solution
    // from real part of eigenvector, period, and parameter.
    nld::vector_xdd us(4);
    us << s * nc, T, lambda;

    // Create initial tangent for continuation of periodic solution
    // from imaginary part of eigenvector, and zeros.
    nld::vector_xdd vs(4);
    vs << -nc, 0.0, 0.0;

    // Collect amplitudes and periods of periodic solutions
    std::vector<double> amplitudes;
    std::vector<double> periods;

    // Run pseudo-arclength continuation of periodic solutions
    // use concat function to make mapping of arc_length iteration step
    // to tuple<period, mean_amplitude>.
    for (auto [T, A] :
         arc_length(bvp, params, us, vs,
                    concat(nld::unknown(2), nld::mean_amplitude(0)))) {
        periods.push_back(T);
        amplitudes.push_back(A);
    }

    // Plot amplitude of periodic solution vs period of periodic solution
    plt::ylabel(R"($\frac{max(u(t)) - min(u(t))}{2}$)");
    plt::xlabel(R"(T)");
    plt::named_plot("Periodic from Hopf", periods, amplitudes, "-k");
    plt::show();
}

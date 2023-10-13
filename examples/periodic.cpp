// C++ includes
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

#include <nld/autocont.hpp>
using namespace nld;

vector_xdd NLTVA(const vector_xdd &y, dual t, dual Omega) {
    vector_xdd dy(4);

    auto epsilon = 0.05;
    auto m1 = 1.0;
    auto c1 = 0.002;
    auto k1 = 1.0;
    auto knl1 = 1.0;
    auto k2u = 8.0 * epsilon *
               (16.0 + 23.0 * epsilon + 9.0 * epsilon * epsilon +
                (4.0 + 2.0 * epsilon) * sqrt(4.0 + 3.0 * epsilon));
    auto k2l = 3.0 * (1.0 + epsilon) * (1.0 + epsilon) *
               (64.0 + 80.0 * epsilon + 27.0 * epsilon * epsilon);
    auto k2 = k2u / k2l;
    auto knl2 = 2.0 * epsilon * epsilon * knl1 / (1.0 + 4 * epsilon);
    auto caux = (k2 * m1 * epsilon *
                 (8.0 + 9.0 * epsilon - 4.0 * sqrt(4.0 + 3.0 * epsilon))) /
                4.0 / (1.0 + epsilon);
    auto c2 = sqrt(caux);

    auto f0 = 0.15;

    dy[0] = y[2];
    dy[1] = y[3];
    dy[2] = -c1 * y[2] - k1 * y[0] - knl1 * y[0] * y[0] * y[0] -
            c2 * (y[2] - y[3]) - k2 * (y[0] - y[1]) -
            knl2 * (y[0] - y[1]) * (y[0] - y[1]) * (y[0] - y[1]) +
            f0 * cos(t * 2.0 * PI);
    dy[3] = -c2 * (y[3] - y[2]) - k2 * (y[1] - y[0]) -
            knl2 * (y[1] - y[0]) * (y[1] - y[0]) * (y[1] - y[0]);

    dy[3] = dy[3] / epsilon;

    dy *= 2.0 * PI;
    dy /= Omega;

    return dy;
}

int main() {
    ofstream fs("afc_loop.csv");
    fs << 'x' << ';' << 'y' << endl;

    continuation_parameters params(newton_parameters(25, 0.00001), 26.5, 0.003,
                                   0.001, direction::forward);

    auto ip = periodic_parameters{1, 200};
    auto bvp = periodic<runge_kutta_4>(non_autonomous(NLTVA), ip);

    vector_xdd u0(5);
    u0 << 0.0, 0.0, 0.0, 0.0, 0.3;

    vector_xdd v0(5);
    v0 << 0.0, 0.0, 0.0, 0.0, 1.0;

    std::cout << "Start\n";
    vector_xdd u1(5);
    for (auto [solution, monodromy, amplitude] :
         arc_length(bvp, params, u0, v0,
                    concat(solution(), monodromy(), mean_amplitude(0)))) {
        auto frequency = solution(solution.size() - 1);

        Eigen::EigenSolver<Eigen::MatrixXd> es(monodromy);
        auto ev = es.eigenvalues();
        vector_xd r = ev.real().array().pow(2) + ev.imag().array().pow(2);
        nld::index i = 0;
        auto is_saddle_node = false;
        for (i = 0; i < ev.size(); i++) {
            is_saddle_node = (std::abs(ev(i).real() - 1.0) < 1.0e-3) &&
                             (std::abs(ev(i).imag()) < 1.0e-3);
            if (is_saddle_node)
                break;
        }

        if (is_saddle_node) {
            vector_xd e_vector = es.eigenvectors().real().col(i);
            vector_xd wwww = monodromy * e_vector - e_vector;
            cout << "Saddle node: eigen vectors are - \n"
                 << es.eigenvectors().col(i) << '\n';
            cout << "Saddle node: variables are - \n" << wwww << '\n';
        }

        cout << ((r.array() < 1.0).all() ? "stable" : "unstable") << endl;
        cout << frequency << endl;
        fs << frequency << ';' << amplitude << endl;
    }
    std::cout << "u1: \n" << u1;

    auto parameter = u1(4);

    auto bvp2 = periodic<runge_kutta_4>(non_autonomous(NLTVA), ip);
    auto wrpd = [&bvp2, u1, parameter](const auto &vs) {
        vector_xdd values(5);

        values.head(4) = vs.head(4);
        values(4) = parameter;
        vector_xdd aux = vs.head(4) - u1.head(4);
        dual M = (1.0 / aux.lpNorm<2>());

        vector_xdd res = M * bvp2(values);
        return res;
    };

    nld::vector_xdd known = nld::vector_xdd::Ones(5);
    known(4) = 0.0;
    known.normalize();
    // x0 << 1.28826 , -1.28716, 2.62268, -0.693648, 1.90048;

    newton_homotopy homotopy(wrpd, known);

    nld::vector_xdd unknown(5);
    unknown << nld::vector_xdd::Ones(4), parameter;

    nld::vector_xdd tan = nld::vector_xdd::Zero(5);
    tan(4) = 1.0;

    for (auto value : arc_length(homotopy, params, known, tan, solution())) {
        auto kappa = value(4);
        std::cout << "kappa:" << kappa << endl;
        // fs << kappa << ';' << value(1) << endl;
        unknown.head(4) = value.head(4);
        if (abs(kappa - 1.0) < 1.0e-3) {
            break;
        }
    }

    for (auto [p, v] : arc_length(bvp, params, unknown, v0, half_swing(0))) {
        std::cout << p << ';' << v << endl;
        fs << p << ';' << v << endl;
    }
}

// C++ includes
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include <nld/autocont.hpp>
using namespace nld;

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;
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

vector_xdd weakly_nonlinear_system(const vector_xdd &y, dual Omega) {
    vector_xdd dy(4);

    dy[0] = y[2];
    dy[1] = y[3];
    dy[2] = -1.0 * (2.0 * y[0] - y[1] + 0.5 * y[0] * y[0] * y[0]);
    dy[3] = -1.0 * (2.0 * y[1] - (1.0) * y[0]);

    return dy;
}

auto duffing_autonomous(const vector_xdd &y, dual omega) {
    vector_xdd dy(4);

    double c = 0.01;
    double k = 1.0;
    double alpha = 0.7;
    double A = 3.0;

    dy(0) = y(1);
    dy(1) = -k * y(0) - c * y(1) - alpha * y(0) * y(0) * y(0) + A * y(3);
    dy(2) = y(2) + omega * y(3) - y(2) * (y(2) * y(2) + y(3) * y(3));
    dy(3) = -omega * y(2) + y(3) - y(3) * (y(2) * y(2) + y(3) * y(3));

    return dy;
}

auto predator_prey(const vector_xdd &z, dual p1) {
    vector_xdd dz(2);

    auto u1 = z[0];
    auto u2 = z[1];
    auto p2 = 3.0;
    auto p3 = 5.0;
    auto p4 = 4.0;

    dz[0] = p2 * u1 * (1 - u1) - u1 * u2 - p1 * (1 - exp(-p3 * u1));
    dz[1] = -u2 + p4 * u1 * u2;

    return dz;
}

auto simple_system_3(const vector_xdd &u, dual t, dual T) {
    vector_xdd f(2);

    f[0] = 0 - u[1];
    f[1] = u[0] - u[0] * u[0];

    f *= T;

    return f;
}

int main() {
    ofstream fs("afc_loop.csv");
    fs << 'x' << ';' << 'y' << endl;

    continuation_parameters params(newton_parameters(25, 0.00001), 2.1, 0.003,
                                   0.001, direction::forward);

    auto ip = periodic_parameters_constant{1, 200};
    auto bvp = periodic<runge_kutta_4>(non_autonomous(simple_system_3), ip);

    vector_xdd u0(3);
    u0 << 0.0966822, 0.0, 6.30641;

    vector_xdd v0(3);
    v0 << 0.0, 0.0, 1.0;

    std::vector<double> Am;
    std::vector<double> L;
    std::cout << "Start\n";
    for (auto [solution, monodromy, amplitude] :
         arc_length(bvp, params, u0, v0,
                    concat(solution(), monodromy(), mean_amplitude(0)))) {
        auto period = solution(solution.size() - 1);
        std::cout << solution << std::endl;
        std::cout << "Frq: " << period << " Amplitude: " << amplitude
                  << std::endl;

        Am.push_back(double(amplitude));
        L.push_back(double(period));
    };

    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($T$)");
    plt::xlim(6.25, 8.0);
    plt::named_plot("Saddle Node", L, Am, "-b");
    plt::show();
}

// C++ includes
inline constexpr auto PI = 3.14159265358979323846264338327950288;
#include <iostream>
using namespace std;

#include <nld/autocont.hpp>
using namespace nld;

// Duffing oscillator with energy dissipation
vector_xdd duffing(const vector_xdd &y, dual t, dual Omega) {
    vector_xdd dy(y.size());

    dual t5 = pow(y[0], 0.2e1);
    dual t8 = cos(t);

    dy[0] = y[1] / Omega;
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * t5 * y[0] - 0.8600261454e-3 * t8;

    return dy;
}

int main() {
    vector_xdd x = vector_xdd::Zero(2);
    dual omega = 0.05;

    auto ip = periodic_parameters_constant{1, 200};

    auto bvp = periodic<runge_kutta_4>(non_autonomous(duffing), ip);

    // if (newton(bvp, wrt(x, omega), at(x, omega),
    //            newton_parameters(10, 0.00005)))
    //     cout << "Great work x = " << x << '\n';
}

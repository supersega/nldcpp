// C++ includes
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <iostream>

#include <nld/math.hpp>
using namespace nld;

vector_xdd ode_p(const vector_xdd& y, dual t, dual Omega)
{
    vector_xdd dy(y.size());

    dual t5 = pow(y[0], 0.2e1);
    dual t8 = cos(t);

    dy[0] = y[1] / Omega;
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] - 0.1499999998e2 * t5 * y[0] - 0.8600261454e-3 * t8;

    return dy;
}

auto ode(const vector_xdd& y, dual t)
{
    return ode_p(y, t, 1.3);
}

int main() {
    vector_xdd x = vector_xdd::Zero(2);

    auto sln = runge_kutta_4::solution(ode, constant_step_parameters{ 0.0, 2.0 * PI, 200 }, x);

    std::cout << "Solution is : " << sln << '\n';
}

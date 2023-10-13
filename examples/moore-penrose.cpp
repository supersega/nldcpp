// C++ includes
#include <iostream>
#include <fstream>
using namespace std;

#include <nld/autocont.hpp>
using namespace nld;

// Algebraic system from matcont example
vector_xdd f(const vector_xdd& u, dual alpha)
{
    vector_xdd f(u.size());

    f[0] = - 2 * u[0] + u[1] + alpha * exp(u[0]);
    f[1] = u[0] - 2 * u[1] + alpha * exp(u[1]);

    return f;
}

int main() {
    // continuation_parameters params(
    //    newton_parameters(10, 0.000001), 1.45, 0.0002, 0.008, direction::forward);

    // vector_xdd u0 = vector_xdd::Zero(2);
    // dual alpha0 = 0.0;

    // for (auto p: nld::continuation::moore_penrose(f, params, u0, alpha0, nld::continuation::to_2d<0>)) {
    //     std::cout << "param: " << std::get<1>(p) << "value: " << std::get<0>(p) << std::endl;
    // }
}

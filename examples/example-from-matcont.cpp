// C++ includes
#include <iostream>
using namespace std;

#include <nld/autocont.hpp>
using namespace nld;

// Equilibrium problem for ode system right parts of ode
nld::vector_xdd f(const nld::vector_xdd &u, nld::dual alpha) {
    nld::vector_xdd f(2);

    f[0] = -2.0 * u[0] + u[1] + alpha * exp(u[0]);
    f[1] = u[0] - 2.0 * u[1] + alpha * exp(u[1]);

    return f;
}

int main() {
    // continuation parameters
    nld::continuation_parameters cp(nld::newton_parameters(25, 0.00009), 8.5,
                                    0.0002, 0.0025, nld::direction::forward);

    // Initial solution for continuation (f0, f1, alpha)
    nld::vector_xdd u0 = nld::vector_xdd::Zero(3);
    // Initial tangent for continuation (d(f0)/ds, df(f1)/ds, d(alpha)/ds)
    nld::vector_xdd v0(3);
    v0 << 0, 0, 1;

    for (auto [p, v] :
         arc_length(equilibrium(f), dimension(2), cp, nld::point2d(2, 1))) {
        cout << "alpha: " << p << " ; "
             << "f1: " << v << endl;
    }
}

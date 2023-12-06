// C++ stl includes
#include <iostream>

// nld includes
#include <nld/math.hpp>

nld::vector_xdd f(nld::dual x, nld::dual a) {
    nld::vector_xdd y(1);
    y << x * x - a;
    return y;
}

int main() {
    nld::dual x = 1.9;
    if (newton(f, nld::wrt(x), nld::at(x, 4.0),
               nld::newton_parameters(10, 1.0e-5)))
        std::cout << "Root value is : " << x << '\n';
    else
        std::cout << "Newton method does not converged \n";
}

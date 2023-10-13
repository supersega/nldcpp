// C++ stl includes
#include <iostream>

// nld includes
#include <nld/math.hpp>
using namespace nld;

vector_xdd f(dual x, dual a) {
    vector_xdd y(1);
    y << x * x - a;
    return y;
}

int main() {
    dual x = 1.9;
    if (newton(f, wrt(x), at(x, 4.0), newton_parameters(10, 1.0e-5)))
        std::cout << "Root value is : " << x << '\n';
    else
        std::cout << "Newton method does not converged \n";
}

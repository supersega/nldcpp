# nldcpp

Nonlinear dynamics library

This library is suposed to solve nonlinear dynamics continuation problems.

## Example

In example below saddle node bifurcation continuation is shown

```cpp
int main() {
    continuation_parameters params(
        newton_parameters(25, 0.00001), 26.5, 0.003, 0.001, direction::forward);

    auto ip = periodic_parameters{ 1, 200 };
    auto snb = saddle_node<runge_kutta_4>(non_autonomous(NLTVA), ip);

    vector_xdd2 u0(10);
    u0 << 0.0372658, -2.25952, 0.781638, 2.70618, 0.158379, 0.601359, 0.0393662, 0.782134, 1.11126, 0.15;

    vector_xdd2 v0(10);
    v0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;

    ofstream fs("afc_loop.csv");
    fs << 'x' << ';' << 'y' << endl;

    for (auto [s, A0] : arc_length(snb, params, u0, v0, concat(solution(), mean_amplitude(0)))) {
        auto A = s(8);
        std::cout << "Solution = \n" << A << endl;
        fs << A << ';' << A0 << endl;
    }
}
``` 

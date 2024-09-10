from nld import (
    newton_parameters,
    Direction,
    continuation_parameters,
    RungeKutta4,
    NonAutonomousRnPlusOneToRnMapDual,
    periodic_parameters,
    non_autonomous,
    periodic,
    make_duffing,
    solution,
    mean_amplitude,
    monodromy,
    concat,
    generate_sequence,
    arc_length,
    print_vector,
    # unknown,
)

from autodiff import VectorXdual, VectorXdual0th, dual

import numpy as np
import matplotlib.pyplot as plt

npr = newton_parameters(max_iterations=100, tolerance=2e-5)
cp = continuation_parameters(
    newton_parameters=npr,
    total_param_length=3.1,
    param_min_step=0.01,
    param_max_step=0.01,
    direction=Direction.Forward,
)
# print(test_w(1.0, 2.0))
duff = make_duffing()
# Wrong overload chosen by python
na = non_autonomous(duff)
na2 = NonAutonomousRnPlusOneToRnMapDual(duff)
pa = periodic_parameters(periods=1, intervals=100)
pe = periodic(dynamic_system=na2, parameters=pa, solver=RungeKutta4())

sol = solution()
ma = mean_amplitude(0)
mo = monodromy()

us = VectorXdual(3)
us[0] = dual(0.0)
us[1] = dual(0.0)
us[2] = dual(0.37)

# print(us)
us2 = VectorXdual0th(3)
us2[0] = 0.01
us2[1] = 0.01
us2[2] = 0.01
print(us)
print_vector(us2)

vs = VectorXdual(3)
vs[0] = 0.0
vs[1] = 0.0
vs[2] = 1.0

mapper = concat(sol, ma, mo)

print("Generating sequence")
print(repr(duff))
print(repr(na))
print(repr(na2))

Omega = []
Amplitude = []
for sln, A0, M in arc_length(pe, us, mapper):
    Omega.append(sln[2])
    Amplitude.append(A0)
    print("Omega: ", sln[2])
    print("Amplitude: ", A0)

plt.plot(Omega, Amplitude)
plt.show()

print(repr(npr))
print(repr(cp))
print(repr(pe))
print(repr(sol))
print(repr(ma))
print(repr(mapper))

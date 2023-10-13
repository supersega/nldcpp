import matplotlib.pyplot as plt
import numpy as np

main = np.genfromtxt("beam.csv", delimiter=";", names=["x", "y"])
#loop = np.genfromtxt("afc_loop.csv", delimiter=";", names=["x", "y"])

plt.plot(main['x'], main['y'])
#plt.plot(loop['x'], loop['y'])

plt.show()


# def pend(t, y):
#     y0, y1 = y
#     dydt = [t * 2.0 * np.pi * y1 / 0.01, t * 2.0 * np.pi * (- 0.01 * y1 - 16.0 * y0 + 0.22 * np.sin(t * 2.0 * np.pi)) / 0.01]
#     return dydt

# y0 = [0.1, 0.1]

# t = np.linspace(0, 10.0, 300)

# from scipy.integrate import solve_ivp
# sol = solve_ivp(pend, [0.0, 10.0], y0, dense_output=True)

# z = sol.sol(t)

# plt.plot(t, z.T, 'b', label='theta(t)')
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

curve = np.loadtxt("curve.txt", delimiter = ' ')
omega = curve[:, 0]
amplitude = curve[:, 1]
stability = curve[:, 2]
period_doubling = np.loadtxt("pd.txt", delimiter = ' ')

amplitude_stable = np.ma.masked_where(stability == 0, amplitude)
amplitude_unstable = np.ma.masked_where(stability == 1, amplitude)

plt.plot(omega, amplitude_stable, color = 'black')
plt.plot(omega, amplitude_unstable, linestyle = '--', color = 'black')
#plt.scatter(period_doubling[:, 0], period_doubling[:, 1], color = 'tab:red', alpha = 1.0, label = "Period doubling")
plt.legend()
plt.ylabel(r"$A_2$");
plt.xlabel(r"$\Omega$");
plt.show()

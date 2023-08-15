import seaborn as sns
import random
import numpy as np
from matplotlib import pyplot as plt

peg_radius = 2.1
hole_radius = 2.4

r_low = 0.6
r_high = 0.8
R_hole = hole_radius
R_peg = peg_radius
theta = np.linspace(0, 2*np.pi, 1000)
x_hole = R_hole*np.cos(theta)
y_hole = R_hole*np.sin(theta)
x_peg = R_peg*np.cos(theta)
y_peg = R_peg
x_low = r_low*np.cos(theta)
y_low = r_low*np.sin(theta)
x_high = r_high*np.cos(theta)
y_high = r_high*np.sin(theta)

plt.figure()
plt.plot(x_hole, y_hole, color='k', label='hole',linewidth=3)
plt.plot(x_low, y_low, color='r', linestyle='dashed')
plt.plot(x_high, y_high, color='r', linestyle='--')
for i in range(1000):
    r = random.uniform(r_low, r_high)
    theta = random.uniform(0, 2 * np.pi)
    x_error = r * np.cos(theta)
    y_error = r * np.sin(theta)
    # #
    plt.scatter(x_error, y_error,c='blue', label=f'x:{np.round(x_error,2)}|y:{np.round(y_error,2)}')
plt.grid()
plt.show()
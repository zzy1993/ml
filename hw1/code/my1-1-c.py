import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10.0, 10.0, 1000)

y = 2 * np.exp(np.abs(x - 1) / 2 - np.abs(x))

plt.plot(x, y)
plt.title('Likelihood Ratio (1-1-c)')
plt.xlabel('x')
plt.ylabel('P(x|w1)/P(x|w2)')
plt.grid('on')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10.0, 10.0, 1000)

y1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp( -1.0 * np.power(x, 2) / 2.0)
y2 = (1.0 / (2.0 * np.sqrt(np.pi))) * np.exp( -1.0 * np.power(x - 1, 2) / 4.0)

plt.plot(x, y1, label="P(x|w1)")
plt.plot(x, y2, label="P(x|w2)")
plt.title('Class conditional PDF (1-2-b)')
plt.xlabel('x')
plt.ylabel('P(x|wi)')
plt.grid('on')
plt.legend()
plt.show()

z1 = y1 / (y1 + y2)
z2 = y2 / (y1 + y2)

plt.plot(x, z1, label="P(w1|x)")
plt.plot(x, z2, label="P(w2|x)")
plt.title('P(wi|x) (1-2-b)')
plt.xlabel('x')
plt.ylabel('P(wi|x)')
plt.grid('on')
plt.legend()
plt.show()
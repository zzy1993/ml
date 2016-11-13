import numpy as np
import matplotlib.pyplot as plt
import discriminant
import gaussian

mu = [[0.0,0.0], [3.0,3.0]]
sigma = [[[3.0, 1.0], [1.0, 0.8]], [[3.0, 1.0], [1.0, 0.8]]]
nSamples = 400
prior=[0.5, 0.5];

data, classIndex = gaussian.gaussian(mu, sigma, nSamples, prior);
option = 1
label, AP = discriminant.discriminant(data, classIndex, mu, sigma, nSamples, prior, option)

x1, y1, x2, y2 = [], [], [], []
for i in range(len(classIndex)):
	if label[i] == 0:
		x1.append(data[i][0])
		y1.append(data[i][1])
	elif label[i] == 1:
		x2.append(data[i][0])
		y2.append(data[i][1])
plt.plot(x1, y1, 'r.', label='x1')
plt.plot(x2, y2, 'b.', label='x2')

x = np.linspace(-10.0, 10.0, 1000)
y = np.linspace(-10.0, 10.0, 1000)
X, Y = np.meshgrid(x,y)
F = X+Y-3
plt.contour(X,Y,F,[0])
plt.grid('on')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('Plot (1-4-b-1), accuracy = ' + str(AP))
plt.show()

data, classIndex = gaussian.gaussian(mu, sigma, nSamples, prior);
option = 2
label, AP = discriminant.discriminant(data, classIndex, mu, sigma, nSamples, prior, option)

x1, y1, x2, y2 = [], [], [], []
for i in range(len(classIndex)):
	if label[i] == 0:
		x1.append(data[i][0])
		y1.append(data[i][1])
	elif label[i] == 1:
		x2.append(data[i][0])
		y2.append(data[i][1])
plt.plot(x1, y1, 'r.', label='x1')
plt.plot(x2, y2, 'b.', label='x2')

x = np.linspace(-10.0, 10.0, 1000)
y = np.linspace(-10.0, 10.0, 1000)
X, Y = np.meshgrid(x,y)
F = 0.4286*X-4.2857*Y+5.7857
plt.contour(X,Y,F,[0])
plt.grid('on')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.title('Plot (1-4-b-2), accuracy = ' + str(AP))
plt.show()
import numpy as np
import numpy.matlib
from scipy.stats import multivariate_normal
import sys

def discriminant(data, classIndex, mu, sigma, nSamples, prior, option):

	k = len(mu)
	if not (len(sigma) == k and len(prior) == k):
		return None

	d = len(mu[0])
	for i in range(k):
		if not (len(mu[i]) == d and len(sigma[i]) == d):
			return None
		for j in range(d):
			if not len(sigma[i][j]):
				return None

	M = np.matrix(mu) 
	D = np.matrix(data)
	P = np.zeros((nSamples, k))

	label = [0 for i in range(nSamples)]

	if option == 1:
		sigmaAvg = 0.0

		for i in range(k):
			tempSum = 0.0
			for j in range(d):
				tempSum += sigma[i][j][j]
			sigmaAvg += tempSum / d
		sigmaAvg = sigmaAvg / k

		for i in range(nSamples):
			for j in range(k):
				P[i][j] =  M[j] * D[i].T / sigmaAvg - 0.5 * M[j] * M[j].T / sigmaAvg + np.log(prior[j])
			maxValue = - sys.maxint
			for j in range(k):
				if maxValue < P[i][j]:
					maxValue = P[i][j]
					label[i] = j

	elif option == 2:
		S = np.matlib.zeros((d, d))
		for i in range(k):
			S += np.matrix(sigma[i])
		S = S / k

		for i in range(nSamples):
			for j in range(k):
				P[i][j] = M[j] * S.I * D[i].T - 0.5 * M[j] * S.I * M[j].T + np.log(prior[j])
			maxValue = - sys.maxint
			for j in range(k):
				if maxValue < P[i][j]:
					maxValue = P[i][j]
					label[i] = j

	elif option == 3:
		for i in range(nSamples):
			for j in range(k):
				S = np.matrix(sigma[j])
				P[i][j] = -0.5 * (D[i] - M[j]) * S.I * (D[i] - M[j]).T - 0.5 * np.log(np.linalg.det(S)) + np.log(prior[j])
			maxValue = - sys.maxint
			for j in range(k):
				if maxValue < P[i][j]:
					maxValue = P[i][j]
					label[i] = j

	else:
		for i in range(nSamples):
			for j in range(k):
				S = np.matrix(sigma[j])
				P[i][j] = multivariate_normal(D[i], M[j], S) * prior[j]
			maxValue = - sys.maxint
			for j in range(k):
				if maxValue < P[i][j]:
					maxValue = P[i][j]
					label[i] = j

	summ = 0
	for i in range(len(classIndex)):
		if label[i] == classIndex[i]:
			summ += 1
	AP = float(summ) / float(nSamples)
	return label, AP
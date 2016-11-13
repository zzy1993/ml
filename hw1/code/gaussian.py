import numpy as np

def gaussian(mu, sigma, nSamples, prior):

	k = len(mu)
	if not (len(prior) == k and len(sigma) == k):
		return None

	d = len(mu[0])
	for i in range(k):
		if not (len(mu[i]) == d and len(sigma[i]) == d):
			return None
		for j in range(k):
			if not (len(sigma[i][j]) == d):
				return None

	classIndex = []
	data = []
	for i in range(nSamples):
		temp = np.random.rand() - prior[0]
		cls = 0
		while temp > 0.0:
			cls += 1
			temp -= prior[cls]
		ls = list(np.random.multivariate_normal(mu[cls],sigma[cls]))
		data.append(ls)
		classIndex.append(cls)

	return data, classIndex

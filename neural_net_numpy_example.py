import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
X = np.array([[1.51617, 14.95, 0.0, 2.27, 73.3, 0.0, 8.71, 0.67, 0.0], 
    [1.51732, 14.95, 0.0, 1.8, 72.99, 0.0, 8.61, 1.55, 0.0], 
    [1.51645, 14.94, 0.0, 1.87, 73.11, 0.0, 8.67, 1.38, 0.0], 
    [1.51831, 14.39, 0.0, 1.82, 72.86, 1.41, 6.47, 2.88, 0.0], 
    [1.5164, 14.37, 0.0, 2.74, 72.85, 0.0, 9.45, 0.54, 0.0], 
    [1.51623, 14.14, 0.0, 2.88, 72.61, 0.08, 9.18, 1.06, 0.0], 
    [1.51685, 14.92, 0.0, 1.99, 73.06, 0.0, 8.4, 1.59, 0.0], 
    [1.52065, 14.36, 0.0, 2.02, 73.42, 0.0, 8.44, 1.64, 0.0], 
    [1.51651, 14.38, 0.0, 1.94, 73.61, 0.0, 8.48, 1.57, 0.0], 
    [1.51711, 14.23, 0.0, 2.08, 73.36, 0.0, 8.62, 1.67, 0.0]])

# output dataset
y = np.array([[0, 0, 1, 1, 0, 0, 1, 0, 1, 0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((9, 1)) - 1

for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)

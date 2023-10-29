import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from examples.classification.tools import (
    sigmoid,
    relu,
    mean_square_error,
    initialize_parameter_deep
)
from examples.classification.nn import (
    Train,
)

N = 1000
gaussian_quantities = make_gaussian_quantiles(
    mean=None,
    cov=0.1,
    n_samples=N,
    n_features=2,
    n_classes=2,
    shuffle=True,
    random_state=None
)

X, Y = gaussian_quantities
Y = Y[:, np.newaxis]


# Activation function
plt.grid()
plt.scatter(X[:,0], X[:,1], c=Y[:,0], s=20, cmap=plt.cm.Spectral)
plt.plot()
plt.show()

test = np.linspace(10,-10,100)
plt.plot(test, sigmoid(test), label='sigmoid')
plt.plot(test, relu(test), label='relu')
plt.legend()
plt.show()

# Set the neurons 
layer_dim = [2, 4, 8, 1]
params = initialize_parameter_deep(layer_dim)

# start error list
errors = []

for _ in range(50_000):
    output = Train(params=params, x_data=X, y=Y, lr=0.001)

    if _ % 25 == 0:
        error = mean_square_error(Y, output)
        print(error)
        errors.append(error)

plt.plot(errors)
plt.show()

data_test = np.random.rand(1000, 2) * 2 - 1
y = Train(params=params, x_data=data_test, lr=0.001, y=Y, training=False)
y = np.where(y > 0.5, 1, 0)


plt.scatter(data_test[:,0], data_test[:,1], c=y[:,0], s=20, cmap=plt.cm.Spectral)
plt.show()
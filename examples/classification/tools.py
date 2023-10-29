import numpy as np

def sigmoid(x, derivate=False):

    if derivate:
        return np.exp(-x) / (1 + np.exp(-x)) ** 2
    else:
        return  1 / (1 + np.exp(-x))

def relu(x, derivate = False):
    if derivate:
        x[x<=0] = 0
        x[x>0]  = 1
        return x
    else:
        return np.maximum(0, x)
    

def mean_square_error(y,y_hat,derivate=False):
    if derivate:
        return (y_hat - y)
    else:            
        return np.mean((y_hat - y)**2)
    
def initialize_parameter_deep(layers_dim):

    parameters = {}
    LENGTH = len(layers_dim)

    for l in range(0, LENGTH - 1):
        # Se le asignas números aleatorios a los pesos, note que se debe multiplicar por el intervalo [2]
        parameters['W' + str(l + 1)] = (np.random.rand(layers_dim[l], layers_dim[l+1]) * 2) - 1
        parameters['b' + str(l + 1)] = (np.random.rand(1, layers_dim[l+1]) * 2) - 1 

    return parameters



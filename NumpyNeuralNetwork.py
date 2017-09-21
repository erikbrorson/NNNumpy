import numpy as np
import matplotlib as plt
import pandas as pd

# global parameters 
n_iterations = 20000
learning_rate = 0.1

def initialize_parameters(n_x, n_y):
    # initializes parameters:
    #     w1 = [n_x, 5]
    #     b1 = [5, 1]
    #     w2 = [n_y, 5]
    #     b2 = [n_y, 1]
    # Returns a dictionary, params
    
    # Initialize the parameters
    w1 = np.random.randn(5, n_x) * 0.01
    b1 = np.zeros((5, 1))
    w2 = np.random.randn(n_y, 5) * 0.01
    b2 = np.zeros((n_y, 1))
    
    # now create the return dict params
    params = {'w1':w1,
                        'b1':b1,
                        'w2':w2,
                        'b2':b2}
    
    return(params)
def linear_forward(A, w, b):

    Z = np.dot(w, A) + b
    cache = (A, w, b)
    return(Z, cache)

def forward_prop(A, w, b, activation):
    # implements forward propagation 
    # returns z, cache containing A, w, b
    Z, linCache = linear_forward(A, w, b)

    if activation == 'relu':
        A, actCache = relu(Z)

    if activation == 'softmax':
        A, actCache = softmax(Z)

    cache = (linCache, actCache)

    return(A, cache)

def softmax(Z):
    exp = np.exp(Z)
    A     = exp / np.sum(exp, axis=0)
    return(A, Z)

def relu(Z):
    A = np.maximum(0, Z)
    return(A, Z)

def calculate_cost(A, Y):
    # implements the cross entropy cost
    m = Y.shape[0]
    cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m
    cost = np.squeeze(cost)

    return(cost)

def backward_prop_linear(dZ, lin_cache):
    A_prev, w, b = lin_cache
    m = A_prev.shape[1]

    dW            = np.dot(dZ, A_prev.T) / m
    db            = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(w.T, dZ) / m

    return(dA_prev, dW, db)

def backward_prop_relu(dA, actCache):
    Z = actCache
    dZ = dA * (Z > 0)
    return(dZ)


def backward_prop(dA, cache):
    linear_cache, activation_cache = cache

    dZ = backward_prop_relu(dA, activation_cache)
    dA_prev, dW, db = backward_prop_linear(dZ, linear_cache)

    return(dA_prev, dW, db) 

def get_grads(dA, cache):
    cache1, cache2 = cache
    dA1, dw2, db2 = backward_prop(dA, cache2)
    _, dw1, db1 = backward_prop(dA1, cache1)

    grads = {'dw2':dw2,
                     'db2':db2,
                     'db1':db1,
                     'dw1':dw1}
    return(grads)

def update_parameters(params, grads, learning_rate):
    params['w1'] = params['w1'] - learning_rate * grads['dw1']
    params['b1'] = params['b1'] - learning_rate * grads['db1']
    params['w2'] = params['w2'] - learning_rate * grads['dw2']
    params['b2'] = params['b2'] - learning_rate * grads['db2']

    return(params)


def model(Y, X, learning_rate, n_iterations):

    # Initialize parameters
    params = initialize_parameters(n_x = X.shape[0], n_y = Y.shape[0])

    for epoch in range(n_iterations):
        # move forwards in the net...
        A1, cache1 = forward_prop(X, params['w1'], params['b1'], 'relu')
        A2, cache2 = forward_prop(A1, params['w2'], params['b2'], 'softmax')
        caches = (cache1, cache2)
        # calculate cost
        cost = calculate_cost(A = A2, Y = Y)
        if epoch % 500 == 0:
            print('epoch number ' + str(epoch) + ' and the cost is ' + str(cost))
        
        # calculate gradient
        dA2 = A2 - Y
        grads = get_grads(dA = dA2, cache = caches)
        # update parameters
        params = update_parameters(params, grads, learning_rate)
    return(params)

def main():
    iris = pd.read_csv('iris.csv')

    y = pd.get_dummies(iris[['species']]).as_matrix().T
    x = iris.drop('species', 1).as_matrix().T

    np.random.seed(111)
    model(Y = y, X = x, learning_rate = learning_rate, n_iterations = n_iterations)

if __name__ == "__main__":
        main()



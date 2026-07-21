'''
Consider the following two networks.  W is a matrix, x is a vector, z is a vector, and a is a vector.
 y^ is a scalar and a final prediction. Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).
   Initialize X and W randomly. Every neuron has a bias term.

'''

import numpy as np

x= [[0.3],[0.2]]

w1=[[[0.1],[-0.1],[0.2]], # W11 , w21 , w31
    [[-1.1],[0.4],[1.1]]]  # W12,W22,W32

w2=[[[0.2],[0.3],[0.1]], # W11 , w21 , w31
    [[-0.1],[-0.2],[-0.2]]] #  W12,W22,W32

w3=[[[0.3]],    #W11
    [[0-0.3]]]  #W12

def Relu_activatin_fun(z):
    fun = []
    for i in range(len(x)):
        if x[i] > 0:
            fun.append(x[i])
        else:
            fun.append(0)
    return fun

def layer_1 (x,w1):

    a1=[]
    x_shape=x.shape
    w1_shape=w1.shape
    if x_shape[0]==w1_shape[0]:
        z1= np.multiply(x,w1)
    
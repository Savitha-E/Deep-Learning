
'''
Consider the following two networks.  W is a matrix, x is a vector, z is a vector, and a is a vector.
 y^ is a scalar and a final prediction. Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).
   Initialize X and W randomly. Every neuron has a bias term.

'''

import numpy as np


x=np.matrix([[0.3],[-1.2]]) #x2

w1=np.matrix([[0.1 , -1.1],#w11,w12
              [-0.1,0.4], # w21,w22
              [0.2,1.1]]) #w31,w32

bias_layer1=np.matrix ([[1],[1],[1]])

w2=np.matrix([ [0.2 , 0.1 , -0.2] ,
               [0.3,-0.1, -0.1]])

print(np.shape(w2))
print ("This is w2 shape ",np.shape(w2))
bias_layer2=np.matrix([[1],[1]])

w3=np.matrix([[0.3],
              [-0.3]])

bias_layer3=np.matrix([[1],
               [1]])


def Relu_activation_function(z):
 np.maximum(z,0)
 return z


def layer_1 (x,w1,bias_layer1,Relu_activation_function):
    m= np.dot(w1,x)
    z=m+bias_layer1
    activation_layer=Relu_activation_function(z)
    return activation_layer

activation_layer1=layer_1(x,w1,bias_layer1,Relu_activation_function)
print("This is activation layer 1 ",activation_layer1)
print(np.shape(activation_layer1))

def layer_2 (w2,bias_layer2,Relu_activation_function):
    m= np.dot(w2,activation_layer1)
    z=m+bias_layer2
    activation_layer=Relu_activation_function(z)
    return activation_layer

activation_layer2=layer_2(w2,bias_layer2,Relu_activation_function)
print("This is activation layer 2 ",activation_layer2)


#Back propogation


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

bias_layer2=np.matrix([[1],[1]])

w3=np.matrix([[0.3,-0.3]])

bias_layer3=np.matrix([[1]])

def Relu_activation_function(z):
    return np.maximum(z, 0)


def layer_1 (x,w1,bias_layer1,Relu_activation_function):
    m= np.dot(w1,x)
    z=m+bias_layer1
    activation_layer=Relu_activation_function(z)
    return activation_layer

layer1=layer_1(x,w1,bias_layer1,Relu_activation_function)



def layer_2 (w2,bias_layer2,Relu_activation_function, layer1):
    m= np.dot(w2,layer1)
    z=m+bias_layer2
    activation_layer=Relu_activation_function(z)
    return activation_layer

layer2=layer_2(w2,bias_layer2,Relu_activation_function,layer1)




def layer_w3 ( w3,bias_layer3,Relu_activation_function,layer2):
    m= np.dot(w3,layer2)
    z=m+bias_layer3
    activation_layer=Relu_activation_function(z)
    return activation_layer
layer3=layer_w3(w3,bias_layer3,Relu_activation_function,layer2)


def main():
    layer1 = layer_1(x, w1, bias_layer1, Relu_activation_function)

    layer2 = layer_2(
        w2,
        bias_layer2,
        Relu_activation_function,
        layer1
    )

    layer3 = layer_w3(
        w3,
        bias_layer3,
        Relu_activation_function,
        layer2
    )

    print("Activation layer 1:")
    print(layer1)

    print("Activation layer 2:")
    print(layer2)

    print("Predicted output:")
    print(layer3)

    return layer1, layer2, layer3

if __name__ == "__main__":
    layer1, layer2, layer3 = main()

### Back propogation for these models:


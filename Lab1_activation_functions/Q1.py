'''
Implement the following functions in Python from scratch. Do not use any library functions. You are allowed to use
numpy and matplotlib. Generate 100 equally spaced values between -10 and 10. Call this list as  z.
Implement the following functions and its derivative. Use class notes to find the expression for these functions.
Use z as input and plot both the function outputs and its derivative outputs.


Sigmoid
Tanh
ReLU (Rectified Linear Unit)
Leaky ReLU
Softmax (no need for visualization)

'''
import numpy as np
import matplotlib.pyplot as plt

z=np.linspace(-10,10,100)
alpha = 0.01

def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    sigmoid=1/(1+np.exp(-x))
    return sigmoid*(1-sigmoid)

def tanh(z):
    x= ((np.exp(z)-np.exp(-z))/((np.exp(z)+np.exp(-z))))
    return x

def tanh_derivative(x):
    tanh_derivative= 1 - ((np.exp(z)-np.exp(-z))/((np.exp(z)+np.exp(-z))))
    return tanh_derivative

def relu(x):
    #return np.maximum(0,x)
    fun = []
    for i in range(len(x)):
        if x[i] > 0:
            fun.append(x[i])
        else:
            fun.append(0)
    return fun

def relu_derivative(x):
    fun=[]
    for i in range (len(x)):
        if x[i] >0 :
          fun.append(x[i])
        elif x[i]==0:
          fun.append(np.nan)
        else:
          fun.append(0)

    return fun

print(relu_derivative(z))


def leaky_relu(x,alpha):
    fun = []
    for i in range(len(x)):
        if x[i] > 0:
            fun.append(x[i])

        else:
            fun.append(alpha*x[i])
    return fun

def leaky_relu_derivative(x,alpha):
    fun = []
    for i in range(len(x)):
        if x[i] > 0:
            fun.append(x[i])
        elif x[i] == 0:
            fun.append(np.nan)
        else:
            fun.append(alpha)
    return fun


def softmax(z):
    exp_fun = []
    fun=[]
    for i in range(len(z)):
        exp_fun.append(np.exp(z[i]))
    sum_fun = np.sum(exp_fun)
    for j in range(len(exp_fun)):
        y= (exp_fun[j])/ sum_fun
        fun.append(y)
    return fun


def main(z,alpha):

    sigmoid_function_output=sigmoid(z)
    sigmoid_derivative_function_output=sigmoid_derivative(z)
    tanh_function_output=tanh(z)
    tanh_derivative_function_output=tanh_derivative(z)
    relu_function_output=relu(z)
    relu_derivative_function_output=relu_derivative(z)
    leaky_relu_function_output=leaky_relu(z,alpha)
    leaky_relu_derivative_function_output=leaky_relu_derivative(z,alpha)
    softmax_function_output=softmax(z)



    print("This is the sigmoid function output", sigmoid_function_output)
    print("")
    print("This is the sigmoid derivative function output", sigmoid_derivative_function_output)
    print("")
    print("This is the tanh function output", tanh_function_output)
    print("")
    print("This is the tanh derivative function output", tanh_derivative_function_output)
    print("")
    print("This is the relu function output", relu_function_output)
    print("")
    print("This is the relu derivative function output", relu_derivative_function_output)
    print("")
    print("This is the leaky relu function output", leaky_relu_function_output)
    print("")
    print("This is the leaky relu derivative function output", leaky_relu_derivative_function_output)
    print("")
    print("This is the softmax function output", softmax_function_output)
if __name__ == '__main__':
    main(z,alpha)

'''
Write down the observations from the plot for all the above functions in the code.
What are the min and max values for the functions?
Sigmoid min=0
Sigmoid max=1

Tanh min=-1
Tanh max=1

ReLU min= 0
ReLU max=infinity

Leaky ReLu min= - infinity
Leaky ReLumax= infinity

Is the output of the function zero-centred?

sigmoid-no
tanH-yes
ReLu-no
Leaky Relu-no

What happens to the gradient when the input values are too small or too big?

sigmoid-
tanH-
ReLu-
Leaky Relu-

What is the relationship between sigmoid and tanh?

'''

#Plots


import numpy as np
import matplotlib.pyplot as plt

z=np.linspace(-10,10,100)
alpha = 0.01

def sigmoid(x):
    return 1/(1+np.exp(-x))

sigmoid_fun=sigmoid(z)

def plot_sigmoid_fun(z,sigmoid_fun):
    x=z
    y=sigmoid_fun
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The sigmoid function')
    plt.title('Sigmoid function')
    plot=plt.show()
    return plot

plot_sigmoid_fun=plot_sigmoid_fun(z,sigmoid_fun)
print(plot_sigmoid_fun)

def sigmoid_derivative(x):
    sigmoid=1/(1+np.exp(-x))
    return sigmoid*(1-sigmoid)

sigmoid_fun_derivative=sigmoid_derivative(z)

def plot_sigmoid_derivative(z,sigmoid_fun_derivative):
    x=z
    y=sigmoid_fun_derivative
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The sigmoid derivative function')
    plt.title('Sigmoid derivative function')
    plot=plt.show()
    return plot

plot_sigmoid_derivative=plot_sigmoid_derivative(z,sigmoid_fun_derivative)

def tanh(z):
    x= ((np.exp(z)-np.exp(-z))/((np.exp(z)+np.exp(-z))))
    return x

tanh_fun=tanh(z)


def plot_tanh_fun(z,tanh_fun):
    x=z
    y=tanh_fun
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The tanh function')
    plt.title('Tanh function')
    plot=plt.show()
    return plot
plot_tanh_fun=plot_tanh_fun(z,tanh_fun)
print(plot_tanh_fun)

def tanh_derivative(x):
    tanh_derivative= 1 - ((np.exp(z)-np.exp(-z))/((np.exp(z)+np.exp(-z))))
    return tanh_derivative

tanh_fun_derivative=tanh_derivative(z)


def plot_tanh_derivative(z,tanh_fun_derivative):
    x=z
    y=tanh_fun_derivative
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The tanh derivative function')
    plt.title('Tanh derivative function')
    plot=plt.show()
    return plot
plot_tanh_derivative=plot_tanh_derivative(z,tanh_fun_derivative)
print(plot_tanh_derivative)


def relu(x):
    #return np.maximum(0,x)
    fun = []
    for i in range(len(x)):
        if x[i] > 0:
            fun.append(x[i])
        else:
            fun.append(0)
    return fun

relu_fun=relu(z)

def plot_relu_fun(z,relu_fun):
    x=z
    y=relu_fun
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The relu function')
    plt.title('ReLU function')
    plot=plt.show()
    return plot
plot_relu_fun=plot_relu_fun(z,relu_fun)
print(plot_relu_fun)



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

relu_fun_derivative=relu_derivative(z)

def plot_relu_derivative(z,relu_fun_derivative):
    x=z
    y=relu_fun_derivative
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The relu derivative function')
    plt.title('ReLU derivative function')
    plot=plt.show()
    return plot
plot_relu_derivative=plot_relu_derivative(z,relu_fun_derivative)
print(plot_relu_derivative)


def leaky_relu(x,alpha):
    fun = []
    for i in range(len(x)):
        if x[i] > 0:
            fun.append(x[i])

        else:
            fun.append(alpha*x[i])
    return fun

leaky_fun=leaky_relu(z,alpha)

def plot_leaky_relu_fun(z,leaky_fun):
    x=z
    y=leaky_fun
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The leaky relu function')
    plt.title('Leaky relu function')
    plot=plt.show()
    return plot

plot_leaky_relu_fun=plot_leaky_relu_fun(z,leaky_fun)
print(plot_leaky_relu_fun)

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

leaky_fun_derivative=leaky_relu_derivative(z,alpha)

def plot_leaky_relu_derivative(z,leaky_fun_derivative):
    x=z
    y=leaky_fun_derivative
    plt.plot(x,y)
    plt.xlabel('The z values')
    plt.ylabel('The leaky relu derivative function')
    plt.title('Leaky relu derivative function')
    plot=plt.show()
    return plot
plot_leaky_relu_derivative=plot_leaky_relu_derivative(z,leaky_fun_derivative)
print(plot_leaky_relu_derivative)

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
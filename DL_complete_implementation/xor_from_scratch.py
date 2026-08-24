import numpy as np

# xor truth table
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
y = [0, 1, 1, 0]

# assign biases
b1 = 0.01
b2 = 0.01
b3 = 0.01

# assign random weights
w = np.random.rand(6).reshape(6, 1)


# loss function
def loss_fn(y_hat, y):
    return 0.5 * ((y_hat - y) ** 2)


# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def forward_pass(x1, x2, w, b1, b2, b3):
    h1z = x1 * w[0, 0] + x2 * w[2, 0] + b1
    h1a = relu(h1z)

    h2z = x1 * w[1, 0] + x2 * w[3, 0] + b2
    h2a = relu(h2z)

    yz = h1a * w[4, 0] + h2a * w[5, 0] + b3
    y = relu(yz)

    return y


def

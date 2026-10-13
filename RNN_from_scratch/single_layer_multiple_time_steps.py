import numpy as np

X = np.array([
    [1, 2, 3],
    [2, 3, 4]
])

W_xh = np.array([
    [0.5, -0.3],
    [0.8,  0.2],
    [0.1,  0.4]
])

W_hh = np.array([
    [0.1,  0.4, 0.0],
    [-0.2, 0.3, 0.2],
    [0.05, 0.01, 0.2]
])

W_hy = np.array([
    [1,   -1,   0.5],
    [0.5,  0.5, -0.5]
])

H0 = np.array([
    [0],
    [0],
    [0]
])

h = H0

for t in range(X.shape[1]):

    x = X[:, t].reshape(-1, 1)

    h = np.tanh(
        W_xh @ x +
        W_hh @ h
    )

    y = W_hy @ h

    print("t =", t + 1)
    print("x:")
    print(x)

    print("h:")
    print(h)

    print("y:")
    print(y)

    print("----------------")

    
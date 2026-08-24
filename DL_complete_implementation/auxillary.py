import sympy as sp

# Define a symbolic variable
x = sp.Symbol('x')

# Define a function
f = 1 / (1 + sp.exp(-x))

# Compute the gradient (derivative)
grad_f = sp.diff(f, x)

# Print the result
print("The derivative of sin(x) is:", grad_f)

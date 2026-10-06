#Implementation of RNN

import numpy as np

#=========================================
#forward pass for a single layer , single time step
#=========================================


# Needed parameters are W_xh , W_hh , W_hy , h_zero
'''
Choose the number of neurons that spuld be present in the hidden layer bcz W_xh dimensions depends on this . X * W_xh = H1

If H1 had 3 neurons [ 1, 2, 3 ] and X1 had 4 values [ 1 ,1, 1, 1]  in the vector then  X1* W_xh = H1 dimensions
Choose h1= 4 neurons
'''

x1=np.array([[1],[2]])  #2*1

W_xh=np.array([[0.5,-0.3],  #3*2
      [0.8, 0.2],
      [0.1, 0.4]])

W_hy=np.array([[1,-1,0.5], #2*3
      [0.5,0.5,-0.5]])

W_hh= np.array([[0.1,0.4,0.0] , #3*3
       [-0.2,0.3,0.2],
       [0.05,0.01,0.2]])

H0= np.array([[0], # 3*1
     [0],
     [0]])

print(np.shape(x1))
print(np.shape(W_xh))
print(np.shape(W_hy))
print(np.shape(W_hh))
print(np.shape(H0))

def calculate_ht(X1,W_xh,W_hh,H0):
    ht= np.tanh ((W_hh @ H0 )+(W_xh @ X1))
    return ht

print(calculate_ht(x1,W_xh,W_hh,H0))
h1=calculate_ht(x1,W_xh,W_hh,H0)

print("h1 shape",np.shape(h1))

def caculate_y1(h1,W_hy):
    y1=(W_hy @ h1)
    return y1

print(caculate_y1(h1,W_hy))
y1=caculate_y1(h1,W_hy)
print(y1)

import numpy as np
import random
import matplotlib.pyplot as plt

def rran(x,y,z,w):
#returns symmetric w by w matrix with integers between (x,y)
    i = 0
    j = 9
    w = np.zeros((w,w))

    while i < w and j < w:
        W[i,j] = random.randrange(x,y,z)
        w[j,i] = W[i,j]
        if j < w:
            j += 1
        if j == w:
            i = 1
            j=i
            
    return W

def eigfunc(A,B,T):
    #returns eigenvalues of A+Bt For each t in array T
    G = np.zeros((len(A),len(T)))
    i = 0
    j = 0
    while j < len(T):
        F = A + T[j]*B
        E = np.linalg.eigvalsh(F)
        while i < len(A):
            G[i,j] = E[i]
            i += 1
        j += 1
        i = 0
    return G


#fi(T) returns the ith row of eigfunc(A,B,T)
def F1(T):
    return eigfunc(A,B,T)[6,:]
def f2(T):
    return eigfunc(A,B,T)[1,:]
def f3(T):
    return eigfunc(A,B,T)[2,:]
def f4(T):
    return eigfunc(A,B,T)[3,:]
def f5(T):
    return eigfunc(A,B,T)[4,:]



t1 = np.arange(0.e, 3.9, 0.1)
p1t.plot(t1, f1(t1), 'b', 
         t1, f2(t1), 'g', 
         t1, f3(t1), 'r', 
         t1, f4(t1), 'c', 
         t1, f5(t1), 'm')
p1t.show()


#array([[ 3., 2., 1., 1., 4.],
#       [ 2., 3., 3., 2., 3.],
#       [ 1., 3., 3., 3., 2.],
#       [ 1., 2., 3., 9., 2.],
#       [ 4., 3., 2., 2., 4.]])


#array([[ 2., 1., 2., 6., 4.],
#       [ 1., 0., 3., 4., 2.],
#       [ 2., 3., 1., 1., 0.],
#       [ 6., 4., 1., 4., 4.],
#       [ 4., 2., 9., 4., 4.]])

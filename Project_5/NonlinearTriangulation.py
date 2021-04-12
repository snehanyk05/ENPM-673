import numpy as np
from scipy.optimize import leastsq
from Functions import linear_triangulation
import cv2
import matplotlib.pyplot as plt
#from IPython import get_ipython
#

#convert from homogenous to normal
def fromHomogenous(X):
    print('Homo',X)
    x = X[:-1, :]
    x /= X[-1, :]
    return x

#convert from normal to homogenous
def toHomogenous(x):
    X = np.vstack([x, np.ones((1, x.shape[1]))])
    return X

#perform non-linear triangulation
def nonlinearTriangulation(pose1, pose2, pt1, pt2):
    cmatrix = [[964.828979,0, 643.788025],[0, 964.828979, 484.40799],[0,0,1]]
    _,X,_ = linear_triangulation(pose1, pose2, pt1, pt2)
    print('XX',X)
    X = np.transpose(X)
    mat, success = leastsq(triangulationError, X, args=(pose1, pose2, pt1, pt2), maxfev=10000)

    #non-linear triangulated matrix
    mat = np.matrix(mat)
    mat = np.transpose(mat)
    return mat

#compute the error while doing triangulation
def triangulationError(x, Rt1, Rt2, x1, x2):
    cmatrix = [[964.828979,0, 643.788025],[0, 964.828979, 484.40799],[0,0,1]]
    X = np.matrix(x).T
    print(Rt1,R)
    print('X',X)
    px1 = fromHomogenous(np.matmul(cmatrix, np.matmul(Rt1, toHomogenous(fromHomogenous(X)))))
    px2 = fromHomogenous(np.matmul(cmatrix, np.matmul(Rt2, toHomogenous(fromHomogenous(X)))))
    diff1 = px1 - x1[:2]
    diff2 = px2 - x2[:2]

    return np.asarray(np.vstack([diff1, diff2]).T)[0, :]

import numpy as np

#perform linear triangulation
def linearTriangulation(rt1, rt2, pt1, pt2):
    cmatrix = [[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]]
    rt = np.vstack([np.matmul(cmatrix, rt1), np.matmul(cmatrix, rt2)])
    pt = np.vstack([pt1, pt2])
    print(pt1)
    return np.linalg.lstsq(rt, pt, rcond=None)[0] 

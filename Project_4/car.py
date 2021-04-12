# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:29:23 2019

@author: Sneha
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:19:08 2019

@author: Sneha
"""

import numpy as np
import cv2
import random
longestDistance = 100


def new_coordinate(original, frame, x, y, size, Ix, Iy):
    if (((y + size) > len(original)) or ((x + size) > len(original[0]))): return np.matrix([[-1], [-1]])
    T = np.matrix([[original[i, j] for j in range(x, x + size)] for i in range(y, y + size)])
    x1 = np.matrix([[q for q in range(size)] for z in range(size)])
    y1 = np.matrix([[z] * size for z in range(size)])

    Ix = np.matrix([[Ix[i, j] for j in range(x, x + size)] for i in range(y, y + size)])
    Iy = np.matrix([[Iy[i, j] for j in range(x, x + size)] for i in range(y, y + size)])

    P = [np.multiply(x1, Ix), np.multiply(x1, Iy), np.multiply(y1, Ix),np.multiply(y1, Iy), Ix, Iy]

    hessian_orig = [[np.sum(np.multiply(P[a], P[b])) for a in range(6)] for b in range(6)]
    hessian_inv = np.linalg.pinv(hessian_orig)

    p1, p2, p3, p4, p5, p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    k = 0
    bad_itr = 0
    min_cost = -1
    minW = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    W = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while (k <= 10):
        pos = [[W.dot(np.matrix([[x + i], [y + j], [1]], dtype='float')) for i in range(size)] for j in range(size)]
        if not (0 <= (pos[0][0])[0, 0] < cols and 0 <= (pos[0][0])[1, 0] < rows and 0 <= pos[size - 1][0][
            0, 0] < cols and 0 <= pos[size - 1][0][1, 0] < rows and 0 <= pos[0][size - 1][0, 0] < cols and 0 <=
            pos[0][size - 1][1, 0] < rows and 0 <= pos[size - 1][size - 1][0, 0] < cols and 0 <=
            pos[size - 1][size - 1][1, 0] < rows):
            return np.matrix([[-1], [-1]])

        I = np.matrix([[frame[int((pos[i][j])[1, 0]), int((pos[i][j])[0, 0])] for j in range(size)] for i in range(size)])

        error = np.absolute(np.matrix(I, dtype='int') - np.matrix(T, dtype='int'))

        steepest_error = np.matrix([[np.sum(np.multiply(g, error))] for g in P])
        mean_cost = np.sum(np.absolute(steepest_error))
        deltap = hessian_inv.dot(steepest_error)
        dp = warping(deltap)
        p1, p2, p3, p4, p5, p6 = p1 + dp[0, 0] + p1 * dp[0, 0] + p3 * dp[1, 0], p2 + dp[1, 0] + dp[0, 0] * p2 + p4 * dp[1, 0], p3 + dp[2, 0] + p1 * dp[2, 0] + p3 * dp[3, 0], p4 + dp[3, 0] + p2 * dp[2, 0] + p4 * dp[3, 0], p5 + \
                                 dp[4, 0] + p1 * dp[4, 0] + p3 * dp[5, 0], p6 + dp[5, 0] + p2 * dp[4, 0] + p4 * dp[5, 0]
        W = np.matrix([[1+p1,p3,p5], [p2,1+p4,p6]])

        if (min_cost == -1):
            min_cost = mean_cost
        elif (min_cost >= mean_cost):
            min_cost = mean_cost
            bad_itr = 0
            minW = W
        else:
            bad_itr += 1
        if (bad_itr == 3):
            W = minW
            return W.dot(np.matrix([[x], [y], [1.0]]))

        if (np.sum(np.absolute(deltap)) < 0.0006):
            return W.dot(np.matrix([[x], [y], [1.0]]))
            
def warping(p):
    p_output = np.matrix([[0.1]] * 6)
    val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
    p_output[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    p_output[1, 0] = (-p[1, 0]) / val
    p_output[2, 0] = (-p[2, 0]) / val
    p_output[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    p_output[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
    p_output[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val
    return p_output

cap = cv2.VideoCapture("data/car_vid.mp4")
im=cv2.imread('data/car/'+"frame0020.jpg")
height , width , layers =  im.shape
video = cv2.VideoWriter('Car.mp4',-1,10,(width,height))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

rows, cols = len(old_gray), len(old_gray[0])



#feature_point =[[259, 315]]
#w=283-259
#h=255-295
#Car
feature_point = [  [184, 188]]
w=318-159
h=237-113


feature_points = feature_point
mask = np.zeros_like(old_frame)

while (len(feature_point) > 0):
    ret, frame = cap.read()
    if(frame is None):
        break
    else:
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        Ix = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=5)
        Iy = cv2.Sobel(old_gray, cv2.CV_32F, 0, 1, ksize=5)
        good_new = [new_coordinate(old_gray, frame_gray, int(x), int(y), 15, Ix, Iy) for x, y in feature_point]
        newfeature_point = []
        # draw the tracks
        for i in range(len(feature_point)):
            a, b = feature_point[i]
            c, d = int((good_new[i])[0]), int((good_new[i])[1])
            if (0 <= c < cols and 0 <= d < rows):
                

                tl=(int(a-(w/2)+20),int(b+h/2))
                br=(int(a+w-15),int(b-h/1.5))
                frame = cv2.rectangle(frame,tl,br, (0, 255, 0), 2)
#                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                newfeature_point.append((c,d))
        img = cv2.add(frame,mask)
        video.write(img)
    
        old_gray = frame_gray.copy()
        feature_point = newfeature_point[:]


video.release()
cv2.destroyAllWindows()
cap.release()

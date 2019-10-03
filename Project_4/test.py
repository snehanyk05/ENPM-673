# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:19:08 2019

@author: Sneha
"""

import numpy as np
import cv2
import random
longestDistance = 100
def warpInv(p):
    inverse_output = np.matrix([[0.1]] * 6)
    val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
    inverse_output[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    inverse_output[1, 0] = (-p[1, 0]) / val
    inverse_output[2, 0] = (-p[2, 0]) / val
    inverse_output[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
    inverse_output[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
    inverse_output[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val
    return inverse_output

def anchorOutliers(init_features,features, maxIterations = 100):
        global longestDistance
        random.shuffle(features)

        anchorFeature = None
        outliersToAnchor = features
        inliersToAnchor = features
        i=-1
        for f in features:
            i+=1
			# Limit the amount of iterations for huge ROI's
            maxIterations-=1
            if maxIterations < 0:
                break

            outliersList = []
            inliersList = []

            if anchorFeature is None:
                anchorFeature = f
            j=-1
            for g in features:
                j+=1
                if f == g:
                    continue
                print(np.linalg.norm(np.subtract(np.subtract(g, f), distance[str(init_features[i])][str(init_features[j])])))
#				 If a feature is far away from its position relative to other features -> outlier, else inlier.
                if np.linalg.norm(np.subtract(np.subtract(g, f), distance[str(init_features[i])][str(init_features[j])])) > longestDistance:
                    outliersList.append(g)
                    features.pop(j)
                else:
                    inliersList.append(g)
                print(outliersList)
                print(init_features[i],init_features[j])
			# Keep the best feature found (also its inliers and outliers)
            if len(outliersToAnchor) > len(outliersList) :
                anchorFeature = f
                outliersToAnchor = outliersList
                inliersToAnchor = inliersList
#            print(anchorFeature, outliersToAnchor, inliersToAnchor)

        return anchorFeature, outliersToAnchor, inliersToAnchor
def get_New_Coordinate(Original, frame, x, y, size, gradOriginalX, gradOriginalY):
    if (((y + size) > len(Original)) or ((x + size) > len(Original[0]))): return np.matrix([[-1], [-1]])
    T = np.matrix([[Original[i, j] for j in range(x, x + size)] for i in range(y, y + size)])
    x1 = np.matrix([[q for q in range(size)] for z in range(size)])
    y1 = np.matrix([[z] * size for z in range(size)])

    gradOriginalX = np.matrix([[gradOriginalX[i, j] for j in range(x, x + size)] for i in range(y, y + size)])
    gradOriginalY = np.matrix([[gradOriginalY[i, j] for j in range(x, x + size)] for i in range(y, y + size)])

    gradOriginalP = [np.multiply(x1, gradOriginalX), np.multiply(x1, gradOriginalY), np.multiply(y1, gradOriginalX),np.multiply(y1, gradOriginalY), gradOriginalX, gradOriginalY]

    HessianOriginal = [[np.sum(np.multiply(gradOriginalP[a], gradOriginalP[b])) for a in range(6)] for b in range(6)]
    Hessianinv = np.linalg.pinv(HessianOriginal)

    p1, p2, p3, p4, p5, p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    k = 0
    bad_itr = 0
    min_cost = -1
    minW = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    W = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while (k <= 10):
        position = [[W.dot(np.matrix([[x + i], [y + j], [1]], dtype='float')) for i in range(size)] for j in range(size)]
        if not (0 <= (position[0][0])[0, 0] < cols and 0 <= (position[0][0])[1, 0] < rows and 0 <= position[size - 1][0][
            0, 0] < cols and 0 <= position[size - 1][0][1, 0] < rows and 0 <= position[0][size - 1][0, 0] < cols and 0 <=
            position[0][size - 1][1, 0] < rows and 0 <= position[size - 1][size - 1][0, 0] < cols and 0 <=
            position[size - 1][size - 1][1, 0] < rows):
            return np.matrix([[-1], [-1]])

        I = np.matrix([[frame[int((position[i][j])[1, 0]), int((position[i][j])[0, 0])] for j in range(size)] for i in range(size)])

        error = np.absolute(np.matrix(I, dtype='int') - np.matrix(T, dtype='int'))

        steepest_error = np.matrix([[np.sum(np.multiply(g, error))] for g in gradOriginalP])
        mean_cost = np.sum(np.absolute(steepest_error))
        deltap = Hessianinv.dot(steepest_error)
        dp = warpInv(deltap)
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


cap = cv2.VideoCapture("vase_vid.mp4")
im=cv2.imread('data/vase/'+"0019.jpg")
height , width , layers =  im.shape
video = cv2.VideoWriter('Vase3.mp4',-1,10,(width,height))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

rows, cols = len(old_gray), len(old_gray[0])
#p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#feature_point = [p.ravel() for p in p0]
#feature_point = feature_point[:1]
#[(124, 104), (335, 272)]
#Vase
#(124, 95), (172, 150)
feature_point =[[151,140]]
#[[139,107],[130,101],[133,140],[151,140],[176,125],[165,124]]
w=174-134
h=150-93
xs=np.arange(124,169,10)
ys=np.arange(98,145,10)

#feature_point =[[259, 315]]
#
#xs=np.arange(259,283,20)
#ys=np.arange(295,355,20)
##feature_point = [[124, 104], [335, 272]]
##feature_points = [[159, 113], [318, 237]]
##xs=np.arange(124,335,25)
##ys=np.arange(104,272,25)
#distance = {}
#for i in xs:
#    for j in ys:
#        feature_point.append([i,j])
##Human


#feature_point =[[259, 315]]
#w=283-259
#h=255-295
#xs=np.arange(259,283,20)
#ys=np.arange(295,355,20)


###feature_point = [[124, 104], [335, 272]]
###feature_points = [[159, 113], [318, 237]]
###xs=np.arange(124,335,25)
###ys=np.arange(104,272,25)
##distance = {}
#for i in xs:
#    for j in ys:
#        feature_point.append([i,j])

#        newFeature=[i,j]
#        for f in feature_point:
#            if (str(f) in distance):
#                distance[str(f)][str(newFeature)] = tuple(np.subtract(f,newFeature))
#            else:
#                distance[str(f)] = {str(newFeature) : tuple(np.subtract(f,newFeature))}
#
#
#            if (str(newFeature) in distance):
#                distance[str(newFeature)][str(f)] = tuple(np.subtract(newFeature,f))
#            else:
#                distance[str(newFeature)] = {str(f) : tuple(np.subtract(newFeature,f))}
#feature_points=feature_point
#Car
#feature_point = [[159, 113], [318, 237]]
#feature_points = [[159, 113], [318, 237]]
#xs=np.arange(159,318,25)
#ys=np.arange(113,237,25)
##feature_point = [[124, 104], [335, 272]]
##feature_points = [[159, 113], [318, 237]]
##xs=np.arange(124,335,25)
##ys=np.arange(104,272,25)
#for i in xs:
#    for j in ys:
#        feature_point.append([i,j])
#print(feature_points)
        
mask = np.zeros_like(old_frame)

while (len(feature_point) > 0):
    ret, frame = cap.read()
    if(frame is None):
        break
    else:
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        gradOriginalX = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=5)
        gradOriginalY = cv2.Sobel(old_gray, cv2.CV_32F, 0, 1, ksize=5)
        good_new = [get_New_Coordinate(old_gray, frame_gray, int(x), int(y), 15, gradOriginalX, gradOriginalY) for x, y in feature_point]
        newfeature_point = []
        # draw the tracks
        for i in range(len(feature_point)):
            a, b = feature_point[i]
            c, d = int((good_new[i])[0]), int((good_new[i])[1])
            if (0 <= c < cols and 0 <= d < rows):
                
#                mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
                tl=(int(a-w/2),int(b-h))
                br=(int(a+w/1.5),int(b+h/4))
                frame = cv2.rectangle(frame,tl,br, (0, 255, 0), 2)
#                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                newfeature_point.append((c,d))
        img = cv2.add(frame,mask)
        video.write(img)
    
        old_gray = frame_gray.copy()
        feature_point = newfeature_point[:]
#        s=anchorOutliers(feature_points,feature_point)

video.release()
cv2.destroyAllWindows()
cap.release()

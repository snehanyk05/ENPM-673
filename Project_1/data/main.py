# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:17:29 2019

@author: Sneha
"""

import numpy as np
import cv2


#for d in range(0:300):
img = cv2.imread('frame2.jpg')
gray = cv2.imread('frame2.jpg',0)
gray = cv2.medianBlur(gray,5)
ret,thresh = cv2.threshold(gray,240,255,1)

z,contours,h= cv2.findContours(thresh,1,2)

#cnt0 = contours[0]
#areas=[]
#for c in contours:
#    areas.append(cv2.contourArea(c))
points = []
for cnt in contours:
       approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
       points.append(approx)
#       if len(approx)==4:
#        cv2.drawContours(img,[cnt],0,(0,0,255))

corner=[]
for i in points[1]:
    x,y = i.ravel()
    corner.append((x,y))
#
for j in range(len(corner)):
    x = corner[j][0]
    y = corner[j][1]
    cv2.circle(img,(x,y),3,225,-1)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imwrite('corner.jpg',img)


img1 = cv2.imread('corner.jpg')
img = cv2.imread('ref_marker.png')
X_ref = np.array([[1,1],[200,1],[200,200],[1,200]])

X_img = [corner[1],corner[0],corner[3],corner[2]]
n = 4
A = []
for i in range(0,n):
    x = [X_img[i][0],X_img[i][1],1,0,0,0,-X_ref[i][0]*X_img[i][0],-X_ref[i][0]*X_img[i][1], -X_ref[i][0]]
    y = [0,0,0,X_img[i][0],X_img[i][1],1,-X_ref[i][1]*X_img[i][0],-X_ref[i][1]*X_img[i][1], -X_ref[i][1]]
    A.append(x)
    A.append(y)

u,d,v = np.linalg.svd(A)
h = v[-1,:]/(v[-1,-1])
H = np.reshape(h,(3,3))
im = cv2.warpPerspective(img1,H,(img.shape[1],img.shape[0]))
cv2.imwrite('map.jpg',im)
cv2.imshow('d',im)
cv2.waitKey(0)



im_gray = cv2.imread('map.jpg', cv2.IMREAD_GRAYSCALE)
X_ref = [corner[1],corner[0],corner[3],corner[2]]
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)
lp = np.array([[1,1],[512,1],[512,512],[1,512]])
X = cv2.resize(im_bw,(8,8))
if X[6,5] ==255:
    X_ref = [X_ref[0],X_ref[1],X_ref[2],X_ref[3]]
if X[6,2]==255:
    X_ref = [X_ref[1],X_ref[2],X_ref[3],X_ref[0]]
if X[2,6]==255:
    X_ref = [X_ref[3],X_ref[0],X_ref[1],X_ref[2]]
if X[2,2] ==255:
    X_ref = [X_ref[2],X_ref[3],X_ref[0],X_ref[1]]

img = cv2.imread('corner.jpg')
lena = cv2.imread('Lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
#X_ref = [corner[1],corner[0],corner[3],corner[2]]
X_img = lp
n = 4
A = []
for i in range(0,n):
    x = [X_img[i][0],X_img[i][1],1,0,0,0,-X_ref[i][0]*X_img[i][0],-X_ref[i][0]*X_img[i][1], -X_ref[i][0]]
    y = [0,0,0,X_img[i][0],X_img[i][1],1,-X_ref[i][1]*X_img[i][0],-X_ref[i][1]*X_img[i][1], -X_ref[i][1]]
    A.append(x)
    A.append(y)

u,d,v = np.linalg.svd(A)
h = v[-1,:]/(v[-1,-1])
H = np.reshape(h,(3,3))
im = cv2.warpPerspective(lena,H,(img.shape[1],img.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=[0,0,0,0])
cv2.imshow('d',im)
cv2.waitKey(0)


roi = img[439:570,795:948]
#cv2.imwrite("roi.jpg", roi)
#tag =img[corner[0][1]:corner[3][1],corner[0][0]:corner[3][0]]
#cv2.imshow('tag',tag)
#cv2.waitKey(0)





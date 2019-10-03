# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 19:47:56 2019

@author: Sneha
"""
img = cv2.imread('2.jpg')
blur = cv2.GaussianBlur(img,(5,5),0)
imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


edged = cv2.Canny(imgray,100,200)
cv2.imshow('edges',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
( cnts, _) = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
im = cv2.drawContours(edged, cnts, -1, (0,255,0), 3)
cv2.imshow('contours',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
im = cv2.dilate(im,None)
im = cv2.blur(im,(9,9))

corners = cv2.goodFeaturesToTrack(im,4,0.1,60)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,225,-1)


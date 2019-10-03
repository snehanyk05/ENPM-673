# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:54:07 2019

@author: Sneha
"""

import cv2
import numpy as np

img = cv2.imread('frame400.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
  
dst = cv2.cornerHarris(gray,2,3,0.04)
#ret,binary = cv2.threshold(dst, 120, 255, cv2.THRESH_BINARY);
        #result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)


#
#cv2.imshow('edges',binary)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#  for i in dst:

  
        # Threshold for an optimal value, it may vary depending on the image.
img[dst>0.21*dst.max()]=[0,0,255]
#pix = img.load()
#coord = np.where(np.all(img == (0, 255, 0), axis=-1))
#print (coord)
width, height = img.shape[:2]
for i in range(0,width):
   for j in range(0,height):
      px = img[i,j]
      if(px==(0,0,255)):
          print (i,j)
      
 
#x = 50
#y = 50
#rad = 20
#cv2.circle(raw_img,(x,y),rad,(0,255,0),-1)
#
## Here is where you can obtain the coordinate you are looking for
#combined = raw_img[:,:,0] + raw_img[:,:,1] + raw_img[:,:,2]
#rows, cols, channel = np.where(combined > 0) 
cv2.imshow('edges',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


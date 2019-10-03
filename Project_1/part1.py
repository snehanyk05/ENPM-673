# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

filename = 'ref_marker_grid.png'
img = cv2.imread(filename)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#gray = np.float32(gray)
#dst = cv2.cornerHarris(gray,2,3,0.04)
#
##result is dilated for marking the corners, not important
#dst = cv2.dilate(dst,None)
#
## Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]
#
#cv2.imshow('dst',img)
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()
from PIL import Image

im = Image.open(filename)
width, height = im.size

unit_h=width/8;
unit_w=height/8;
quad_w=int(4*(unit_w))
quad_h=int(4*(unit_h))
add_w=width+quad_h
img = cv2.rectangle(img,(quad_w,quad_h),(add_w,add_w),(0,255,0),3)
cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
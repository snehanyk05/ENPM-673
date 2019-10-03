# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:32:49 2019

@author: Sneha
"""
import numpy as np
import cv2
import time
major_ver=(cv2.__version__)


vidcap = cv2.VideoCapture('Tag0.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
 
    # Number of frames to capture
num_frames = 100;
     
     
print ("Capturing {0} frames".format(num_frames))
start = time.time()
for i in range(0, num_frames) :
     ret, frame = vidcap.read()
 
     
    # End time
end = time.time()
 
    # Time elapsed
seconds = end - start
print ("Time taken : {0} seconds".format(seconds))
 
    # Calculate frames per second
fps  = num_frames / seconds;
print ("Estimated frames per second : {0}".format(fps))
success,image = vidcap.read()
count = 0
success = True
while count<300 and success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

img=cv2.imread("frame0.jpg")
height , width , layers =  img.shape

video = cv2.VideoWriter('video.mp4',-1,fps-60,(width,height))
for i in range(0,count):
  img=cv2.imread("frame%d.jpg" % i);
     
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray,2,3,0.04)
        
        #result is dilated for marking the corners, not important
  dst = cv2.dilate(dst,None)
        
        # Threshold for an optimal value, it may vary depending on the image.
  img[dst>0.2*dst.max()]=[0,0,255]
  video.write(img)
  

cv2.destroyAllWindows()
#cap = cv2.VideoCapture(1)


video.release()
